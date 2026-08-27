# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""对 `yolo` CLI 进行蒙特卡洛模糊测试，以发现有限测试矩阵之外的缺陷。.

在子进程中运行随机化或变异后的 `yolo` 命令，并对每个结果进行分类（通过、预期的配置错误、
环境跳过、网络抖动、挂起、崩溃、疑似缺陷）；通过重放命令确认疑似缺陷，并为每个分片输出
JSONL 试验日志和发现结果文件。仅依赖标准库的 `report` 子命令会在 CI 中汇总各分片的发现结果，
根据稳定签名与现有 GitHub issue 去重，并且每次运行最多新建 `--max-issues` 个 issue。

子命令：
    fuzz    运行限定预算的模糊测试循环（会导入 ultralytics）。
    repro   多次重放一条精确命令并打印分类结果（会导入 ultralytics）。
    report  汇总分片发现并通过 `gh` 创建 GitHub issue（仅使用标准库，不导入 ultralytics）。

用法：
    python .github/scripts/fuzz.py fuzz --budget-minutes 300 --seed 123 --personality chaos --out fuzz-out
    python .github/scripts/fuzz.py repro "train detect model=yolo26n.pt data=coco8.yaml epochs=abc imgsz=32"
    python .github/scripts/fuzz.py report --in fuzz-out --max-issues 3 --dry-run
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PureWindowsPath

# Linux CPU runner 上各模式的子进程超时（秒），约为观测正常值的 6～10 倍余量
# Windows runner 慢约 2 倍（解释器启动和文件系统开销），因此所有超时在此按比例调整
TIMEOUT_SCALE = 2 if os.name == "nt" else 1
MODE_TIMEOUTS = {"train": 360, "val": 180, "predict": 180, "track": 240, "export": 480}
CONFIRM_TIMEOUT = 180  # 确认挂起时使用较短的二次超时（不重复支付完整超时时间）
MAX_HANG_CONFIRMS = 5  # 限制每个分片的挂起确认次数，避免单个异常类别耗尽预算
MIN_FREE_GB = 5  # 可用磁盘低于此值时优雅地停止模糊测试
CANARY_FAIL_FRACTION = 0.2  # 未变异的已知正常语料失败率超过 20% 时，将分片标记为基础设施失败

MODES = ["train", "val", "predict", "track", "export"]  # benchmark 会吞掉异常；solutions 暂不处理
PERSONALITIES = {  # 仅用于模式选择的权重；所有分片使用相同的变异内核和分类器
    "train": {"train": 0.7, "val": 0.075, "predict": 0.075, "track": 0.075, "export": 0.075},
    "export": {"train": 0.075, "val": 0.075, "predict": 0.075, "track": 0.075, "export": 0.7},
    "predict-val": {"train": 0.05, "val": 0.35, "predict": 0.35, "track": 0.2, "export": 0.05},
    "chaos": {m: 0.2 for m in MODES},
}
STRATEGY_WEIGHTS = [
    ("invalid", 0.3),
    ("combo", 0.3),
    ("malformed", 0.1),
    ("model", 0.1),
    ("source", 0.1),
    ("dataset", 0.1),
]
STRATEGY_MODES = {"source": {"predict", "track"}, "dataset": {"train", "val"}}  # 策略仅限于部分模式
RESAMPLE_ATTEMPTS = 20  # 接受重复命令前，允许抽取命令并查找近期未执行过的命令
HISTORY_DAYS = 7  # 历史记录超过此天数后重新探索命令，从而重新抽样回归问题

# 成本/风险键固定为经过限制的已知正常值，不参与变异（`time` 是以小时计的训练时长）
NEVER_MUTATE = frozenset(
    {
        "model",
        "data",
        "epochs",
        "imgsz",
        "batch",
        "workers",
        "source",
        "project",
        "name",
        "time",
        "resume",
        "cfg",
        "tracker",
        "show",
        "mode",
        "task",
        "format",
    }
)
CLAMPS = {
    "train": "imgsz=32 epochs=1 batch=4 workers=2 cache=disk",
    "val": "imgsz=32",
    "predict": "imgsz=32",
    "track": "imgsz=160",
    "export": "imgsz=32",
}
EXPORT_POOL = ["torchscript", "onnx", "openvino"]  # CPU-friendly formats installed on every shard

# 其他遵循常规 CLI 约定的预训练模型系列。限制模式范围，以避免仅提示训练路径。
# 最后一个字段会覆盖 CLAMPS：RT-DETR 的 300 查询解码器需要至少 160 像素的锚点（低于该值属于 T2 缺口）。
ALTERNATE_CORPUS = (
    ("detect", "rtdetr-l.pt", "coco8.yaml", {"val", "predict", "export"}, "imgsz=160"),
    ("detect", "yolov8s-worldv2.pt", "coco8.yaml", {"predict", "export"}, ""),
    ("segment", "yoloe-11s-seg-pf.pt", "coco8-seg.yaml", {"predict", "export"}, ""),
)

# 对排除在任意变异之外的成本敏感键执行受控变化。
SAFE_BOUNDARIES = {
    "train": ["imgsz=48", "imgsz=64", "batch=1", "batch=2", "workers=0", "workers=1"],
    "val": ["imgsz=48", "imgsz=64", "batch=1", "batch=2", "workers=0", "workers=1"],
    "predict": ["imgsz=48", "imgsz=64"],
    "track": ["imgsz=128", "imgsz=192", "vid_stride=2"],
    "export": ["imgsz=48", "imgsz=64", "batch=2"],
}

# 探测池："valid" 值是受支持的输入（深层失败属于 T1 缺陷）；"invalid" 值是 cfg 层应当拒绝的输入，
# 无论是当前检查拒绝，还是由于缺少范围检查而未拒绝，深层失败都属于 T2 验证缺口。
# 标签表达的是该意图，而不是 check_cfg 当前恰好接受的内容（例如它会放过负整数）。
ENUM_POOLS = {
    "optimizer": {
        "valid": ["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto", "sgd"],  # 大小写会被规范化
        "invalid": ["Ranger", "", "none"],
    },
    "split": {"valid": ["val", "test", "train"], "invalid": ["trainval", "", "0.5"]},
    "cache": {"valid": ["True", "False", "ram", "disk"], "invalid": ["gpu", "1.5"]},
    "compile": {"valid": ["True", "False", "default", "reduce-overhead", "max-autotune"], "invalid": ["turbo"]},
    "auto_augment": {"valid": ["randaugment", "autoaugment", "augmix"], "invalid": ["randaug", ""]},
    "copy_paste_mode": {"valid": ["flip", "mixup"], "invalid": ["paste", ""]},
    "quantize": {"valid": ["fp16", "w8a8", "none"], "invalid": ["half", "int8_dynamic", "int4"]},
    # CPU runner 会使所有加速器请求无效；`mps` 在 macOS 分片上确实有效，因此其失败归为 T2 而不是 T1。
    # 说明：select_device 是被测试的层，而不是用于判定等级的依据。
    "device": {"valid": ["cpu"], "invalid": ["0,1", "cuda:5", "mps", "-1", "gpu0", "0"]},
}
PROBES = {  # 按类型键族划分的边界值和错误类型探测（值为 CLI 字符串）
    "fraction": {"valid": ["0.0", "1.0", "0.5"], "invalid": ["-0.1", "1.5", "half", "True", "none"]},
    "int": {"valid": ["0", "1", "7"], "invalid": ["-1", "3.5", "ten", "none"]},
    "bool": {"valid": ["True", "False"], "invalid": ["yes", "1", "none"]},
    "float": {"valid": ["0.0", "0.1", "10"], "invalid": ["-5", "big", "none"]},
}
CHAOS_PROBES = ["[]", "[1,2]", "{}", "🚀", "1e309", "nan", "-0"]  # 混沌分片对任意键添加的额外值，全部无效

# 数据集变异 -> 检查其内容是否受支持。`data` 仍保持固定，因此数据集加载（根据软件包遥测，
# 这是受影响用户最多的错误面）完全未参与模糊测试。仅背景、灰度、webp 和 CRLF 输入均受支持，
# 因此这些输入的深层失败属于 T1 缺陷，而不是验证缺口。
DATASET_MUTATIONS = {
    "baseline": True,
    "background-only": True,
    "mixed-background": True,
    "grayscale": True,
    "webp": True,
    "crlf-labels": True,
    "duplicate-rows": True,
    "tiny-boxes": True,
    "single-image": True,
    "class-index-oob": False,
    "coords-out-of-range": False,
    "negative-coords": False,
    "wrong-columns": False,
    "nonnumeric": False,
    "tiny-image": False,
    "missing-val-key": False,
    "missing-val-dir": False,
    "nc-names-mismatch": False,
    "bad-yaml": False,
    "empty-train-dir": False,
}

# 有效但少见的组合（模式、额外参数）——T1 语义缺陷通常位于此处
COMBO_POOL = [
    ("train", "rect=True"),
    ("train", "single_cls=True"),
    ("train", "cos_lr=True close_mosaic=0"),
    ("train", "fraction=0.5"),
    ("train", "freeze=10"),
    ("train", "multi_scale=0.5"),
    ("train", "optimizer=NAdam warmup_epochs=0"),
    ("train", "deterministic=False seed=7"),
    ("train", "overlap_mask=False mask_ratio=1"),
    ("train", "mosaic=0 mixup=1.0 cutmix=1.0"),
    ("train", "copy_paste=1.0 copy_paste_mode=mixup"),
    ("train", "hsv_h=1.0 hsv_s=0.0 degrees=180 perspective=0.001"),
    ("val", "save_json=True"),
    ("val", "split=train"),
    ("val", "end2end=True max_det=5"),
    ("val", "agnostic_nms=True conf=0.9 iou=0.1"),
    ("predict", "save_txt=True save_conf=True save_crop=True"),
    ("predict", "visualize=True"),
    ("predict", "augment=True"),
    ("predict", "classes=0"),
    ("predict", "classes=[0,2]"),
    ("predict", "retina_masks=True"),
    ("predict", "line_width=1 show_labels=False show_conf=False"),
    ("predict", "vid_stride=2 stream_buffer=True"),
    ("track", "tracker=bytetrack.yaml"),
    ("track", "tracker=botsort.yaml"),
    ("track", "tracker=fasttrack.yaml"),
    ("track", "save_txt=True save_conf=True"),
    ("track", "vid_stride=2 stream_buffer=True"),
    ("export", "dynamic=True"),
    ("export", "nms=True"),
    ("export", "simplify=False"),
    ("export", "opset=12"),
    ("export", "quantize=fp16"),
    ("export", "end2end=True max_det=10"),
]
TASK_COMBO_POOL = [
    ("train", "depth", "dlog=0.5 dgrad=1.0 dlam=0.0"),
]

# 判定器：预期的干净错误是验证层（模块或确切调用帧）抛出的以下类型
EXPECTED_TYPES = {"SyntaxError", "ValueError", "TypeError", "AssertionError", "FileNotFoundError"}
EXPECTED_MODULES = (
    "ultralytics/cfg/__init__.py",
    "ultralytics/utils/checks.py",
    "ultralytics/utils/torch_utils.py:select_device",  # 设备验证层；此处的 ValueError 是有意抛出的
    "ultralytics/data/utils.py",
    "ultralytics/data/augment.py:classify_augmentations",
    "ultralytics/data/loaders.py:__init__",  # 源加载器就是源验证层；其异常属于干净错误
    "ultralytics/data/base.py:get_img_files",  # 图像发现验证层
    "ultralytics/data/dataset.py:cache_labels",  # 抛出干净的 "No labels found" 摘要；get_labels 不会抛出
    "ultralytics/engine/trainer.py:_build_train_pipeline",  # 优化器设置前实际验证训练批次和图像尺寸
    "ultralytics/engine/model.py:_check_is_pytorch_model",  # 只会抛出有意设计的格式错误
    "ultralytics/engine/exporter.py:validate_args",  # exporter's intentional per-format argument validation
    "ultralytics/engine/exporter.py:__call__",  # intentional compat asserts; per-format bugs raise in deeper frames
    "ultralytics/nn/autobackend.py:__init__",  # 格式分发器；各后端缺陷会在更深层调用帧中抛出
    "ultralytics/nn/tasks.py:torch_safe_load",  # 检查点可读性层；加载器错误会在更深层抛出
    "ultralytics/nn/backends/onnx.py:load_model",  # 抛出有意设计的不可解析图错误
)
NETWORK_MARKERS = (  # 仅匹配特定的下载/网络特征；本地源也可能直接抛出 ConnectionError
    "urlopen error",
    "Read timed out",
    "Download failure",
    "HTTPError",
    "requests.exceptions",
    "name resolution",
    "getaddrinfo",
    "RemoteDisconnected",
    "SSLError",
)


def load_universe():
    """延迟从 ultralytics 软件包导入参数全集（为 `report` 保持在模块作用域之外）。."""
    from ultralytics.cfg import (
        CFG_BOOL_KEYS,
        CFG_FLOAT_KEYS,
        CFG_FRACTION_KEYS,
        CFG_INT_KEYS,
        TASK2DATA,
        TASK2MODEL,
        TASKS,
    )
    from ultralytics.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER

    return {
        "tasks": sorted(TASKS),
        "task2model": TASK2MODEL,
        "task2data": TASK2DATA,
        "defaults": DEFAULT_CFG_DICT,
        "fraction_keys": sorted(CFG_FRACTION_KEYS - NEVER_MUTATE),
        "int_keys": sorted(CFG_INT_KEYS - NEVER_MUTATE),
        "bool_keys": sorted(CFG_BOOL_KEYS - NEVER_MUTATE),
        "float_keys": sorted(CFG_FLOAT_KEYS - NEVER_MUTATE - CFG_FRACTION_KEYS),
        "enum_keys": sorted(set(ENUM_POOLS) - NEVER_MUTATE),
        "source": str(ASSETS / "bus.jpg"),
        "export_pool": [*EXPORT_POOL, *(["coreml"] if importlib.util.find_spec("coremltools") else [])],
        "logger": LOGGER,
    }


def precache_assets(uni):
    """将测试语料权重和数据集下载到共享缓存中（已存在时不执行任何操作）。."""
    from ultralytics.data.utils import check_cls_dataset, check_det_dataset
    from ultralytics.utils import WEIGHTS_DIR
    from ultralytics.utils.downloads import attempt_download_asset

    for task in uni["tasks"]:
        attempt_download_asset(WEIGHTS_DIR / uni["task2model"][task])
        data = uni["task2data"][task]
        check_cls_dataset(data) if str(data).startswith("imagenet") else check_det_dataset(data, autodownload=True)
    for _task, model, *_ in ALTERNATE_CORPUS:
        attempt_download_asset(WEIGHTS_DIR / model)

    prepare_sources(uni)
    prepare_datasets(uni)
    prepare_models(uni)


def prepare_models(uni):
    """根据缓存的测试语料权重，创建模型试验所需的损坏模型文件和格式错误模型文件。.

    其他试验会固定 `model`，因此即使损坏或格式错误的检查点是软件包遥测中的主要错误来源， 检查点加载仍未经过模糊测试。此处的每个文件都是不受支持的输入。
    """
    from ultralytics.utils import ASSETS, WEIGHTS_DIR
    from ultralytics.utils.downloads import attempt_download_asset

    root = WEIGHTS_DIR.parent / "fuzz-models"
    root.mkdir(parents=True, exist_ok=True)
    good = Path(attempt_download_asset(WEIGHTS_DIR / uni["task2model"]["detect"])).read_bytes()
    image = (ASSETS / "bus.jpg").read_bytes()
    blobs = {
        "truncated.pt": good[: len(good) // 2],  # "无法找到中央目录" 类型的损坏
        "empty.pt": b"",
        "random-bytes.pt": bytes(range(256)) * 64,
        "image-as-pt.pt": image,
        "garbage.onnx": b"not an onnx graph" * 100,  # right suffix, wrong contents
        "image.jpg": image,  # 将真实图像作为 model= 传入，autobackend 会因格式不符而拒绝
    }
    for name, blob in blobs.items():
        (root / name).write_bytes(blob)
    uni["models"] = [str(root / name) for name in blobs]


def prepare_datasets(uni):
    """创建数据集试验使用的合成检测数据集，每种变异对应一个目录。."""
    from ultralytics.utils import WEIGHTS_DIR

    root = WEIGHTS_DIR.parent / "fuzz-datasets"
    uni["datasets"] = [(str(make_dataset(root / name, name)), valid) for name, valid in DATASET_MUTATIONS.items()]


def make_dataset(root, mutation):
    """在 root 下构建一个合成检测数据集，并返回其 YAML 文件路径。.

    数据集完全在磁盘上生成，不需要下载；整个数据池只有几 KB 的 64 像素图像，而精选语料有数 MB。 每种变异都会复现用户实际遇到的一种故障特征。
    """
    from PIL import Image

    shutil.rmtree(root, ignore_errors=True)
    extension = "webp" if mutation == "webp" else "jpg"
    image_mode = "L" if mutation == "grayscale" else "RGB"
    size = (8, 8) if mutation == "tiny-image" else (64, 64)  # 加载器要求尺寸大于 9 像素，因此 8 像素应被拒绝
    rows = {  # label file contents; anything unlisted gets one well-formed box
        "class-index-oob": "22 0.5 0.5 0.2 0.2",  # "Label class 22 exceeds dataset class count 1"
        "coords-out-of-range": "0 1.5 0.5 0.2 0.2",
        "negative-coords": "0 -0.5 0.5 0.2 0.2",
        "wrong-columns": "0 0.5 0.5",
        "nonnumeric": "cat 0.5 0.5 0.2 0.2",
        "duplicate-rows": "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2",
        "tiny-boxes": "0 0.5 0.5 0.0001 0.0001",
        "background-only": "",
    }.get(mutation, "0 0.5 0.5 0.2 0.2")
    if mutation == "crlf-labels":
        rows = rows.replace("\n", "\r\n") + "\r\n"
    count = 1 if mutation == "single-image" else 4
    for split in ("train", "val"):
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        if mutation == "empty-train-dir" and split == "train":
            continue
        for i in range(count):
            shade = i * 60 % 256
            Image.new(image_mode, size, shade if image_mode == "L" else (shade, shade, 90)).save(
                images / f"{i}.{extension}"
            )
            # 混合背景只给奇数图像标注，而全背景批次不会命中
            (labels / f"{i}.txt").write_text("" if mutation == "mixed-background" and i % 2 == 0 else rows)
    if mutation == "missing-val-dir":
        shutil.rmtree(root / "images" / "val")
    yaml = f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: item\n"
    if mutation == "missing-val-key":
        yaml = f"path: {root}\ntrain: images/train\nnames:\n  0: item\n"
    elif mutation == "nc-names-mismatch":
        yaml = f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [a, b, c, d]\n"
    elif mutation == "bad-yaml":
        yaml = f"path: {root}\ntrain: [images/train\nval: images/val\nnames: {{0: item\n"
    (root / "data.yaml").write_text(yaml)
    return root / "data.yaml"


def prepare_sources(uni):
    """创建预测和跟踪试验使用的有效及格式错误媒体源，并将其缓存。."""
    from ultralytics.utils import ASSETS, ASSETS_URL, WEIGHTS_DIR
    from ultralytics.utils.downloads import safe_download

    source_dir = WEIGHTS_DIR.parent / "fuzz-sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    unicode_image = source_dir / "path with spaces 🚀.jpg"
    shutil.copy2(ASSETS / "bus.jpg", unicode_image)
    empty_image, corrupt_image, empty_dir = source_dir / "empty.jpg", source_dir / "corrupt.jpg", source_dir / "empty"
    empty_image.touch()
    corrupt_image.write_bytes(b"not an image")
    empty_dir.mkdir(exist_ok=True)
    video = source_dir / "decelera_portrait_min.mov"
    safe_download(f"{ASSETS_URL}/{video.name}", file=video)
    uni["sources"] = {
        "predict": [
            (str(ASSETS / "bus.jpg"), True),
            (str(ASSETS / "zidane.jpg"), True),
            (str(ASSETS), True),
            (str(unicode_image), True),
            (str(video), True),
            (str(empty_image), False),
            (str(corrupt_image), False),
            (str(empty_dir), False),
        ],
        "track": [
            (str(video), True),
            (str(unicode_image), True),
            (str(empty_image), False),
            (str(corrupt_image), False),
            (str(empty_dir), False),
        ],
    }
    uni["video"] = str(video)


def strip_defaults(pairs, defaults):
    """删除值等于软件包默认值的 k=v 参数：argv 只保留发生变化的参数。."""
    return [a for a in pairs if a.partition("=")[2].lower() != str(defaults.get(a.partition("=")[0])).lower()]


def build_corpus(uni):
    """构建已知正确的种子语料：覆盖每个任务和模式，并设置受限的快速参数及明确的本地源。."""
    corpus = []
    for task in uni["tasks"]:
        # 裸权重名称可保持问题复现命令的可移植性；它们会在预缓存的 weights_dir 中解析
        model, data = uni["task2model"][task], uni["task2data"][task]
        for mode in MODES:
            if mode == "track" and task not in {"detect", "segment", "pose", "obb"}:
                continue
            argv = [mode, task, f"model={model}", *strip_defaults(CLAMPS[mode].split(), uni["defaults"])]
            if mode in {"train", "val"}:
                argv.append(f"data={data}")
            elif mode == "predict":
                argv.append(f"source={uni['source']}")
            elif mode == "track":
                argv.append(f"source={uni['video']}")
            corpus.append({"mode": mode, "task": task, "argv": argv})  # export: default torchscript stays implicit
    for task, model, data, modes, clamp in ALTERNATE_CORPUS:
        for mode in modes:
            argv = [mode, task, f"model={model}", *strip_defaults((clamp or CLAMPS[mode]).split(), uni["defaults"])]
            if mode == "val":
                argv.append(f"data={data}")
            elif mode == "predict":
                argv.append(f"source={uni['source']}")
            pinned = {a.partition("=")[0] for a in clamp.split()}  # 被覆盖的键是该键族的下限：绝不变异
            corpus.append({"mode": mode, "task": task, "argv": argv, "pinned": pinned})
    return corpus


def sample_trial(rng, uni, corpus, personality):
    """从一种变异策略中使用规范参数采样一次试验。."""
    weights = PERSONALITIES[personality]
    mode = rng.choices(MODES, weights=[weights[m] for m in MODES])[0]
    strategies = [(s, w) for s, w in STRATEGY_WEIGHTS if mode in STRATEGY_MODES.get(s, MODES)]
    strategy = rng.choices([s for s, _ in strategies], weights=[w for _, w in strategies])[0]
    # 合成数据集采用检测格式，因此数据集试验必须从检测基础配置开始
    pool = [c for c in corpus if c["mode"] == mode and (strategy != "dataset" or c["task"] == "detect")]
    base = rng.choice(pool)
    argv, mutated = list(base["argv"]), []

    validity = {}  # key -> 其有效值是否受支持：移除的参数不产生影响，重复参数以后者为准

    def mutate(pairs, valid=True):
        """将非默认的 k=v 参数追加到 argv，并记录这些参数键已发生变异。."""
        for a in pairs:
            key, _, value = a.partition("=")
            argv[2:] = [x for x in argv[2:] if x.partition("=")[0] != key]  # 后一个值优先；argv[:2] 是模式/任务
            changed = value.lower() != str(uni["defaults"].get(key)).lower()
            if changed:
                argv.append(a)
            if key not in mutated:
                mutated.append(key)
            validity[key] = valid or not changed  # 被移除的默认值参数仍会留下受支持的有效值

    def mutate_boundary():
        """在安全范围内改变一个成本敏感参数，同时遵守基础条目固定的参数族下限。."""
        pool = [b for b in SAFE_BOUNDARIES[mode] if b.partition("=")[0] not in base.get("pinned", ())]
        if pool:
            mutate([rng.choice(pool)])

    def mutate_combos(max_groups=4):
        """组合兼容的模式专用参数组，且不重复使用参数键。."""
        options = [c.split() for m, c in COMBO_POOL if m == mode]
        rng.shuffle(options)
        task_options = [c.split() for m, task, c in TASK_COMBO_POOL if m == mode and task == base["task"]]
        options = task_options + options
        used, target = set(), rng.randint(1, max_groups)
        for combo in options:
            keys = {a.partition("=")[0] for a in combo}
            if keys.isdisjoint(used):
                mutate(combo)
                used.update(keys)
                target -= 1
            if not target:
                break

    if mode == "export":  # 从可安装池中对格式进行模糊测试；默认 torchscript 保持隐式
        mutate([f"format={rng.choice(uni['export_pool'])}"])
    if strategy == "combo":
        mutate_combos()
        mutate_boundary()
    elif strategy == "invalid":
        n_keys = rng.randint(1, 4 if personality == "chaos" else 3)
        for _ in range(n_keys):
            key, value, valid = sample_mutation(rng, uni, chaos=personality == "chaos")
            mutate([f"{key}={value}"], valid=valid)
    elif strategy == "malformed":  # 直接追加原始值：这些标记故意不是格式正确的 k=v 对
        for token in malformed_tokens(rng):
            argv.append(token)
            key = token.partition("=")[0]
            mutated.append(key)
            validity[key] = False
    elif strategy == "model":  # 其他策略都会固定 `model`；此策略负责替换它
        mutate([f"model={rng.choice(uni['models'])}"], valid=False)
    elif strategy == "dataset":  # 其他策略都会固定 `data`；此策略负责替换它
        dataset, valid = rng.choice(uni["datasets"])
        mutate([f"data={dataset}"], valid=valid)
        mutate_combos(max_groups=2)  # 与 rect/single_cls 等组合，数据集形状缺陷通常在此处暴露
        mutate_boundary()
    else:
        source, valid = rng.choice(uni["sources"][mode])
        mutate([f"source={source}"], valid=valid)
        mutate_combos(max_groups=2)
        mutate_boundary()
    return {
        "mode": mode,
        "task": base["task"],
        "argv": argv,
        "strategy": strategy,
        "mutated": mutated,
        "valid_input": all(validity.values()),
    }


def probe_supported(key, value):
    """记录在受限模糊测试语料上会退化的有效探测值范围下限。."""
    if key == "fraction":  # 低于 0.25 时，4 张图像的 coco8 训练划分会选不到任何图像
        return float(value) >= 0.25
    if key == "mask_ratio":  # 掩码下采样比例必须保持在受限训练图像尺寸范围内
        return 1 <= float(value) <= 16
    return True


def sample_mutation(rng, uni, chaos=False):
    """选择一个可模糊测试的参数键、探测值，以及该值是否属于该键文档声明的有效值。."""
    family = rng.choices(["enum", "fraction", "int", "bool", "float"], weights=[4, 2, 2, 1, 1])[0]
    if family == "enum":
        key = rng.choice(uni["enum_keys"])
        pool = ENUM_POOLS[key]
    else:
        key = rng.choice(uni[f"{family}_keys"])
        pool = PROBES[family]
    if family in {"fraction", "int", "float"} and rng.random() < 0.5:
        valid = rng.random() < 0.5
        if family == "fraction":
            value = f"{rng.uniform(0.000001, 0.999999):.8g}" if valid else f"{rng.uniform(1.000001, 4):.8g}"
        elif family == "int":
            value = str(rng.randint(1, 4096)) if valid else f"{rng.randint(0, 4096)}.5"
        else:
            value = f"{rng.uniform(0, 180):.8g}" if valid else rng.choice(["nan", "1e309"])
        return key, value, valid and probe_supported(key, value)
    value = rng.choice(pool["valid"] + pool["invalid"] + (CHAOS_PROBES if chaos else []))
    return key, value, value in pool["valid"] and probe_supported(key, value)


def malformed_tokens(rng):
    """按用户实际误输 `yolo` 命令的方式，构建词法层面格式错误的 CLI 标记。.

    仅改变值不会产生格式错误的 *标记*，因此参数解析器只会看到包含已知键的规范 `k=v` 对。 此处每个变异核都对应软件包自身遥测中的真实特征；按用户数量统计，`ultralytics.cfg:entrypoint`
    是主要错误入口之一。所有这些输入都应在 cfg 层干净地抛出 SyntaxError 或 ValueError； 如果错误出现在更深层，则说明验证存在缺口。
    """
    key = rng.choice(["data", "epochs", "imgsz", "conf", "model", "source"])
    return rng.choice(
        [
            [key],  # "'data' 是有效的 YOLO 参数，但缺少 '=' 符号"
            [f"mode={rng.choice(['checks', 'settings', 'help', 'traln'])}"],
            [f"task={rng.choice(['detection', 'segmentaion', 'classify_', ''])}"],
            [rng.choice(["yolo", "epochs10", "?", "coco8"])],  # 完全不是参数的裸单词
            [f"{rng.choice(['--', '-'])}{key}", "1"],  # yolo CLI 不使用的 argparse 风格标志
            [f"{key}="],
            [f"{key}==1"],
            ["classes=[0,", "2]"],  # 未加引号的列表被 shell 拆分为两个标记
            [f"{rng.choice(['epocs', 'imgz', 'batchsize', 'devise'])}=1"],  # 近似正确的键：触发“你是否想输入”路径
        ]
    )


def run_trial(trial, timeout=None):
    """在隔离的临时工作目录中执行一次试验；输出写入每次试验独立的 `project=`，资源保持共享。."""
    mode = trial["mode"]
    timeout = (timeout or MODE_TIMEOUTS[mode]) * TIMEOUT_SCALE
    workdir = Path(tempfile.mkdtemp(prefix="fuzz-trial-"))
    argv = list(trial["argv"])
    if mode == "export":  # 导出结果写在模型文件旁边：复制权重，以保持共享 weights_dir 干净
        src = Path(next(a for a in argv if a.startswith("model=")).split("=", 1)[1])
        if not src.exists():  # 裸权重名称：通过软件包下载器解析（预缓存中存在时不执行操作）
            from ultralytics.utils import WEIGHTS_DIR
            from ultralytics.utils.downloads import attempt_download_asset

            src = Path(attempt_download_asset(WEIGHTS_DIR / src.name))
        local = workdir / src.name
        shutil.copy2(src, local)
        argv = [a if not a.startswith("model=") else f"model={local}" for a in argv]
    else:
        argv.append(f"project={workdir / 'runs'}")
    from ultralytics.cfg import _YOLO_CLI_COMMAND  # trainer/tuner 重启 CLI 时使用的相同调用命令

    cmd = [*_YOLO_CLI_COMMAND, *argv]
    env = {**os.environ, "YOLO_AUTOINSTALL": "false", "PYTHONFAULTHANDLER": "1"}
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH")))
    )
    t0 = time.perf_counter()
    # 使用独立会话/进程组，使超时可以终止整个进程树（数据加载器 worker、导出转换器子进程）
    group = (
        {"start_new_session": True} if os.name == "posix" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    )
    proc = subprocess.Popen(
        cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, **group
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
        _, stderr = proc.communicate()
        rc, stderr = "timeout", stderr or ""
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return rc, stderr, round(time.perf_counter() - t0, 1)


def parse_traceback(stderr):
    """将 stderr 中最后一个回溯块解析为（异常类型，[以 path:function 表示的 ultralytics 调用帧]）。."""
    blocks = stderr.split("Traceback (most recent call last):")
    if len(blocks) < 2:
        return None, []
    block = blocks[-1]
    frames = []
    for path, func in re.findall(r'File "([^"]+)", line \d+, in (\S+)', block):
        _, marker, frame = path.replace("\\", "/").rpartition("ultralytics/")
        if marker and "/site-packages/" not in frame:
            frames.append(f"ultralytics/{frame}:{func}")
    exc = None
    for line in reversed(block.strip().splitlines()):
        m = re.match(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit|SyntaxError))\b", line.strip())
        if m:
            exc = m.group(1).split(".")[-1]
            break
    return exc, frames


def make_signature(exc, frames, mode, task):
    """构建稳定的去重签名：异常类型 + 最深层 ultralytics 调用帧 + 其调用方（不含行号）。."""
    deepest = frames[-1] if frames else f"{mode}:{task}"
    caller = frames[-2] if len(frames) > 1 else ""
    human = f"{exc or 'Unknown'} in {deepest}"
    return hashlib.sha256(f"{exc}|{deepest}|{caller}".encode()).hexdigest()[:12], human


def classify(trial, rc, stderr):
    """将一次试验结果分类为 pass/expected/env-skip/flake/timeout/crash/bug-candidate。."""
    if rc == 0:
        return "pass", None, None
    if rc == "timeout":
        keys = set(trial.get("mutated", []))  # exact mutated k=v pairs: distinct hangs get distinct signatures
        mutated = "|".join(sorted(a for a in trial["argv"] if a.partition("=")[0] in keys)) or "baseline"
        sig = hashlib.sha256(f"Timeout|{trial['mode']}|{trial['task']}|{mutated}".encode()).hexdigest()[:12]
        return "timeout", sig, f"Timeout in yolo {trial['mode']} ({trial['task']})"
    exc, frames = parse_traceback(stderr)
    if isinstance(rc, int) and rc < 0:
        sig, human = make_signature(f"Signal{-rc}", frames, trial["mode"], trial["task"])
        return "crash", sig, human
    missing = re.search(r"No module named '(\w+)", stderr)
    if missing and not importlib.util.find_spec(missing.group(1)):  # module genuinely absent: optional-dep skip
        return "env-skip", None, None
    if any(marker in stderr for marker in NETWORK_MARKERS):
        return "flake", None, None
    if trial.get("mutated") and (
        # 有意设计的不支持选项错误属于预期结果；抽象的“未实现”缺口保留其特征
        (exc == "NotImplementedError" and re.search(r"not supported|(?:doesn't|does not) support", stderr))
        or (exc == "NotImplementedError" and "not found in list of available optimizers" in stderr)
        or (exc == "ValueError" and "Expected `mode` to be `flip` or `mixup`" in stderr)
        or (exc == "AssertionError" and "RTDETR export requires opset>=16" in stderr)
        # 两个数据集验证层都会汇总为 RuntimeError，因此最深调用帧是包装器，而不是 data/utils.py：
        # get_dataset 会重新抛出 YAML 错误，get_labels 会在标签缓存存在后报告每个文件的原因
        # （首次未缓存的试验则会从 cache_labels 抛出 ValueError）。仅对故意提供不受支持数据集的试验豁免：
        # 受支持数据集在此处失败，或格式错误的数据集在其他位置失败，仍然需要报告。
        or (
            exc == "RuntimeError"
            and trial.get("strategy") == "dataset"
            and not trial.get("valid_input", True)
            and frames
            and frames[-1] in {"ultralytics/engine/trainer.py:get_dataset", "ultralytics/data/dataset.py:get_labels"}
        )
        or (exc in EXPECTED_TYPES and frames and frames[-1].startswith(EXPECTED_MODULES))
    ):
        return "expected", None, None  # 只有实际发生变异的试验才应出现预期的干净验证错误
    sig, human = make_signature(exc, frames, trial["mode"], trial["task"])
    return "bug-candidate", sig, human


def stderr_tail(stderr, lines=30):
    """返回 stderr 中最后几行有意义的内容，用于日志和 issue 正文。."""
    return "\n".join(stderr.strip().splitlines()[-lines:])


def command_hash(argv):
    """返回一条精确命令的稳定短哈希，用于在单次运行内及跨运行去重抽样结果。."""
    return hashlib.sha256("\x00".join(argv).encode()).hexdigest()[:16]


def read_history(path, max_age_days):
    """加载之前运行生成的 `hash day` 行，并丢弃早于 max_age_days 的条目。.

    过期机制能让探索结果真实反映回归问题：仓库变化很快，一周前通过的命令并不能说明今天的代码没有问题。 逐条使记录过期可以形成滚动窗口，而不是一次性断崖式清除；每天淘汰最旧的一天，约每周重新尝试每条命令，
    从而避免永久排除导致已覆盖区域的回归问题永远不会再次采样。
    """
    if not path or not Path(path).exists():
        return []
    cutoff = int(time.time() // 86400) - max_age_days
    lines = (line.partition(" ") for line in Path(path).read_text().splitlines())
    return [f"{key} {day}" for key, _, day in lines if key and day.isdigit() and int(day) >= cutoff]


def write_history(path, history, new_keys):
    """保存本次运行新探索的哈希，并标记为当天的日期，以便后续运行使其自然过期。."""
    if path:
        today = int(time.time() // 86400)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join([*history, *(f"{key} {today}" for key in new_keys)]))


def cmd_fuzz(args):
    """运行限额模糊测试循环，并为报告任务写入试验 JSONL 和发现结果 JSON。."""
    uni = load_universe()
    log = uni["logger"]
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials_path = out / f"trials-{args.personality}.jsonl"
    log.info(f"[fuzz] personality={args.personality} seed={args.seed} budget={args.budget_minutes}min")
    from ultralytics.utils.checks import collect_system_info

    environment = {k: str(v) for k, v in collect_system_info().items()}  # 模糊测试前快速失败，避免丢失试验记录
    precache_assets(uni)
    corpus = build_corpus(uni)

    deadline = time.time() + args.budget_minutes * 60
    counters = {k: 0 for k in ("pass", "expected", "env-skip", "flake", "timeout", "crash", "bug-candidate")}
    findings, seen, canary_results, hang_confirms, n = {}, set(), [], 0, 0

    def execute(trial, canary=False):
        nonlocal n, hang_confirms
        rc, stderr, duration = run_trial(trial, timeout=args.debug_timeout)
        outcome, sig, human = classify(trial, rc, stderr)
        if outcome == "flake":  # 网络抖动重试一次，随后丢弃
            rc, stderr, duration = run_trial(trial, timeout=args.debug_timeout)
            outcome, sig, human = classify(trial, rc, stderr)
        confirmed = False
        tier = "T2" if outcome == "bug-candidate" and not trial.get("valid_input", True) else "T1"

        def finding():
            """根据本次出现的新试验、结果和回溯信息构建发现记录。."""
            return {
                "signature": sig,
                "title": human,
                "tier": tier,
                "outcome": outcome,
                "mode": trial["mode"],
                "task": trial["task"],
                "strategy": trial.get("strategy", "corpus"),
                "command": "yolo " + shlex.join(trial["argv"]),
                "stderr_tail": stderr_tail(stderr),
                "duration_s": duration,
            }

        if sig and sig in findings and tier == "T1" and findings[sig]["tier"] == "T2":
            # 有效输入的出现证明已确认的验证缺口特征是真实缺陷：先确认，再升级等级
            rc2, stderr2, _ = run_trial(trial, timeout=args.debug_timeout)
            outcome2, sig2, _ = classify(trial, rc2, stderr2)
            if outcome2 == outcome and sig2 == sig:
                findings[sig] = finding()
        if sig and sig not in seen:  # 每个特征只确认第一次出现
            seen.add(sig)
            if outcome == "timeout":
                if hang_confirms < MAX_HANG_CONFIRMS:
                    hang_confirms += 1
                    rc2, _, _d2 = run_trial(trial, timeout=CONFIRM_TIMEOUT)
                    confirmed = rc2 == "timeout"
                    if not confirmed:
                        outcome = "pass" if rc2 == 0 else outcome  # 重放很快完成：不是挂起
            else:
                rc2, stderr2, _ = run_trial(trial, timeout=args.debug_timeout)
                outcome2, sig2, _ = classify(trial, rc2, stderr2)
                confirmed = outcome2 == outcome and sig2 == sig  # 相同失败，而不只是相同类别
            if not confirmed:  # 抖动或达到上限的确认仍可能是真实失败：允许后续出现时重试
                seen.discard(sig)
            if confirmed:
                findings[sig] = finding()
        counters[outcome] += 1
        if canary:
            canary_results.append(outcome == "pass")
        n += 1
        with trials_path.open("a") as f:
            record = {
                "n": n,
                "outcome": outcome,
                "duration_s": duration,
                "signature": sig,
                "canary": canary,
                **{k: trial[k] for k in ("mode", "task", "argv")},
                "strategy": trial.get("strategy", "corpus"),
            }
            f.write(json.dumps(record) + "\n")
        changed = " ".join(a for a in trial["argv"] if a.partition("=")[0] in set(trial.get("mutated", [])))
        log.info(f"[fuzz] #{n} {outcome:>13} {duration:6.1f}s  yolo {trial['mode']} {trial['task']} {changed}".rstrip())

    history = read_history(args.history, HISTORY_DAYS)  # 近期运行的命令，使每次运行都探索新的内容
    explored = {line.partition(" ")[0] for line in history}
    new_keys, duplicate_samples, saturated_samples = [], 0, 0
    log.info(f"[fuzz] history: {len(history)} commands explored in the last {HISTORY_DAYS} days")
    for base in corpus:  # 先运行金丝雀：未变异语料必须通过，否则说明环境本身已损坏
        if time.time() > deadline or (args.max_trials and n >= args.max_trials):
            break
        explored.add(command_hash(base["argv"]))
        execute(dict(base), canary=True)
    while time.time() < deadline and (not args.max_trials or n < args.max_trials):
        if shutil.disk_usage(tempfile.gettempdir()).free < MIN_FREE_GB * 1024**3:
            log.warning(f"[fuzz] stopping early: <{MIN_FREE_GB}GB free disk")
            break
        for _ in range(RESAMPLE_ATTEMPTS):  # 重新抽取，直到命令此前从未被任何运行执行过
            trial = sample_trial(rng, uni, corpus, args.personality)
            key = command_hash(trial["argv"])
            if key not in explored:
                break
            duplicate_samples += 1
        if key in explored:  # 此人格可达空间已饱和：运行重复项，而不是继续空转
            saturated_samples += 1
        else:
            explored.add(key)
            new_keys.append(key)
        execute(trial)
    write_history(args.history, history, new_keys)

    infra_failed = bool(canary_results) and (canary_results.count(False) / len(canary_results)) > CANARY_FAIL_FRACTION
    summary = {
        "personality": args.personality,
        "seed": args.seed,
        "trials": n,
        "unique_commands": len(new_keys),  # 在当前运行或历史窗口内的任何运行中都未执行
        "duplicate_samples": duplicate_samples,
        "saturated_samples": saturated_samples,  # RESAMPLE_ATTEMPTS 次尝试后仍重复近期命令的抽取次数
        "history_size": len(history) + len(new_keys),
        "counters": counters,
        "infra_failed": infra_failed,
        "findings": sorted(findings.values(), key=lambda x: (x["tier"], x["signature"])),
        "environment": environment,
    }
    (out / f"findings-{args.personality}.json").write_text(json.dumps(summary, indent=2))
    log.info(
        f"[fuzz] done: {n} trials {counters}, {len(findings)} confirmed unique findings, infra_failed={infra_failed}"
    )


def cmd_repro(args):
    """通过分类器多次重放一条精确命令，并打印判定结果。."""
    uni = load_universe()
    log = uni["logger"]
    argv = shlex.split(args.command)
    if argv and argv[0] == "yolo":  # issue 正文会引用完整的 `yolo ...` 命令；按原样接受
        argv = argv[1:]

    def portable(arg):
        """将 issue 命令中的运行器本地绝对模型、源和数据路径重新映射到本机副本。."""
        k, _, v = arg.partition("=")
        if k in {"model", "source", "data"} and ("/" in v or "\\" in v) and not Path(v).exists():
            from ultralytics.utils import ASSETS, WEIGHTS_DIR

            # PureWindowsPath 会按两种分隔符拆分，因此来自 Windows 的 issue 命令可在任意操作系统上重新映射。
            if k == "model":
                prepare_models(uni)  # 损坏的模糊测试模型位于 weights 目录旁，而不是目录内部
                candidates = [WEIGHTS_DIR / PureWindowsPath(v).name, *(Path(p) for p in uni["models"])]
            elif k == "data":  # 每个合成数据集 YAML 都是 data.yaml，因此变异目录可以标识它
                prepare_datasets(uni)
                mutation = PureWindowsPath(v).parent.name
                candidates = [Path(p) for p, _valid in uni["datasets"] if Path(p).parent.name == mutation]
            else:
                prepare_sources(uni)
                candidates = [ASSETS, ASSETS / PureWindowsPath(v).name]
                candidates.extend(Path(p) for pool in uni["sources"].values() for p, _valid in pool)
            if local := next((p for p in candidates if p.name == PureWindowsPath(v).name and p.exists()), None):
                return f"{k}={local}"
        return arg

    argv = [portable(a) for a in argv]
    mode = next((a for a in argv if a in MODES), "predict")
    task = next((a for a in argv if a in uni["tasks"]), "detect")
    trial = {"mode": mode, "task": task, "argv": argv, "mutated": ["repro"]}  # 重放的命令已经经过模糊变异
    outcomes = []
    for i in range(args.runs):
        rc, stderr, duration = run_trial(trial, timeout=args.debug_timeout)
        outcome, _sig, human = classify(trial, rc, stderr)
        outcomes.append(outcome)
        log.info(f"[repro] run {i + 1}/{args.runs}: {outcome} ({duration}s) {human or ''}")
        if outcome not in {"pass", "expected"}:
            log.info(stderr_tail(stderr))
    reproduces = all(o == outcomes[0] for o in outcomes) and outcomes[0] not in {"pass", "expected"}
    log.info(f"[repro] verdict: {'REPRODUCES as ' + outcomes[0] if reproduces else 'does not reproduce'}")
    sys.exit(1 if reproduces else 0)


def gh(*cli_args, dry_run=False):
    """运行一条 `gh` CLI 命令并返回标准输出（dry_run 时改为打印命令）。."""
    if dry_run:
        print(f"[dry-run] gh {' '.join(cli_args)}")
        return ""
    return subprocess.run(["gh", *cli_args], capture_output=True, text=True, check=True).stdout


def cmd_report(args):
    """汇总分片发现，根据签名与现有 issue 去重，并且最多创建 --max-issues 个 issue。."""
    in_dir = Path(args.in_dir)
    shards = [json.loads(p.read_text()) for p in sorted(in_dir.glob("findings-*.json"))]
    if not shards:
        print("[report] no findings files found")
        return
    run_url = f"https://github.com/{args.repo}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    findings, counters, flagged = {}, {}, []
    unique_commands = duplicate_samples = saturated_samples = history_size = 0
    for shard in shards:
        for k, v in shard["counters"].items():
            counters[k] = counters.get(k, 0) + v
        if shard["infra_failed"]:  # 仅警告：导致金丝雀失败的回归本身就是发现，绝不能丢弃
            flagged.append(shard["personality"])
        unique_commands += shard.get("unique_commands", shard["trials"])
        duplicate_samples += shard.get("duplicate_samples", 0)
        saturated_samples += shard.get("saturated_samples", 0)
        history_size += shard.get("history_size", 0)
        for f in shard["findings"]:
            prev = findings.get(f["signature"])
            if not prev or (prev["tier"] == "T2" and f["tier"] == "T1"):  # 对共享特征优先采用 T1 视图
                findings[f["signature"]] = {**f, "environment": shard["environment"], "seed": shard["seed"]}

    existing = {}
    if not args.dry_run:
        gh(
            "label",
            "create",
            "fuzz",
            "--description",
            "Found by the scheduled Fuzz workflow",
            "--color",
            "8B5CF6",
            "--repo",
            args.repo,
            "--force",
        )
        issues = json.loads(
            gh(
                "issue",
                "list",
                "--repo",
                args.repo,
                "--label",
                "fuzz",
                "--state",
                "all",
                "--limit",
                "500",
                "--json",
                "number,state,title,body",
            )
            or "[]"
        )
    umbrella = None
    if not args.dry_run:
        umbrella = next((i for i in issues if i["title"].startswith("Fuzz: CLI validation gaps")), None)
        for issue in issues:
            if umbrella and issue["number"] == umbrella["number"]:
                continue  # 下面使用 setdefault 收集总括特征，因此独立 issue 始终优先
            for sig in re.findall(r"fuzz-signature: (\w+)", issue.get("body") or ""):
                existing[sig] = issue
    if umbrella:  # 之前运行产生的 T2 特征位于总括 issue 正文和评论中
        view = json.loads(gh("issue", "view", str(umbrella["number"]), "--repo", args.repo, "--json", "comments"))
        for body in [umbrella.get("body") or ""] + [c.get("body") or "" for c in view.get("comments", [])]:
            for sig in re.findall(r"fuzz-signature: (\w+)", body):
                existing.setdefault(sig, umbrella)  # 相同特征的独立 issue 优先

    def only_umbrella(s):
        """当某个签名仅作为 T2 总括条目存在时返回 True，此时再次发现 T1 结果仍应创建独立 issue。."""
        return umbrella and s in existing and existing[s]["number"] == umbrella["number"]

    new_t1 = [f for s, f in findings.items() if f["tier"] == "T1" and (s not in existing or only_umbrella(s))]
    new_t2 = [f for s, f in findings.items() if f["tier"] == "T2" and s not in existing]
    regressions = [(f, existing[s]) for s, f in findings.items() if s in existing and existing[s]["state"] == "CLOSED"]

    created = 0
    for f in new_t1:
        if created >= args.max_issues:
            print(f"[report] issue cap ({args.max_issues}) reached; {len(new_t1) - created} T1 findings deferred")
            break
        body = issue_body(f, run_url)
        title = f"yolo {f['mode']}: {f['title']}"
        print(f"[report] filing T1 issue: {title}")
        gh(
            "issue",
            "create",
            "--repo",
            args.repo,
            "--title",
            title,
            "--body",
            body,
            "--label",
            "bug,fuzz",
            dry_run=args.dry_run,
        )
        created += 1

    if new_t2:  # T2 验证缺口汇总到一个总括 issue；只有创建该 issue 才计入上限
        lines = [f"- `{f['command']}` → {f['title']} `<!-- fuzz-signature: {f['signature']} -->`" for f in new_t2]
        comment = f"New CLI validation gaps found by [fuzzing]({run_url}):\n\n" + "\n".join(lines)
        if umbrella and umbrella["state"] == "OPEN":
            gh(
                "issue",
                "comment",
                str(umbrella["number"]),
                "--repo",
                args.repo,
                "--body",
                comment,
                dry_run=args.dry_run,
            )
        elif umbrella:  # 已关闭的总括 issue 表示有意退出 T2 报告：仅生成摘要
            print(f"[report] umbrella issue #{umbrella['number']} is closed; {len(new_t2)} T2 gaps in summary only")
        elif created < args.max_issues:
            gh(
                "issue",
                "create",
                "--repo",
                args.repo,
                "--title",
                "Fuzz: CLI validation gaps (rolling)",
                "--body",
                "Deep tracebacks from invalid CLI input, found by scheduled fuzzing. Each should raise "
                "a clean error from the cfg layer instead.\n\n" + comment + "\n<!-- fuzz-signature: umbrella -->",
                "--label",
                "bug,fuzz",
                dry_run=args.dry_run,
            )
            created += 1

    by_issue = {}
    for f, issue in regressions:
        if umbrella and issue["number"] == umbrella["number"]:
            continue  # 已关闭总括 issue 中的已知 T2 特征只是去重状态，不是单个缺陷的回归信号
        by_issue.setdefault(issue["number"], []).append(f)
    for number, group in by_issue.items():  # 每次运行每个 issue 只评论一次，同一特征绝不重复评论
        commented = ""
        if not args.dry_run:
            view = json.loads(gh("issue", "view", str(number), "--repo", args.repo, "--json", "comments") or "{}")
            commented = " ".join(c.get("body") or "" for c in view.get("comments", []))
        fresh = [f for f in group if f"fuzz-regression: {f['signature']}" not in commented]
        if not fresh:
            continue
        blocks = "\n\n".join(f"```bash\n{f['command']}\n```\n<!-- fuzz-regression: {f['signature']} -->" for f in fresh)
        gh(
            "issue",
            "comment",
            str(number),
            "--repo",
            args.repo,
            "--body",
            f"Reproduced again by [fuzzing]({run_url}) after this issue was closed — possible regression.\n\n{blocks}",
            dry_run=args.dry_run,
        )

    total = sum(counters.values())
    table = ["| Outcome | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(counters.items())]
    summary = (
        f"## Fuzz — {total} trials\n\n"
        + "\n".join(table)
        + f"\n\nExploration: {unique_commands} newly explored commands · {duplicate_samples} duplicate draws"
        + f" · {saturated_samples} saturated · {history_size} in the {HISTORY_DAYS}-day history window"
        + f"\n\nNew issue threads created: {created} (cap {args.max_issues})"
        + (f" · ⚠️ shards with >20% canary failures: {', '.join(flagged)}" if flagged else "")
    )
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(step_summary).write_text(summary)
    if gh_output := os.environ.get("GITHUB_OUTPUT"):
        with Path(gh_output).open("a") as f:
            f.write(f"new_issues={created}\n")
    print(summary)


def issue_body(f, run_url):
    """为一个已确认的 T1 发现格式化 GitHub issue 正文。."""
    env = "\n".join(f"{k}: {v}" for k, v in f["environment"].items())
    return f"""Automated fuzzing found a reproducible failure (confirmed 2/2 runs).

### Reproduce

```bash
{f["command"]}
```

### Details

- Outcome: `{f["outcome"]}` · strategy: `{f["strategy"]}` · task: `{f["task"]}` · seed: `{f["seed"]}`
- Run: {run_url}

### Traceback (tail)

```
{f["stderr_tail"]}
```

### Environment

```
{env}
```

<!-- fuzz-signature: {f["signature"]} -->
"""


def main():
    """解析参数，并分发到 fuzz/repro/report 子命令。."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fuzz", help="run a budgeted fuzzing loop")
    p.add_argument("--budget-minutes", type=float, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--personality", choices=sorted(PERSONALITIES), default="chaos")
    p.add_argument("--out", default="fuzz-out")
    p.add_argument("--max-trials", type=int, default=0, help="optional hard trial cap (smoke tests)")
    p.add_argument("--history", default=None, help="file of command hashes explored by prior runs, read and updated")
    p.add_argument("--debug-timeout", type=float, default=None, help="override all trial timeouts (test hang path)")

    p = sub.add_parser("repro", help="replay one exact command and classify it")
    p.add_argument("command", help='yolo args, e.g. "train detect model=yolo26n.pt data=coco8.yaml epochs=abc"')
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--debug-timeout", type=float, default=None)

    p = sub.add_parser("report", help="aggregate shard findings and file GitHub issues (stdlib only)")
    p.add_argument("--in", dest="in_dir", default="fuzz-out")
    p.add_argument("--max-issues", type=int, default=3, help="hard cap on `gh issue create` calls per run")
    p.add_argument("--repo", default="ultralytics/ultralytics")
    p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    {"fuzz": cmd_fuzz, "repro": cmd_repro, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
