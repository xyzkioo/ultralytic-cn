# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import csv
import os
import shutil
import tarfile
import urllib
import zipfile
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

import ultralytics.data.build as data_build
from tests import CFG, MODEL, MODELS, SOURCE, SOURCES_LIST, TASK_MODEL_DATA
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, load_inference_source
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
    DATASETS_DIR,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    IS_JETSON,
    IS_RASPBERRYPI,
    LINUX,
    LOGGER,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    YAML,
    checks,
    is_github_action_running,
)
from ultralytics.utils.downloads import download, safe_download
from ultralytics.utils.torch_utils import TORCH_1_11, TORCH_1_13


def test_dataloader_caps_workers_to_batches():
    """测试极小数据集不会创建超出有效批次数量的持久 worker。."""
    single_batch = build_dataloader(range(4), batch=4, workers=8)
    drop_last_single_batch = build_dataloader(range(5), batch=4, workers=8, drop_last=True)
    two_batches = build_dataloader(range(8), batch=4, workers=8)
    try:
        assert single_batch.num_workers == 0
        assert drop_last_single_batch.num_workers == 0
        assert two_batches.num_workers <= 2
    finally:
        single_batch.close()
        drop_last_single_batch.close()
        two_batches.close()


def test_dataloader_cap_preserves_distributed_drop_last(monkeypatch):
    """测试 worker 上限会遵循分布式采样器大小，同时不改变全局 drop_last 行为。."""
    sampler_cls = data_build.distributed.DistributedSampler

    def distributed_sampler(dataset, shuffle, seed):
        return sampler_cls(dataset, num_replicas=3, rank=2, shuffle=shuffle, seed=seed)

    monkeypatch.setattr(data_build.distributed, "DistributedSampler", distributed_sampler)
    monkeypatch.setattr(data_build, "RANK", 2)  # 模拟全局 rank 为 2、本地 rank 为 0 的第二个节点
    expected_seed = torch.initial_seed() - 3
    loader = build_dataloader(range(8), batch=4, workers=8, rank=0, drop_last=True)
    try:
        assert len(loader) == 1
        assert loader.num_workers == 0
        assert loader.sampler.seed == expected_seed
    finally:
        loader.close()


def test_dataloader_seed_varies_sampling_order():
    """测试运行种子会传递给加载器随机数生成器，而不是让每次运行都重复固定顺序。."""
    with torch.random.fork_rng():
        loaders = []
        for seed in (0, 0, 1):
            torch.manual_seed(seed)
            loaders.append(build_dataloader(range(64), batch=4, workers=0))
    try:
        first, repeat, other = (torch.cat(list(loader)).tolist() for loader in loaders)
        assert first == repeat  # 相同随机种子应保持可复现
        assert first != other  # 不同随机种子不能产生相同顺序
    finally:
        for loader in loaders:
            loader.close()


def test_dataloader_empty_dataset_uses_dataloader_validation():
    """测试空数据集会通过 DataLoader 验证失败，而不是在 worker 上限计算处失败。."""
    with pytest.raises(ValueError, match="positive integer"):
        build_dataloader([], batch=4, workers=2)


def test_build_yolo_dataset_hyp_isolated():
    """测试构建数据集不会修改其所基于的共享 cfg 中的超参数。."""
    data = check_det_dataset("coco8.yaml")
    cfg = get_cfg(overrides={"data": "coco8.yaml", "imgsz": 32, "rect": True})  # rect 会将当前超参数中的 mosaic 置零
    data_build.build_yolo_dataset(cfg, data["train"], batch=2, data=data, mode="train")
    assert cfg.mosaic == DEFAULT_CFG.mosaic


def test_cfg_rejects_fuzzed_values():
    """测试无效覆盖参数会在配置验证阶段失败。."""
    with pytest.raises(TypeError, match="degrees"):
        get_cfg(overrides={"degrees": None})
    with pytest.raises(ValueError, match="cls_pw"):
        get_cfg(overrides={"cls_pw": 10})
    for key, value in (
        ("split", []),
        ("split", -0.0),
        ("optimizer", []),
        ("copy_paste_mode", {}),
        ("optimizer", None),
        ("split", None),
        ("copy_paste_mode", None),
    ):
        with pytest.raises((TypeError, ValueError), match=key):
            get_cfg(overrides={key: value})
    assert get_cfg(overrides={"auto_augment": None}).auto_augment is None


def skip_rpi_semantic():
    """由于内存限制，在 Raspberry Pi 上跳过语义分割测试。."""
    if IS_RASPBERRYPI:
        pytest.skip("Semantic segmentation tests are skipped on Raspberry Pi due to memory constraints.")


def test_select_device(monkeypatch):
    """同一个设备字符串每次调用都必须解析到同一块 GPU，且不能修改环境变量。."""
    from ultralytics.utils import torch_utils

    set_calls = []
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch_utils.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch_utils.torch.cuda, "set_device", set_calls.append)
    monkeypatch.setattr(torch_utils, "get_gpu_info", lambda i: f"Mock GPU {i}, 1MiB")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert str(torch_utils.select_device("", verbose=False)) == "cuda:0"
    assert not set_calls  # 默认 '' 请求绝不能移动当前设备，例如 check_yolo() 等诊断操作
    for _ in range(2):  # 重复调用必须保持幂等，例如 Trainer.__init__ 后调用 final_eval，或连续两次 predict()
        assert str(torch_utils.select_device("1", verbose=False)) == "cuda:1"
        with pytest.raises(ValueError):
            torch_utils.select_device("3", verbose=False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is None  # 从未写入 CUDA_VISIBLE_DEVICES
    assert set_calls == [1, 1]  # 显式单 GPU 请求会为无索引的 'cuda' 操作设置默认设备
    assert str(torch_utils.select_device("0,1", verbose=False)) == "cuda:0"
    assert set_calls == [1, 1]  # 多 GPU 请求绝不移动当前设备；DDP rank 会在 _setup_ddp 中绑定各自设备
    monkeypatch.setattr(torch_utils.torch.cuda, "current_device", lambda: 1)
    assert str(torch_utils.select_device("", verbose=False)) == "cuda:1"  # 默认 '' 解析为当前设备
    assert str(torch_utils.select_device(torch.device("cuda", 1), verbose=False)) == "cuda:1"
    with pytest.raises(ValueError):  # torch.device 输入与字符串一样进行验证，不抛出原始 CUDA 错误
        torch_utils.select_device(torch.device("cuda", 3), verbose=False)
    set_calls.clear()
    assert str(torch_utils.select_device(torch.device("cuda"), verbose=False)) == "cuda:1"
    assert not set_calls  # 无索引的 torch.device('cuda') 表示当前设备，绝不会移动设备
    assert torch_utils.parse_device([0, 1]) == "0,1"
    assert torch_utils.parse_device("00,01") == "0,1"  # 有效 torch 设备字符串会去除前导零
    assert torch_utils.parse_device(torch.device("cuda")) == ""  # 无索引的 'cuda' 保持为 '' 默认请求
    # 外部 CUDA_VISIBLE_DEVICES 限制下的物理 GPU ID 会转换为 torch 索引
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert str(torch_utils.select_device("3", verbose=False)) == "cuda:1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(torch_utils.torch.cuda, "device_count", lambda: 1)
    assert str(torch_utils.select_device("3", verbose=False)) == "cuda:0"  # 例如预设 CVD 的 Pod
    assert torch_utils.parse_device(torch_utils.parse_device("3")) == "0"  # idempotent: trainer + select_device parse
    # '-1' 空闲 GPU 自动选择只搜索外部可见 GPU，并将物理 ID 转换为 torch 索引
    from ultralytics.utils import autodevice

    monkeypatch.setattr(autodevice.GPUInfo, "__init__", lambda self: self.__dict__.update(nvml_available=False))
    monkeypatch.setattr(
        autodevice.GPUInfo,
        "select_idle_gpu",
        lambda self, count=1, indices=None, **kw: [i for i in (0, 1, 3) if i in indices][:count],
    )
    monkeypatch.setattr(torch_utils.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    assert torch_utils.parse_device("-1") == "0"  # CVD='1,3' 下空闲物理 GPU 1 是 torch 索引 0；0 不可见
    assert torch_utils.parse_device("1") == "1"  # 范围内 ID 是 torch 索引，因此重复解析结果稳定
    assert torch_utils.parse_device("-1,3") == "0,1"  # 混合请求：空闲物理 GPU 1 和物理 GPU 3 作为 torch 索引
    assert torch_utils.parse_device("0,1") == "0,1"  # 已转换的输出再次解析后保持不变（幂等）
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2,9,1")  # 格式错误：CUDA 在无效 ID 9 处停止，前缀有 2 个可用 GPU
    assert torch_utils.parse_device("9") == "9"  # 不可用 ID 不会转换，因此 select_device 会拒绝它
    assert torch_utils.parse_device("2") == "1"  # 物理 GPU 2 是可用前缀中的 torch 索引 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "01,03")  # CUDA 的 atoi 风格解析允许前导零
    assert torch_utils.parse_device("3") == "1"  # 可见 ID 与请求 ID 一样会被规范化
    assert torch_utils.parse_device("-1") == "0"  # 通过规范化后的可见 ID 找到空闲物理 GPU 1


def test_restricted_load_threaded():
    """并发受限加载会共享进程级允许列表，且不能删除彼此添加的条目。."""
    from concurrent.futures import ThreadPoolExecutor

    from ultralytics.nn.tasks import torch_safe_load

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: torch_safe_load(MODEL, safe_only=True), range(32)))


def test_model_forward():
    """测试 YOLO 模型的前向传播。."""
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # 同时测试无输入源和增强


def test_model_methods():
    """测试 YOLO 模型的各种方法和属性，确保功能正确。."""
    model = YOLO(MODEL)

    # 模型方法
    model.info(verbose=True, detailed=True)
    model = model.reset_weights()
    model = model.load(MODEL)
    model.to("cpu")
    model.fuse()
    model.clear_callback("on_train_start")
    model.reset_callbacks()

    # 模型属性
    _ = model.names
    _ = model.device
    _ = model.transforms
    _ = model.task_map


def test_model_load_remaps_cls_head_by_names():
    """测试类别名称重映射仅限于封闭集合类别 logits 检测头。."""
    from types import SimpleNamespace

    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.yolo.obb.train import OBBTrainer
    from ultralytics.models.yolo.pose.train import PoseTrainer
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.nn.tasks import DetectionModel, OBBModel, PoseModel, SegmentationModel, YOLOEModel

    src = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
    tgt = DetectionModel("yolo26n.yaml", nc=2, verbose=False)
    src.names, tgt.names = {0: "cat", 1: "dog", 2: "car"}, {0: "dog", 1: "cat"}
    for seq in src.model[-1].cv3:
        seq[-1].bias.data.copy_(torch.tensor([10.0, 20.0, 30.0]))
    tgt.load(src, verbose=False)
    assert all(seq[-1].bias.tolist() == [20.0, 10.0] for seq in tgt.model[-1].cv3)

    src = YOLOEModel("yoloe-26n.yaml", nc=3, verbose=False)
    tgt = YOLOEModel("yoloe-26n.yaml", nc=2, verbose=False)
    src.names, tgt.names = {0: "cat", 1: "dog", 2: "car"}, {0: "dog", 1: "cat"}
    tgt.load(src, verbose=False)  # YOLOE cv3 输出嵌入，而不是类别行

    names = {0: "dog", 1: "cat"}
    for trainer_cls, model in (
        (DetectionTrainer, DetectionModel("yolo26n.yaml", nc=2, verbose=False)),
        (SegmentationTrainer, SegmentationModel("yolo26n-seg.yaml", nc=2, verbose=False)),
        (PoseTrainer, PoseModel("yolo26n-pose.yaml", nc=2, data_kpt_shape=[17, 3], verbose=False)),
        (OBBTrainer, OBBModel("yolo26n-obb.yaml", nc=2, verbose=False)),
    ):
        trainer = object.__new__(trainer_cls)
        trainer.args = SimpleNamespace(cls_remap=True)
        trainer.data = {"names": names}
        assert trainer.set_model_names_for_load(model).names == names


def test_model_profile():
    """使用 `profile=True` 分析 YOLO 模型，以评估性能和资源使用情况。."""
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel()  # 构建模型
    im = torch.randn(1, 3, 64, 64)  # 要求最小 imgsz=64
    _ = model.predict(im, profile=True)


def test_predict_txt(tmp_path):
    """测试 YOLO 使用文本文件中列出的文件、目录和模式源进行预测。."""
    file = tmp_path / "sources_multi_row.txt"
    with open(file, "w") as f:
        f.writelines(f"{src}\n" for src in SOURCES_LIST)
    results = YOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7, f"Expected 7 results from source list, got {len(results)}"


@pytest.mark.skipif(True, reason="disabled for testing")
def test_predict_csv_multi_row(tmp_path):
    """测试 YOLO 使用 CSV 文件多行中列出的源进行预测。."""
    file = tmp_path / "sources_multi_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source"])
        writer.writerows([[src] for src in SOURCES_LIST])
    results = YOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7, f"Expected 7 results from multi-row CSV, got {len(results)}"


@pytest.mark.skipif(True, reason="disabled for testing")
def test_predict_csv_single_row(tmp_path):
    """测试 YOLO 使用 CSV 文件单行中列出的源进行预测。."""
    file = tmp_path / "sources_single_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SOURCES_LIST)
    results = YOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7, f"Expected 7 results from single-row CSV, got {len(results)}"


@pytest.mark.parametrize("model_name", MODELS)
def test_predict_img(model_name):
    """测试 YOLO 模型对各种图像输入类型的预测。."""
    if IS_RASPBERRYPI and model_name == "yolo26n-sem.pt":
        skip_rpi_semantic()
    channels = 1 if model_name == "yolo11n-grayscale.pt" else 3
    model = YOLO(WEIGHTS_DIR / model_name)
    im = cv2.imread(str(SOURCE), flags=cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)  # uint8 NumPy array
    assert len(model(source=Image.open(SOURCE), save=True, verbose=True, imgsz=32)) == 1  # PIL
    assert len(model(source=im, save=True, save_txt=True, imgsz=32)) == 1  # ndarray
    assert len(model(torch.rand((2, channels, 32, 32)), imgsz=32)) == 2  # batch-size 2 Tensor, FP32 0.0-1.0 RGB order
    assert len(model(source=[im, im], save=True, save_txt=True, imgsz=32)) == 2  # batch
    assert len(list(model(source=[im, im], save=True, stream=True, imgsz=32))) == 2  # stream
    assert len(model(torch.zeros(320, 640, channels).numpy().astype(np.uint8), imgsz=32)) == 1  # 张量转 NumPy
    batch = [
        str(SOURCE),  # 文件名
        Path(SOURCE),  # Path 对象
        im,  # OpenCV 图像
        Image.open(SOURCE),  # PIL 图像
        np.zeros((320, 640, channels), dtype=np.uint8),  # NumPy 数组
    ]
    assert len(model(batch, imgsz=32, classes=0)) == len(batch)  # multiple sources in a batch


@pytest.mark.parametrize("model_name", ["yolo26n.pt", "yolo11n.pt"])  # 端到端模型和基于 NMS 的模型
def test_predict_classes_with_max_det(model_name):
    """测试 end2end 模型和基于 NMS 的模型中的 classes-before-max_det 及重复调用过滤器重置。."""
    boxes = YOLO(WEIGHTS_DIR / model_name)(SOURCE, classes=[0], max_det=300, verbose=False)[0].boxes
    assert len(boxes) > 1  # bus.jpg 包含多个人
    top1_model = YOLO(WEIGHTS_DIR / model_name)
    top1 = top1_model(SOURCE, classes=[0], max_det=1, verbose=False)[0].boxes
    assert len(top1) == 1 and int(top1.cls) == 0
    assert float(top1.conf) == pytest.approx(float(boxes.conf.max()))  # 保留置信度最高的人，而不是任意一个

    reused = top1_model(SOURCE, verbose=False)[0].boxes  # 同一个模型，这次完全不传 kwargs
    assert len(reused) > 1  # 上一次调用的 classes=[0]/max_det=1 不能泄漏到本次调用


@pytest.mark.parametrize("model", MODELS)
def test_predict_visualize(model):
    """使用 visualize=True 测试模型预测方法生成预测可视化结果。."""
    if IS_RASPBERRYPI and model == "yolo26n-sem.pt":
        skip_rpi_semantic()
    YOLO(WEIGHTS_DIR / model)(SOURCE, imgsz=32, visualize=True)


def test_load_tensor_uint8():
    """测试张量归一化支持 uint8，同时保持浮点 epsilon 容差。."""
    from ultralytics.data.loaders import LoadTensor

    loaded = LoadTensor(torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)).im0
    assert loaded.dtype == torch.float32 and loaded.max() == 1
    normalized = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    normalized[..., 0, 0] += torch.finfo(normalized.dtype).eps
    assert LoadTensor(normalized).im0.max() > 1


def test_predict_gray_and_4ch(tmp_path):
    """测试 YOLO 对转换为灰度图和四通道图像的 SOURCE 进行预测，并覆盖各种文件名。."""
    im = Image.open(SOURCE)

    source_grayscale = tmp_path / "grayscale.jpg"
    source_rgba = tmp_path / "4ch.png"
    source_non_utf = tmp_path / "non_UTF_测试文件_tést_image.jpg"
    source_spaces = tmp_path / "image with spaces.jpg"

    im.convert("L").save(source_grayscale)  # 灰度图像
    im.convert("RGBA").save(source_rgba)  # 带 alpha 通道的 4 通道 PNG
    im.save(source_non_utf)  # 文件名包含非 UTF 字符
    im.save(source_spaces)  # 文件名包含空格

    # 推理
    model = YOLO(MODEL)
    for f in source_rgba, source_grayscale, source_non_utf, source_spaces:
        for source in Image.open(f), cv2.imread(str(f)), f:
            results = model(source, save=True, verbose=True, imgsz=32)
            assert len(results) == 1, f"Expected 1 result for {f.name}, got {len(results)}"
        f.unlink()  # 清理


def test_predict_ndarray_channels():
    """测试灰度模型和彩色模型的 NumPy 通道归一化。."""
    from ultralytics.data.loaders import LoadPilAndNumpy

    model = YOLO(MODEL)  # default 3-channel model
    gray = np.asarray(Image.open(SOURCE).convert("L"))  # genuine 2D (H, W) uint8 array
    assert gray.ndim == 2, "Expected a 2D grayscale array for this test"
    assert len(model(source=gray, imgsz=32, verbose=False)) == 1  # 二维 ndarray 自动扩展为 3 个通道
    assert len(model(source=gray.astype("float64"), imgsz=32, verbose=False)) == 1  # 非 OpenCV 数据类型同样有效
    for source_channels, model_channels in ((1, 3), (2, 1), (2, 3), (3, 1), (4, 1), (4, 3)):
        im = np.zeros((8, 8, source_channels), dtype=np.uint8)
        assert LoadPilAndNumpy(im, channels=model_channels).im0[0].shape == (8, 8, model_channels)


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_predict_all_image_formats():
    """在 COCO12-Formats 的 12 种图像格式扩展名上进行预测（AVIF、BMP、DNG、HEIC、JP2、JPEG、JPG、MPO、PNG、TIF、 TIFF、WebP）。.
    """
    # 如有需要则下载数据集
    data = check_det_dataset("coco12-formats.yaml")
    dataset_path = Path(data["path"])

    # 收集 train 和 val 中的所有图像
    expected = {"avif", "bmp", "dng", "heic", "jp2", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"}
    images = [im for im in (dataset_path / "images" / "train").glob("*.*") if im.suffix.lower().lstrip(".") in expected]
    images += [im for im in (dataset_path / "images" / "val").glob("*.*") if im.suffix.lower().lstrip(".") in expected]
    assert len(images) == 12, f"Expected 12 images, found {len(images)}"

    # 确认所有格式扩展名都已覆盖
    extensions = {img.suffix.lower().lstrip(".") for img in images}
    assert extensions == expected, f"Missing formats: {expected - extensions}"

    # 对所有图像执行推理
    model = YOLO(MODEL)
    results = model(images, imgsz=32)
    assert len(results) == 12, f"Expected 12 results, got {len(results)}"


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(is_github_action_running(), reason="No auth https://github.com/JuanBindez/pytubefix/issues/166")
def test_youtube():
    """在 YouTube 视频流上测试 YOLO 模型，并处理潜在的网络错误。."""
    model = YOLO(MODEL)
    try:
        model.predict("https://youtu.be/G17sBkb38XQ", imgsz=32, save=True)
    # 处理网络连接错误和 'urllib.error.HTTPError: HTTP Error 429: Too Many Requests'
    except (urllib.error.HTTPError, ConnectionError) as e:
        LOGGER.error(f"YouTube Test Error: {e}")


def test_track_second_association_indices():
    """在第二次关联中匹配的低置信度检测仍保留其在完整检测集合中的索引。."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    args = IterableSimpleNamespace(**{**YAML.load(ROOT / "cfg/trackers/bytetrack.yaml"), "fuse_score": False})
    tracker = BYTETracker(args)
    boxes = [[10, 10, 50, 50], [200, 200, 260, 260], [400, 400, 480, 480]]
    for confs in ([0.9, 0.9, 0.9], [0.9, 0.9, 0.2]):  # 第三个检测结果在第 2 帧降为低置信度
        data = torch.tensor([[*b, c, 0] for b, c in zip(boxes, confs)], dtype=torch.float32)
        tracks = tracker.update(Boxes(data, (640, 640)))
    low = tracks[np.isclose(tracks[:, 5], 0.2)]  # 列依次为 [x1, y1, x2, y2, id, score, cls, idx]
    assert len(low) == 1 and int(low[0, -1]) == 2, f"second-association idx not preserved:\n{tracks}"


def test_track_split_detections_degenerate_boxes():
    """`_split_detections` 必须从两个置信度分区中删除宽高为零或负值的框，同时保留每个有效检测在完整 检测集合空间中的索引（稍后赋给 `track.idx`）。.
    """
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    args = IterableSimpleNamespace(**YAML.load(ROOT / "cfg/trackers/bytetrack.yaml"))
    tracker = BYTETracker(args)
    boxes = [
        [10, 10, 50, 50, 0.9, 0],  # idx 0: valid, high-confidence partition
        [300, 480, 350, 480, 0.9, 0],  # idx 1: degenerate, zero height, high-confidence partition
        [100, 100, 150, 150, 0.15, 0],  # idx 2: valid, low-confidence partition
        [300, 490, 350, 480, 0.15, 0],  # idx 3: degenerate, negative height, low-confidence partition
        [150, 100, 100, 150, 0.9, 0],  # idx 4: degenerate, negative width, high-confidence partition
    ]
    results = Boxes(torch.tensor(boxes, dtype=torch.float32), (640, 640))
    high, low, mask_high, mask_low = tracker._split_detections(results)
    assert np.flatnonzero(mask_high).tolist() == [0] and len(high) == 1, f"degenerate box leaked high band: {mask_high}"
    assert np.flatnonzero(mask_low).tolist() == [2] and len(low) == 1, f"degenerate box leaked low band: {mask_low}"


@pytest.mark.parametrize("tracker_type", ["bytetrack", "fasttrack"])
def test_track_second_association_low_conf_keeps_id(tracker_type):
    """在默认 fuse_score=True 下，低置信度检测会通过第二次关联得到恢复。."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    args = IterableSimpleNamespace(**YAML.load(ROOT / f"cfg/trackers/{tracker_type}.yaml"))  # 默认 fuse_score=True
    tracker = TRACKER_MAP[tracker_type](args)
    box = [100, 100, 200, 200]  # 两帧中的边界框相同，因此 IoU 为 1.0
    # 第 1 帧：高分检测启动轨迹；第 2 帧：分数降至低分区间（track_low_thresh < 0.15 < track_high_thresh）
    frame1 = tracker.update(Boxes(torch.tensor([[*box, 0.9, 0]], dtype=torch.float32), (640, 640)))
    frame2 = tracker.update(Boxes(torch.tensor([[*box, 0.15, 0]], dtype=torch.float32), (640, 640)))
    assert len(frame1) == 1, f"expected one track on frame 1:\n{frame1}"
    tid = int(frame1[0, 4])
    # 低分边界框必须保留，并通过第二次关联映射到相同 ID
    assert len(frame2) == 1, f"low-confidence detection lost by second association:\n{frame2}"
    assert int(frame2[0, 4]) == tid, f"id switched on low-confidence frame: {tid} -> {int(frame2[0, 4])}\n{frame2}"


def test_tracktrack_new_lifecycle():
    """TrackTrack 会预测新轨迹，并在轨迹历史达到 min_track_len 后确认轨迹。."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = {**YAML.load(ROOT / "cfg/trackers/tracktrack.yaml"), "gmc_method": "none", "min_track_len": 4}
    tracker = TRACKER_MAP["tracktrack"](IterableSimpleNamespace(**cfg))
    tracker.update(Boxes(torch.empty((0, 6)), (640, 640)))  # avoid first-frame auto-activation
    outputs = []
    for center_x in (100, 135, 170, 205):
        box = torch.tensor([[center_x - 50, 50, center_x + 50, 150, 0.9, 0]], dtype=torch.float32)
        outputs.append(tracker.update(Boxes(box, (640, 640))))
    assert [len(output) for output in outputs] == [0, 0, 0, 1]
    for min_track_len in (0, 1):
        cfg["min_track_len"] = min_track_len
        tracker = TRACKER_MAP["tracktrack"](IterableSimpleNamespace(**cfg))
        tracker.update(Boxes(torch.empty((0, 6)), (640, 640)))
        assert len(tracker.update(Boxes(box, (640, 640)))) == 1

    from ultralytics.trackers.basetrack import TrackState

    cfg["min_track_len"] = 4
    tracker = TRACKER_MAP["tracktrack"](IterableSimpleNamespace(**cfg))
    for center_x in (100, 135, 170, 205):  # frame_id == 1 包含真实检测，而不是空的预热帧
        box = torch.tensor([[center_x - 50, 50, center_x + 50, 150, 0.9, 0]], dtype=torch.float32)
        tracker.update(Boxes(box, (640, 640)))
        if tracker.frame_id == 2:
            assert tracker.tracked_stracks[0].state != TrackState.Tracked, "frame_id==1 track confirmed after 2 hits"
    assert tracker.tracked_stracks[0].state == TrackState.Tracked


@pytest.mark.parametrize("tracker_type", ["botsort", "deepocsort", "tracktrack"])
def test_track_reid_auto_user_detections(tracker_type):
    """原生 ReID（model='auto'）在用户提供检测结果时必须降级为仅运动模式，不能对原始帧编码。."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = {**YAML.load(ROOT / f"cfg/trackers/{tracker_type}.yaml"), "with_reid": True, "model": "auto"}
    tracker = TRACKER_MAP[tracker_type](IterableSimpleNamespace(**cfg))
    img = np.full((640, 640, 3), 128, dtype=np.uint8)  # 使用非零值，避免错误的帧特征被丢弃
    data = torch.tensor([[10, 10, 50, 50, 0.9, 0], [200, 200, 260, 260, 0.9, 0]], dtype=torch.float32)
    for _ in range(3):  # 过去在将图像行保存为轨迹特征后，第 2 帧会在 embedding_distance 中崩溃
        tracks = tracker.update(Boxes(data, (640, 640)), img)
    assert len(tracks) == 2, f"native-ReID tracker must keep tracking without feats:\n{tracks}"


@pytest.mark.parametrize("fuse_score", [True, False])
def test_deepocsort_ocr_proximity_gate(fuse_score):
    """无论 fuse_score 如何设置，DeepOCSORT OCR 都必须拒绝 IoU 为零的匹配对，即使外观特征相同。."""
    from types import SimpleNamespace

    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.trackers.deep_oc_sort import DeepOCSORT

    tracker = object.__new__(DeepOCSORT)
    tracker.args = SimpleNamespace(fuse_score=fuse_score, match_thresh=0.8)
    tracker.encoder, tracker.appearance_thresh, tracker.proximity_thresh, tracker.frame_id = object(), 0.9, 0.5, 2
    track = SimpleNamespace(
        angle=None,
        last_observation=np.array([0, 0, 10, 10]),
        smooth_feat=np.array([1.0, 0.0]),
        state=TrackState.Tracked,
        update=lambda *_: None,
    )
    detection = SimpleNamespace(xyxy=np.array([20, 20, 30, 30]), curr_feat=np.array([1.0, 0.0]), score=1.0)
    # 证明外观特征处于启用状态，并且在不设门控时会覆盖这一对象对，因此下面的 OCR 结果由邻近门控导致，
    # 而不是由外观特征不可用导致。
    assert tracker._fuse_appearance(np.array([[1.0]]), [track], [detection]) == 0.0
    assert tracker._ocr_associate([track], [detection], [], []) == ([0], [0])


def test_reid_invalid_crops():
    """测试 ReID 会跳过越界的检测裁剪，同时保持特征对齐。."""
    from types import SimpleNamespace

    from ultralytics.trackers.utils.reid import ReID

    encoder = ReID.__new__(ReID)
    encoder.is_pt = True
    encoder.model = SimpleNamespace(predictor=lambda crops: [torch.ones(4) for _ in crops])
    img = np.full((640, 640, 3), 128, dtype=np.uint8)
    feats = encoder(img, np.array([[30, 30, 40, 40], [1100, 1100, 200, 200]], dtype=np.float32))
    assert feats[0] is not None and feats[1] is None


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.parametrize("model", MODELS)
def test_track_stream(model, tmp_path, solution_assets):
    """使用所有内置跟踪器和各种 GMC/ReID 配置，在短视频上测试流式跟踪。.

    注意：为获得更高置信度和更好的匹配，跟踪需要 imgsz=160。
    """
    if model in {
        "yolo26n-cls.pt",
        "yolo26n-sem.pt",
        "yolo26n-depth.pt",
    }:  # 不支持分类、语义分割和深度任务
        return
    from ultralytics.trackers.track import TRACKER_MAP

    video_url = solution_assets("track_video")
    model = YOLO(model)

    # 对所有内置跟踪器执行默认端到端运行
    for tracker_type in TRACKER_MAP:
        kwargs = {"save_frames": True} if tracker_type == "botsort" else {}
        model.track(video_url, imgsz=160, tracker=f"{tracker_type}.yaml", **kwargs)

    # 测试 botsort 的全局运动补偿（GMC）方法和 ReID
    for gmc, reidm in zip(["orb", "sift", "ecc"], ["auto", "auto", "yolo26n-cls.pt"]):
        default_args = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
        custom_yaml = tmp_path / f"botsort-{gmc}.yaml"
        YAML.save(custom_yaml, {**default_args, "gmc_method": gmc, "with_reid": True, "model": reidm})
        model.track(video_url, imgsz=160, tracker=custom_yaml)

    # 测试 ONNX ReID 编码器自动下载
    if model == "yolo26n.pt":
        default_args = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
        custom_yaml = tmp_path / "botsort-reid-onnx.yaml"
        YAML.save(custom_yaml, {**default_args, "with_reid": True, "model": "yolo26n-reid.onnx"})
        model.track(video_url, imgsz=160, tracker=custom_yaml)


@pytest.mark.parametrize("task,weight,data", TASK_MODEL_DATA)
def test_val(task: str, weight: str, data: str) -> None:
    """测试 YOLO 模型的验证模式。."""
    if IS_RASPBERRYPI and task == "semantic":
        skip_rpi_semantic()
    model = YOLO(weight)
    for plots in (True, False):  # 测试 plots=True 和 plots=False 两种情况
        metrics = model.val(data=data, imgsz=32, plots=plots)
        metrics.to_df()
        metrics.to_csv()
        metrics.to_json()
        if task != "depth":  # depth 是稠密回归：没有类别，也没有混淆矩阵
            metrics.confusion_matrix.to_df()
            metrics.confusion_matrix.to_csv()
            metrics.confusion_matrix.to_json()
            cm = metrics.confusion_matrix
            expected = cm.nc if task in {"classify", "semantic"} else cm.nc + 1  # 检测任务包含背景类别
            assert cm.matrix.shape == (expected, expected), f"{task} confusion matrix is {cm.matrix.shape}"
            assert len(cm.tp_fp()[0]) == cm.nc  # per-class TP/FP never include background


def test_val_save_txt_pose(tmp_path):
    """测试 val(save_txt=True) 和 val(save_json=True) 保存的姿态关键点位于原始图像空间。."""
    model = YOLO(WEIGHTS_DIR / "yolo26n-pose.pt")
    # imgsz=640（不是其他位置使用的 imgsz=32）：coco8-pose 图像不是正方形，因此 letterbox 偏移只有在完整分辨率下
    # 才足以将缩放错误的关键点推到 [0, 1] 外；较小 imgsz 会使其留在范围内并隐藏回归问题。
    # save_json=True 还会执行 pred_to_json，这是缩放键的另一个使用方。
    metrics = model.val(
        data="coco8-pose.yaml", imgsz=640, conf=0.25, save_txt=True, save_json=True, project=tmp_path, name="val"
    )
    txt_files = list((Path(metrics.save_dir) / "labels").glob("*.txt"))
    assert txt_files, "val(save_txt=True) saved no label files"
    assert (Path(metrics.save_dir) / "predictions.json").exists(), "val(save_json=True) saved no predictions.json"
    for txt_file in txt_files:
        for line in txt_file.read_text().splitlines():
            values = [float(v) for v in line.split()]
            x, y, w, h = values[1:5]  # normalized xywh box
            kpts = torch.tensor(values[5:]).view(-1, 3)  # 归一化的 (x, y, conf) 关键点，形状为 (17, 3)
            assert ((kpts[:, :2] >= 0) & (kpts[:, :2] <= 1)).all(), f"keypoints not in [0, 1] in {txt_file.name}"
            # 缩放到错误 letterbox 空间的关键点也会落到人体外，因此检查可见关键点是否聚集在边界框内；
            # 0.05 的边距允许关节（手腕、脚踝）略微超出紧框。
            visible = kpts[kpts[:, 2] > 0.5, :2]
            if len(visible):
                cx, cy = visible.mean(0)
                assert abs(cx - x) < w / 2 + 0.05 and abs(cy - y) < h / 2 + 0.05, "keypoints misaligned with box"


def test_pose_metrics_curves():
    """测试姿态曲线标签包含四条唯一的框和姿态序列。."""
    from ultralytics.utils.metrics import PoseMetrics

    curves = PoseMetrics().curves
    assert len(curves) == len(set(curves)) == 8


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(IS_JETSON or IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_multi():
    """测试在数据集集合上微调基础模型，此过程会针对 list/tuple 数据触发 MultiTrainer。."""
    model = YOLO(MODEL)
    results = model.train(data=["coco8.yaml", "coco8.yaml"], epochs=1, imgsz=32)
    assert isinstance(results, dict) and len(results) == 2  # 每次运行一个条目（coco8、coco8-2），不合并重复项
    assert all(m and "fitness" in m for m in results.values())  # 每次运行的检查点训练指标
    assert len(model.trainer.trainers) == 2  # both list entries fine-tuned in series
    sweep_dir = model.trainer.save_dir
    assert sweep_dir.name.startswith("multitrain")  # 所有运行都归入同一个 sweep 目录
    assert (sweep_dir / "multitrain_results.json").exists()  # 用于后处理的结果 JSON
    assert (sweep_dir / "multitrain_results.png").exists()  # cross-dataset results plot


def test_normalize_platform_uri():
    """测试 Platform 网页 URL 会重写为 ul:// URI，使数据集和模型可以直接从粘贴的 URL 加载。."""
    from ultralytics.utils.checks import normalize_platform_uri

    base = "https://platform.ultralytics.com/glenn-jocher"
    assert normalize_platform_uri(f"{base}/datasets/coco8") == "ul://glenn-jocher/datasets/coco8"
    assert normalize_platform_uri(f"{base}/project/model/") == "ul://glenn-jocher/project/model"
    assert normalize_platform_uri("coco8.yaml") == "coco8.yaml"  # non-Platform inputs unchanged


def test_convert_signed_ndjson(monkeypatch):
    """测试带签名的 NDJSON URL 会在数据集 YAML 验证前完成转换。."""
    from ultralytics.data import converter, utils

    captured = []

    async def convert(path):
        captured.append(path)
        return "dataset.ndjson.yaml"

    monkeypatch.setattr(converter, "convert_ndjson_to_yolo", convert)
    url = "https://storage.googleapis.com/bucket/dataset-v1.ndjson?X-Goog-Signature=abc"
    assert utils.convert_ndjson_to_yolo_if_needed(url) == "dataset.ndjson.yaml"
    assert captured == [url]


@pytest.mark.parametrize("task", ["detect", "classify"])
def test_ndjson_conversion_concurrency_and_resume(monkeypatch, tmp_path, task):
    """测试并发转换会共享工作内容，且中断的转换会在发布完成状态前恢复。."""
    import asyncio
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import aiohttp

    from ultralytics.data import converter

    counts, failures, conversions, count_lock = {}, set(), 0, threading.Lock()

    class Response:
        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        async def read(self):
            await asyncio.sleep(0.01)
            with count_lock:
                counts[self.url] = counts.get(self.url, 0) + 1
                fail = self.url in failures
                failures.discard(self.url)
            if fail:
                raise OSError("interrupted")
            return image_bytes

    class Session:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def get(self, url, **_):
            return Response(url)

    ok, image = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    image_bytes = image.tobytes()
    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    original_convert = converter._convert_ndjson_to_yolo

    async def track_conversion(*args):
        nonlocal conversions
        with count_lock:
            conversions += 1
        return await original_convert(*args)

    monkeypatch.setattr(converter, "_convert_ndjson_to_yolo", track_conversion)
    annotations = {"classification": [7]} if task == "classify" else {"boxes": [[0, 0.5, 0.5, 1, 1]]}

    def write_ndjson(name):
        path = tmp_path / f"{name}.ndjson"
        records = [
            {"type": "dataset", "task": task, "class_names": {"7": "item", "8": "rare"}},
            {
                "file": "train.jpg",
                "url": f"https://example.com/{name}-train.jpg",
                "split": "train",
                "annotations": annotations,
            },
            {
                "file": "val.jpg",
                "url": f"https://example.com/{name}-val.jpg",
                "split": "val",
                "annotations": {"classification": [8]} if task == "classify" else annotations,
            },
        ]
        path.write_text("\n".join(json.dumps(record) for record in records))
        return path

    concurrent = write_ndjson("concurrent")
    jobs = 2
    barrier = threading.Barrier(jobs)

    def convert(path):
        barrier.wait()
        return asyncio.run(converter.convert_ndjson_to_yolo(path, tmp_path))

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(convert, [concurrent] * jobs))
    assert len(set(results)) == 1
    assert conversions == 1
    assert sum(counts.values()) == 2
    if task == "classify":
        assert check_cls_dataset(results[0])["nc"] == 2
        from ultralytics.data import dataset as dataset_module

        monkeypatch.setattr(dataset_module, "TORCHVISION_0_18", False)
        args = copy(DEFAULT_CFG)
        train = dataset_module.ClassificationDataset(results[0] / "train", args)
        val = dataset_module.ClassificationDataset(results[0] / "val", args)
        assert train.samples[0][1] == 0
        assert val.samples[0][1] == 1
        assert dataset_module.ClassificationDataset(results[0] / "val", args).samples[0][1] == 1

    resume = write_ndjson("resume")
    failed_url = "https://example.com/resume-val.jpg"
    failures.add(failed_url)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(convert, resume) for _ in range(jobs)]
        errors, results = [], []
        for future in futures:
            try:
                results.append(future.result())
            except RuntimeError as error:
                errors.append(str(error))
    assert len(errors) == len(results) == 1 and "Downloaded 1/2 images" in errors[0]
    result = results[0]
    marker = result / ".ndjson.yaml" if task == "classify" else result
    assert YAML.load(marker)["complete"] is True
    assert counts["https://example.com/resume-train.jpg"] == 1
    assert counts[failed_url] == 2
    request_count = sum(counts.values())
    asyncio.run(converter.convert_ndjson_to_yolo(resume, tmp_path))
    assert conversions == 4
    assert sum(counts.values()) == request_count


def test_platform_job_transport(monkeypatch, tmp_path):
    """使用已有本地检查点测试可配置的 Platform 传输。."""
    from types import SimpleNamespace

    from ultralytics import SETTINGS, cfg
    from ultralytics.utils.callbacks import platform

    monkeypatch.setattr(cfg, "TESTS_RUNNING", False)
    monkeypatch.setitem(SETTINGS, "runs_dir", str(tmp_path))
    args = SimpleNamespace(
        save_dir=None, project="user/project", task="detect", name="model", mode="train", exist_ok=True
    )
    assert cfg.get_save_dir(args) == tmp_path / "detect/user/project/model"

    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return SimpleNamespace(status_code=200, json=lambda: {"received": True}, raise_for_status=lambda: None)

    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr(platform, "_api_key", "api-key")
    monkeypatch.setattr(platform, "PLATFORM_API_URL", "https://example.test/api/webhooks")
    assert platform._send("epoch_end", {"epoch": 0}, "user/project", "model") == {"received": True}
    assert captured["url"] == "https://example.test/api/webhooks/training/metrics"
    assert captured["json"]["data"] == {"epoch": 0}
    assert captured["headers"] == {"Authorization": "Bearer api-key"}

    model = tmp_path / "models" / "best.pt"
    model.parent.mkdir()
    model.write_bytes(b"weights")
    monkeypatch.setenv("PLATFORM_API_URL", "http://127.0.0.1:8765")
    assert platform._upload_model(model, "user/project", "model") == {
        "modelPath": str(model),
        "modelSize": 7,
    }


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(IS_JETSON or IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_scratch():
    """测试 YOLO 模型在 COCO12-Formats 数据集的 12 种不同图像类型上从头训练。."""
    model = YOLO(CFG)
    model.train(data="coco12-formats.yaml", epochs=2, imgsz=32, cache="disk", batch=-1, close_mosaic=1, name="model")
    model(SOURCE)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_ndjson():
    """测试使用 NDJSON 格式数据集训练 YOLO 模型。."""
    model = YOLO(WEIGHTS_DIR / "yolo26n.pt")
    model.train(data=f"{ASSETS_URL}/coco8-ndjson.ndjson", epochs=1, imgsz=32)


@pytest.mark.parametrize("scls", [False, True])
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_pretrained(scls):
    """测试从预训练检查点开始训练 YOLO 模型。."""
    model = YOLO(WEIGHTS_DIR / "yolo26n-seg.pt")
    model.train(
        data="coco8-seg.yaml", epochs=1, imgsz=32, cache="ram", copy_paste=0.5, mixup=0.5, name=0, single_cls=scls
    )
    model(SOURCE)


def test_all_model_yamls():
    """测试根据 `cfg/models` 目录中所有可用 YAML 配置创建 YOLO 模型。."""
    for m in (ROOT / "cfg" / "models").rglob("*.yaml"):
        if "rtdetr" in m.name:
            if TORCH_1_11:
                _ = RTDETR(m.name)(SOURCE, imgsz=160)
        else:
            YOLO(m.name)


@pytest.mark.skipif(WINDOWS, reason="Windows slow CI export bug https://github.com/ultralytics/ultralytics/pull/16003")
def test_workflow(isolated_model):
    """测试包含训练、验证、预测和导出的完整工作流。."""
    model = YOLO(isolated_model)
    model.train(data="coco8.yaml", epochs=1, imgsz=32, optimizer="SGD")
    model.val(imgsz=32)
    model.predict(SOURCE, imgsz=32)
    model.export(format="torchscript")  # WARNING: Windows slow CI export bug


def test_predict_callback_and_setup():
    """测试 YOLO 预测设置和执行期间的回调功能。."""

    def on_predict_batch_end(predictor):
        """在预测批次结束时处理相关操作的回调函数。."""
        path, im0s, _ = predictor.batch
        im0s = im0s if isinstance(im0s, list) else [im0s]
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)  # results 是 list[batch_size]

    model = YOLO(MODEL)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

    dataset = load_inference_source(source=SOURCE)
    bs = dataset.bs  # access predictor properties
    results = model.predict(dataset, stream=True, imgsz=32)  # source already setup
    for r, im0, bs in results:
        print("test_callback", im0.shape)
        print("test_callback", bs)
        boxes = r.boxes  # 边界框输出的 Boxes 对象
        print(boxes)


@pytest.mark.parametrize("model", MODELS)
def test_results(model: str, tmp_path, solution_assets):
    """测试 YOLO 模型结果处理以及各种格式的输出。."""
    if IS_RASPBERRYPI and model == "yolo26n-sem.pt":
        skip_rpi_semantic()
    im = solution_assets("boats") if model == "yolo26n-obb.pt" else SOURCE
    is_semantic = "semantic" in model or "-sem" in model
    results = YOLO(WEIGHTS_DIR / model)([im, im], imgsz=32 if is_semantic else 160)
    for r in results:
        if is_semantic:
            assert r.semantic_mask is not None and r.semantic_mask.shape == r.orig_shape, (
                f"'{model}' semantic_mask should match the original image shape!"
            )
            assert r.semantic_mask.data.dtype == torch.uint8, f"'{model}' semantic_mask should use compact class IDs!"
        else:
            assert len(r), f"'{model}' results should not be empty!"
        r = r.cpu().numpy()
        print(r, len(r), r.path)  # print numpy attributes
        r = r.to(device="cpu", dtype=torch.float32)
        r.save_txt(txt_file=tmp_path / "runs/tests/label.txt", save_conf=True)
        r.save_crop(save_dir=tmp_path / "runs/tests/crops/")
        r.to_df(decimals=3)  # 与 to_ 方法保持一致：https://docs.ultralytics.com/modes/predict#working-with-results
        r.to_csv()
        r.to_json(normalize=True)
        r.plot(pil=True, save=True, filename=tmp_path / "results_plot_save.jpg")
        r.plot(conf=True, boxes=True)
        print(r, len(r), r.path)  # 调用方法后打印


def test_results_plot_without_boxes():
    """测试绘制仅包含掩码的 Results（boxes=None）不会抛出 AttributeError。."""
    from ultralytics.engine.results import Results

    orig_img = np.zeros((640, 640, 3), dtype=np.uint8)
    masks = torch.zeros((2, 640, 640), dtype=torch.float32)
    r = Results(orig_img, path="image.jpg", names={0: "a", 1: "b"}, masks=masks)
    assert r.boxes is None
    for color_mode in ("class", "instance"):
        assert r.plot(color_mode=color_mode).shape == orig_img.shape


def test_results_depth_field():
    """深度数组会转换为 DepthMap，并能顺利经过 .cpu().numpy() 调用链。."""
    from ultralytics.engine.results import DepthMap, Results

    img = np.zeros((20, 24, 3), dtype=np.uint8)
    depth = np.random.rand(20, 24).astype(np.float32)
    r = Results(orig_img=img, path="x.jpg", names={0: "depth"}, depth=depth)
    assert isinstance(r.depth, DepthMap)
    assert r.depth.data.shape == (20, 24)
    rc = r.cpu().numpy()  # 测试 BaseTensor 的 _keys 链路（.cpu()/.numpy()）
    assert rc.depth is not None
    assert rc.depth.data.shape == (20, 24)  # 形状在 .cpu().numpy() 链中保持不变


def test_results_depth_none_summary_len_and_update():
    """仅深度 Results：None 可直接传递，摘要为空，__len__ 会计算深度图，update() 会包装数组。."""
    from ultralytics.engine.results import DepthMap, Results

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert Results(orig_img=img, path="x.jpg", names={}, depth=None).depth is None
    r = Results(orig_img=img, path="x.jpg", names={0: "depth"}, depth=np.ones((8, 8), dtype=np.float32))
    assert r.summary() == []  # 仅包含深度的 Results 没有逐实例摘要
    assert len(r) == 1  # __len__ 返回深度图数量
    r = Results(orig_img=img, path="x.jpg", names={0: "depth"})
    r.update(depth=np.ones((8, 8), dtype=np.float32))
    assert isinstance(r.depth, DepthMap)


def test_results_plot_with_depth():
    """带深度图的 Results.plot() 会将着色后的深度热图叠加到图像上。."""
    from ultralytics.engine.results import Results

    img = np.zeros((24, 24, 3), dtype=np.uint8)
    depth = np.random.rand(24, 24).astype(np.float32)
    r = Results(orig_img=img, path="x.jpg", names={0: "depth"}, depth=depth)
    out = r.plot()  # 不应抛出异常；返回标注图像（默认 masks=True）
    assert out.shape[:2] == (24, 24)  # 叠加热力图，尺寸与输入相同


def test_annotator_depth_map():
    """Annotator.depth_map 会为深度数组着色，同时覆盖全零（没有有效像素）的情况。."""
    from ultralytics.utils.plotting import Annotator

    ann = Annotator(np.zeros((32, 32, 3), dtype=np.uint8))
    ann.depth_map(np.random.rand(32, 32).astype(np.float32))
    assert ann.result().shape == (32, 32, 3)
    ann = Annotator(np.zeros((16, 16, 3), dtype=np.uint8))
    ann.depth_map(np.zeros((16, 16), dtype=np.float32))  # 没有有效像素 -> 不得发生除零
    assert ann.result().shape == (16, 16, 3)


def test_dense_result_tensor_indexing():
    """有效索引会在 SemanticMask/DepthMap 上保留完整图；越界索引会抛出异常；空选择的长度为零。."""
    from ultralytics.engine.results import DepthMap, SemanticMask

    data = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    valid = (0, -1, [0], np.array([0]), torch.tensor(0), torch.tensor([0]), [True], torch.tensor([True]), slice(0, 1))
    invalid = (1, -2, [1], torch.tensor(1))
    empty = ([False], torch.tensor([False]), slice(1, None), slice(0, 0))
    for cls in (SemanticMask, DepthMap):
        dense = cls(data, orig_shape=(4, 5))
        for idx in valid:
            sel = dense[idx]
            assert len(sel) == 1 and torch.equal(torch.as_tensor(sel.data), data), f"{cls.__name__}[{idx!r}]"
        for idx in invalid:
            with pytest.raises(IndexError):
                dense[idx]
        for idx in empty:
            assert len(dense[idx]) == 0, f"{cls.__name__}[{idx!r}] should be empty"


def test_results_plot_empty_dense_selection():
    """对单结果密集型（语义/深度）Results 调用 result[1:].plot() 时返回原图，不叠加图层。."""
    from ultralytics.engine.results import Results

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    dense_map = np.ones((16, 16), dtype=np.float32)
    plain = Results(orig_img=img, path="x.jpg", names={}).plot()
    for kwargs in ({"semantic_mask": dense_map.astype(np.uint8)}, {"depth": dense_map}):
        r = Results(orig_img=img, path="x.jpg", names={0: "a"}, **kwargs)
        assert len(r[1:]) == 0
        np.testing.assert_array_equal(r[1:].plot(), plain)
        with pytest.raises(IndexError):
            r[1]


def test_annotator_tensor_image():
    """Annotator 接受张量图像，并与 Results.plot 的合成像素保持一致。."""
    from ultralytics.engine.results import Results
    from ultralytics.utils.plotting import Annotator

    image = torch.zeros((16, 16, 3), dtype=torch.uint8)
    masks = torch.ones((1, 16, 16), dtype=torch.bool)
    ann = Annotator(image)
    ann.masks(masks, [[255, 0, 0]])
    assert ann.result()[0, 0].tolist() == [127, 0, 0]
    result = Results(np.zeros((16, 16, 3), dtype=np.uint8), path="image.jpg", names={}, masks=masks)
    expected = result.plot(img=np.zeros((16, 16, 3), dtype=np.uint8), boxes=False)
    np.testing.assert_array_equal(result.plot(img=torch.zeros_like(image), boxes=False), expected)


def test_results_update_probs():
    """测试 Results.update(probs=...) 会像其他同级属性一样，将张量包装为 Probs。."""
    from ultralytics.engine.results import Probs, Results

    orig_img = np.zeros((32, 32, 3), dtype=np.uint8)
    r = Results(orig_img, path="image.jpg", names={i: f"c{i}" for i in range(5)}, probs=torch.rand(5))
    r.update(probs=torch.rand(5))
    assert isinstance(r.probs, Probs), "update(probs=) should wrap the tensor in Probs, not store a raw Tensor"
    assert r.verbose() and r.summary(), "verbose()/summary() raise AttributeError on a raw Tensor probs"


def test_labels_and_crops(tmp_path):
    """测试预测参数输出，用于保存 YOLO 检测标签和裁剪图。."""
    imgs = [SOURCE, ASSETS / "zidane.jpg"]
    model = YOLO(WEIGHTS_DIR / "yolo26n.pt")
    results = model(imgs, imgsz=160, save_txt=True, save_crop=True)
    save_path = Path(results[0].save_dir)
    for r in results:
        im_name = Path(r.path).stem
        cls_idxs = r.boxes.cls.int().tolist()
        # 检查是否生成检测结果（每张图像至少应有 2 个检测结果）
        assert len(cls_idxs) >= 2, f"Expected at least 2 detections, got {len(cls_idxs)}"
        # 检查标签路径
        labels = save_path / f"labels/{im_name}.txt"
        assert labels.exists(), f"Label file {labels} does not exist"
        # 检查检测数量是否与标签数量一致
        label_count = len([line for line in labels.read_text().splitlines() if line])
        assert len(r.boxes.data) == label_count, f"Box count {len(r.boxes.data)} != label count {label_count}"
        # 检查裁剪目录和文件
        crop_dirs = list((save_path / "crops").iterdir())
        crop_files = [f for p in crop_dirs for f in p.glob("*")]
        # 裁剪目录与检测结果一致
        crop_dir_names = {d.name for d in crop_dirs}
        assert all(r.names.get(c) in crop_dir_names for c in cls_idxs), (
            f"Crop dirs {crop_dir_names} don't match classes {cls_idxs}"
        )
        # 裁剪数量与检测数量一致
        crop_count = len([f for f in crop_files if im_name in f.name])
        assert crop_count == len(r.boxes.data), f"Crop count {crop_count} != detection count {len(r.boxes.data)}"

    model(SOURCE, imgsz=160, save_crop=True, verbose=False, project=tmp_path, name="crop", exist_ok=True)
    assert any((tmp_path / "crop/crops").rglob("*.jpg")), "save_crop=True alone must write crop files"


def test_data_utils(tmp_path):
    """测试数据工具函数，包括自动划分和 ZIP 归档。."""
    from ultralytics.data.split import autosplit
    from ultralytics.utils.downloads import zip_directory

    images_dir = tmp_path / "coco8/images/val"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(images_dir / "test.jpg")
    metadata_dir = images_dir / "__MACOSX"
    metadata_dir.mkdir()
    nested_metadata_dir = metadata_dir / "nested/__MACOSX"
    nested_metadata_dir.mkdir(parents=True)
    metadata_file = images_dir / ".DS_Store"
    metadata_file.write_bytes(b"metadata")
    (metadata_dir / "._test.jpg").write_bytes(b"metadata")
    (nested_metadata_dir / "._nested.jpg").write_bytes(b"metadata")

    autosplit(tmp_path / "coco8/images")
    assert any((tmp_path / "coco8").glob("autosplit_*.txt"))
    assert zip_directory(images_dir).is_file()
    assert not metadata_dir.exists()
    assert not metadata_file.exists()
    with pytest.raises(ValueError, match="split"):
        check_cls_dataset("imagenet10", split="invalid")
    with pytest.raises(FileNotFoundError, match="'test:' images not found"):
        check_det_dataset("coco8.yaml", split="test")
    data_yaml = tmp_path / "coco8.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\ntest: images/test\nnames: [item]\n")
    with pytest.raises(FileNotFoundError, match="images not found"):
        check_det_dataset(data_yaml, split="test")

    # polygons2masks_overlap must not overflow uint8 on the transient `masks + mask` sum (reaches 2 * i + 1):
    # 当重叠实例超过 128 个时，每个实例仍必须在重叠掩码中保留不同索引
    from ultralytics.data.utils import polygons2masks_overlap

    segments = [
        np.array([[150 - s, 150 - s], [150 + s, 150 - s], [150 + s, 150 + s], [150 - s, 150 + s]], dtype=np.float32)
        for s in range(140, 10, -1)  # 130 concentric squares, all overlapping the center
    ]
    overlap, _ = polygons2masks_overlap((300, 300), segments)
    assert len(np.unique(overlap)) == len(segments) + 1  # background + 130 instances, no uint8 wraparound


def test_safe_download_unzips_local_path_archive(tmp_path):
    """测试 safe_download() 会解压本地归档路径，而不会将其当作远程 URL。."""
    dataset_dir = tmp_path / "coco8 local"
    archive = tmp_path / "coco8 local.zip"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: item\n")

    with zipfile.ZipFile(archive, "w") as zf:
        for path in dataset_dir.rglob("*"):
            zf.write(path, arcname=path.relative_to(tmp_path))

    extracted = safe_download(archive, dir=tmp_path / "datasets", unzip=True, progress=False)
    expected_path = tmp_path / "datasets" / dataset_dir.name
    assert extracted == expected_path, f"Extracted path {extracted} != expected {expected_path}"
    assert (extracted / "data.yaml").is_file(), f"data.yaml not found in {extracted}"
    assert (extracted / "images" / "val").is_dir(), f"images/val not found in {extracted}"


def test_safe_download_skips_unsafe_archive_members(tmp_path):
    """测试 safe_download() 会跳过将解压到目标目录之外的归档成员。."""
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../unsafe.txt", "bad")
        zf.writestr("safe/file.txt", "ok")

    extracted = safe_download(archive, dir=tmp_path / "datasets", unzip=True, progress=False)

    assert not (tmp_path / "unsafe.txt").exists()
    assert (extracted / "safe/file.txt").is_file()


def test_safe_download_skips_unsafe_tar_members(tmp_path):
    """测试 safe_download() 会跳过将解压到目标目录之外的 tar 成员。."""
    source = tmp_path / "safe.txt"
    source.write_text("ok")
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(source, arcname="../unsafe.txt")
        tar.add(source, arcname="safe.txt")

    extracted = safe_download(archive, dir=tmp_path / "datasets", unzip=True, progress=False)

    assert not (tmp_path / "unsafe.txt").exists()
    assert (extracted / "safe.txt").is_file()


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_data_converter(tmp_path):
    """测试从 COCO 到 YOLO 格式的数据集转换函数和类别映射。."""
    from ultralytics.data.converter import coco80_to_coco91_class, convert_coco

    cached_file = DATASETS_DIR / "annotations" / "instances_val2017.json"
    if cached_file.exists():
        shutil.copy2(cached_file, tmp_path / cached_file.name)
    else:
        download(f"{ASSETS_URL}/instances_val2017.json", dir=tmp_path)
    convert_coco(
        labels_dir=tmp_path, save_dir=tmp_path / "yolo_labels", use_segments=True, use_keypoints=False, cls91to80=True
    )
    coco80_to_coco91_class()


def test_data_annotator(tmp_path):
    """测试使用检测模型和分割模型自动标注数据。."""
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(
        ASSETS,
        det_model=WEIGHTS_DIR / "yolo26n.pt",
        sam_model=WEIGHTS_DIR / "mobile_sam.pt",
        output_dir=tmp_path / "auto_annotate_labels",
    )


def test_events():
    """测试事件发送功能。."""
    from ultralytics.utils.events import Events

    events = Events()
    events.enabled = True
    cfg = copy(DEFAULT_CFG)  # does not require deepcopy
    cfg.mode = "test"
    events(cfg)


def test_cfg_init():
    """测试 'ultralytics.cfg' 模块中的配置初始化工具。."""
    from ultralytics.cfg import check_dict_alignment, copy_default_cfg, smart_value

    with contextlib.suppress(SyntaxError):
        check_dict_alignment({"a": 1}, {"b": 2})
    copy_default_cfg()
    (Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")).unlink(missing_ok=False)

    # 使用全面的用例测试 smart_value()
    # 测试 None 转换
    assert smart_value("none") is None
    assert smart_value("None") is None
    assert smart_value("NONE") is None

    # 测试布尔值转换
    assert smart_value("true") is True
    assert smart_value("True") is True
    assert smart_value("TRUE") is True
    assert smart_value("false") is False
    assert smart_value("False") is False
    assert smart_value("FALSE") is False

    # 测试数值转换（ast.literal_eval）
    assert smart_value("42") == 42
    assert smart_value("-42") == -42
    assert smart_value("3.14") == 3.14
    assert smart_value("-3.14") == -3.14
    assert smart_value("1e-3") == 0.001

    # 测试列表/元组转换（ast.literal_eval）
    assert smart_value("[1, 2, 3]") == [1, 2, 3]
    assert smart_value("(1, 2, 3)") == (1, 2, 3)
    assert smart_value("[640, 640]") == [640, 640]

    # 测试字典转换（ast.literal_eval）
    assert smart_value("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}

    # 测试字符串回退（ast.literal_eval 失败时）
    assert smart_value("some_string") == "some_string"
    assert smart_value("path/to/file") == "path/to/file"
    assert smart_value("hello world") == "hello world"

    # 测试代码注入会被阻止（ast.literal_eval 安全性）
    # 这些值应返回字符串，而不是执行代码
    assert smart_value("__import__('os').system('ls')") == "__import__('os').system('ls')"
    assert smart_value("eval('1+1')") == "eval('1+1')"
    assert smart_value("exec('x=1')") == "exec('x=1')"

    assert smart_value("zipfile.ZIP_DEFLATED") == zipfile.ZIP_DEFLATED
    assert smart_value("zipfile.Path") == "zipfile.Path"


def test_depth_calibration_checkpoint_provenance(tmp_path):
    """深度校准会将所选变换和样本数量与检查点一起持久化。."""
    from copy import deepcopy

    from ultralytics.models.yolo.depth.calibrate import _depth_head, calibrate_checkpoint
    from ultralytics.nn.tasks import DepthModel
    from ultralytics.utils.patches import torch_load

    torch.manual_seed(0)
    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    batches = [
        {"img": (torch.rand(2, 3, 64, 64) * 255).to(torch.uint8), "depth": torch.rand(2, 64, 64) * 5 + 0.5}
        for _ in range(4)
    ]
    path = tmp_path / "depth.pt"
    torch.save({"model": deepcopy(model).half()}, path)

    provenance = calibrate_checkpoint(
        path, batches, device="cpu", dataset_hash="manifest-sha256", validation_split="images/val"
    )
    checkpoint = torch_load(path)
    head = _depth_head(checkpoint["model"])

    assert provenance == checkpoint["depth_calibration"]
    assert provenance["candidate"] in {"identity", "scale-only"}
    assert provenance["images"] == 8
    assert provenance["status"] == "selected"
    assert provenance["dataset_hash"] == "manifest-sha256"
    assert provenance["validation_split"] == "images/val"
    assert provenance["strategy"] == "two-fold-held-out-delta1"
    assert set(provenance["scores"]) == {"identity", "scale-only"}
    assert float(head.cal_a) == provenance["a"]
    assert float(head.cal_b) == provenance["b"]


@pytest.mark.parametrize("external", [False, True])
def test_depth_trainer_records_portable_calibration_split(tmp_path, monkeypatch, external):
    """校准来源记录本地数据划分，同时不会拒绝外部验证路径。."""
    from types import SimpleNamespace

    from ultralytics.models.yolo import detect
    from ultralytics.models.yolo.depth import calibrate
    from ultralytics.models.yolo.depth.train import DepthTrainer

    dataset_root = tmp_path / "private" / "dataset"
    validation_path = (tmp_path / "shared" if external else dataset_root) / "images" / "val"
    validation_path.mkdir(parents=True)
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    captured = {}
    monkeypatch.setattr(detect.DetectionTrainer, "final_eval", lambda _self: None)
    monkeypatch.setattr(
        calibrate,
        "calibrate_checkpoint",
        lambda *_args, **kwargs: captured.update(kwargs) or {"status": "selected"},
    )
    trainer = DepthTrainer.__new__(DepthTrainer)
    trainer.best = checkpoint
    trainer.last = tmp_path / "last.pt"
    trainer.save_dir = tmp_path
    trainer.args = SimpleNamespace(plots=False)
    trainer.test_loader = []
    trainer.device = "cpu"
    trainer.data = {"path": dataset_root, "val": str(validation_path), "hash": "manifest-sha256"}

    trainer.final_eval()

    assert captured["validation_split"] == (None if external else "images/val")
    if captured["validation_split"] is not None:
        assert str(tmp_path) not in captured["validation_split"]


def test_depth_dataset_ignores_unreadable_targets(tmp_path):
    """丢弃不可读取的深度图，并接受类别标签为空的单类别模式。."""
    from ultralytics.data.dataset import DepthDataset
    from ultralytics.data.utils import save_depth_png

    images, depth = tmp_path / "images" / "train", tmp_path / "depth" / "train"
    images.mkdir(parents=True)
    depth.mkdir(parents=True)
    for name in ("valid", "scaled", "legacy", "aspect", "corrupt", "missing"):
        cv2.imwrite(str(images / f"{name}.jpg"), np.zeros((32, 32, 3), np.uint8))
    save_depth_png(depth / "valid.png", np.ones((32, 32), dtype=np.float32), scale=100)
    with Image.open(depth / "valid.png") as image:
        assert not image.info
        assert np.asarray(image).max() == 100
    cv2.imwrite(str(depth / "scaled.png"), np.full((32, 32), 150, np.uint16))
    legacy = np.full((32, 32), 2.0, np.float32)
    legacy[0, :3] = np.nan, np.inf, -np.inf
    np.save(depth / "legacy.npy", legacy)
    cv2.imwrite(str(depth / "aspect.png"), np.ones((16, 32), np.uint16))
    (depth / "corrupt.png").write_text("not a png file")

    data = {"names": {0: "depth"}, "nc": 1, "channels": 3, "depth_scale": 100}
    ds = DepthDataset(img_path=str(images), imgsz=32, data=data, augment=False, single_cls=True, batch_size=1)
    assert {Path(f).stem for f in ds.im_files} == {"valid", "scaled", "legacy"}
    assert sorted(ds._load_depth(i).max() for i in range(len(ds))) == [1.0, 1.5, 2.0]
    legacy_index = next(i for i, path in enumerate(ds.im_files) if Path(path).stem == "legacy")
    assert not ds._load_depth(legacy_index)[0, :3].any()
    assert (depth.parent / "train.cache").exists()  # 扫描结果缓存在深度图旁边


def test_utils_init():
    """测试 Ultralytics 库中的初始化工具。."""
    from ultralytics.utils import get_ubuntu_version, is_github_action_running

    get_ubuntu_version()
    is_github_action_running()


def test_utils_checks(monkeypatch):
    """测试文件名、依赖、图像尺寸、显示能力和版本等各种工具检查。."""

    def package_version(name):
        if name == "v2":
            return "1.0"
        raise checks.metadata.PackageNotFoundError

    checks.check_yolov5u_filename("yolov5n.pt")
    checks.check_requirements("numpy")  # 检查 requirements.txt
    checks.check_imgsz([600, 600], max_dim=1)
    with pytest.raises(ValueError):
        checks.check_imgsz("640x480")  # 格式错误的 imgsz 字符串应抛出有帮助的 ValueError，而不是原始 SyntaxError
    checks.check_imshow(warn=True)
    checks.check_suffix("https://example.com/model.pt?token=abc", ".pt")
    checks.check_version("ultralytics", "8.0.0")
    # parse_version 必须补齐到至少 3 个部分并保留所有段，以确保任意版本对都能正确比较
    assert checks.parse_version("2") == (2, 0, 0)
    assert checks.parse_version("4.13.0.92") == (4, 13, 0, 92)
    assert checks.parse_version("2.0.1+cu118") == (2, 0, 1)  # 数字本地/构建后缀不是发行版本段
    assert checks.parse_version("1.0.0rc1") == (1, 0, 0)
    assert checks.parse_version("v2.1") == (2, 1, 0)
    assert checks.parse_version("1.0rc1") == (1, 0, 0)  # 文档规定的非 PEP-440 取舍：预发行版等同于最终版
    monkeypatch.setattr(checks.metadata, "version", package_version)
    monkeypatch.setattr(checks, "ARM64", True)
    monkeypatch.setattr(checks, "AUTOINSTALL", True)
    monkeypatch.setattr(checks, "ONLINE", True)
    commands = []
    monkeypatch.setattr(checks.subprocess, "check_output", lambda command, **kwargs: commands.append(command) or "")
    requirements = ["ray[tune]", "nvidia-modelopt[onnx]>=0.44", "$(touch /tmp/pwned)/missing"]
    assert checks.check_requirements(requirements)
    assert commands[0][5:] == requirements  # requirements remain individual argv entries, never shell source
    assert not checks.check_version("v2", ">=2.0")  # installed version-shaped package keeps metadata precedence
    versions = ("v2.1-rc.1", "v2.1-beta1", "v2.1rev1", "v2.1-dev1", "v2.1+cu118")
    assert all(checks.check_version(v, ">=2.0") for v in versions)
    with pytest.raises(ModuleNotFoundError):
        checks.check_version("v2-missing", ">=2.0", hard=True)
    assert checks.check_version("10.3.0.30", ">=10.3.0,<10.4.0")  # Jetson TensorRT family pin
    assert checks.check_version("6.0", ">=6.0.0")  # 当前版本有 2 个部分时必须满足 3 部分的要求
    assert checks.check_version("2.1", "==2.1.0")
    assert checks.check_version("4.13.0.92", "!=4.13.0.90")  # 4 段版本固定值不能被截断
    assert not checks.check_version("4.13.0.90", "!=4.13.0.90")
    assert checks.check_version("2.0.1", "<2.0.1.5")
    checks.print_args()


@pytest.mark.skipif(WINDOWS, reason="Windows profiling is extremely slow (cause unknown)")
def test_utils_benchmarks():
    """使用 'ultralytics.utils.benchmarks' 中的 'ProfileModels' 测试模型性能。."""
    from ultralytics.utils.benchmarks import ProfileModels

    ProfileModels(["yolo26n.yaml"], imgsz=32, min_time=1, num_timed_runs=3, num_warmup_runs=1).run()


def test_utils_torchutils():
    """测试 Torch 工具函数，包括性能分析和 FLOP 计算。."""
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.utils.torch_utils import profile_ops, time_sync

    x = torch.randn(1, 64, 20, 20)
    m = Conv(64, 64, k=1, s=2)

    profile_ops(x, [m], n=3)
    time_sync()


def test_rtdetr_remap_cls_by_names():
    """测试 RT-DETR 解码器 cls-head 重映射（直接名称匹配、未匹配和去噪部分迁移）。."""
    from types import SimpleNamespace

    from ultralytics.nn.tasks import RTDETRDetectionModel

    # 源模型有 2 个类别（person、bird），目标模型有 3 个（person、bird、airplane）。目标类别 'airplane' 没有源类别。
    dst_state = {
        "score_head.weight": torch.full((3, 1), -1.0),
        "score_head.bias": torch.full((3,), -1.0),
        "decoder.denoising_class_embed.weight": torch.full((3, 4), -1.0),
    }
    csd = {
        "score_head.weight": torch.tensor([[1.0], [2.0]]),
        "score_head.bias": torch.tensor([10.0, 20.0]),
        "decoder.denoising_class_embed.weight": torch.full((2, 4), 9.0),
    }
    tgt = SimpleNamespace(names={0: "person", 1: "bird", 2: "airplane"}, state_dict=lambda: dst_state)
    src = SimpleNamespace(names={0: "bird", 1: "person"})  # 反转顺序，以测试行索引映射
    n = RTDETRDetectionModel._remap_cls_by_names(tgt, csd, src, verbose=False)
    assert n == 3  # score_head.weight + score_head.bias + denoising_class_embed remapped
    assert dst_state["score_head.weight"][0, 0].item() == 2.0  # 'person' <- src[1]
    assert dst_state["score_head.weight"][1, 0].item() == 1.0  # 'bird' <- src[0]
    assert dst_state["score_head.weight"][2, 0].item() == -1.0  # 'airplane' unmatched -> dst init kept
    assert dst_state["score_head.bias"].tolist() == [20.0, 10.0, -1.0]
    dn = dst_state["decoder.denoising_class_embed.weight"]  # row-per-class embedding transfers like score_head
    assert dn[0].eq(9.0).all() and dn[1].eq(9.0).all() and dn[2].eq(-1.0).all()
    assert "decoder.denoising_class_embed.weight" not in csd  # popped so intersect_dicts skips it


@pytest.mark.parametrize("nc", [1, 3])
def test_semantic_loss_all_ignore(nc):
    """整个批次均为 ignore（255）时，SemanticSegmentationLoss 必须保持有限值，例如未标注或无效帧。."""
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SemanticSegmentationModel
    from ultralytics.utils.loss import SemanticSegmentationLoss

    model = SemanticSegmentationModel(cfg="yolo26-sem.yaml", nc=nc, verbose=False)
    model.args = get_cfg()
    loss_fn = SemanticSegmentationLoss(model)
    preds = torch.randn(1, nc, 64, 64, requires_grad=True)
    aux = torch.randn(1, nc, 32, 32, requires_grad=True)
    loss, items = loss_fn((preds, aux), {"semantic_mask": torch.full((1, 64, 64), 255, dtype=torch.long)})
    assert torch.isfinite(loss).all() and all(torch.isfinite(x).all() for x in items.values())
    loss.backward()
    assert preds.grad is not None and aux.grad is not None


class _DepthLossModel(torch.nn.Module):
    """模拟 DepthLoss26 所读取模型接口的简易桩：使用 .parameters() 获取设备，使用 .args 获取超参数。."""

    def __init__(self, **over):
        super().__init__()
        from types import SimpleNamespace

        self.p = torch.nn.Parameter(torch.zeros(1))
        hyp = {"dlog": 1.0, "dgrad": 0.5, "dlam": 1.0}
        hyp.update(over)
        self.args = SimpleNamespace(**hyp)


def _depth_loss_for_scaled_pred(lam, scale):
    """返回结构完美但全局尺度错误的预测结果的纯 SILog 深度损失。."""
    from ultralytics.utils.loss import DepthLoss26

    crit = DepthLoss26(_DepthLossModel(dlam=lam, dgrad=0.0))  # 仅使用 SILog
    gt = torch.rand(2, 1, 16, 16) * 5 + 1.0
    pred = (gt * scale).clone().requires_grad_(True)
    total, _ = crit({"depth": pred}, {"depth": gt})
    return float(total.sum().detach())


def test_v26_depth_loss_lower_lambda_penalizes_scale_error_more():
    """在尺度不变的 SILog（dlam=1）下，全局缩放平移后的预测结果损失近似为零；但随着 dlam 降低， 必须对其进行强烈惩罚（损失变为依赖尺度）。.
    """
    loss_invariant = _depth_loss_for_scaled_pred(lam=1.0, scale=2.0)
    loss_anchored = _depth_loss_for_scaled_pred(lam=0.15, scale=2.0)
    assert loss_invariant < 0.05
    assert loss_anchored > 5 * max(loss_invariant, 1e-6)


def test_utils_ops():
    """测试坐标变换和归一化相关的工具操作。."""
    from ultralytics.utils.ops import (
        ltwh2xywh,
        ltwh2xyxy,
        make_divisible,
        segment2box,
        xywh2ltwh,
        xywh2xyxy,
        xywhn2xyxy,
        xywhr2xyxyxyxy,
        xyxy2ltwh,
        xyxy2xywh,
        xyxy2xywhn,
        xyxyxyxy2xywhr,
    )

    make_divisible(17, torch.tensor([8]))

    boxes = torch.rand(10, 4)  # xywh
    torch.allclose(boxes, xyxy2xywh(xywh2xyxy(boxes)))
    torch.allclose(boxes, xyxy2xywhn(xywhn2xyxy(boxes)))
    torch.allclose(boxes, ltwh2xywh(xywh2ltwh(boxes)))
    torch.allclose(boxes, xyxy2ltwh(ltwh2xyxy(boxes)))

    boxes = torch.rand(10, 5)  # OBB 使用 xywhr
    boxes[:, 4] = torch.randn(10) * 30
    torch.allclose(boxes, xyxyxyxy2xywhr(xywhr2xyxyxyxy(boxes)), rtol=1e-3)

    # segment2box 不能将位于图像左边缘（所有 x == 0）的多边形错误丢弃为零边界框
    assert segment2box(np.array([[0, 100], [0, 150], [0, 200]]), 640, 640).tolist() == [0, 100, 0, 200]

    # 增强后边缘点移出画面时，segment2box 仍必须保留可见范围（issue #24935）
    seg = np.array([[550.0, 100.0], [690.0, 100.0], [690.0, 200.0], [550.0, 200.0]])
    assert segment2box(seg, 640, 640).tolist() == [550, 100, 640, 200]
    seg = np.array([[-10.0, 100.0], [650.0, 100.0], [650.0, 200.0], [-10.0, 200.0]])
    assert segment2box(seg, 640, 640).tolist() == [0, 100, 640, 200]
    assert segment2box(np.array([[100.0, 100.0], [200.0, 100.0], [700.0, -100.0]]), 640, 640).tolist() == [
        100,
        0,
        450,
        100,
    ]
    assert segment2box(np.array([[700.0, 100.0], [750.0, 150.0]]), 640, 640).tolist() == [0, 0, 0, 0]
    assert segment2box(np.empty((0, 2)), 640, 640).tolist() == [0, 0, 0, 0]
    seg = np.array([[-100.0, -100.0], [740.0, -100.0], [740.0, 740.0], [-100.0, 740.0]])  # 包围整张图像
    assert segment2box(seg, 640, 640).tolist() == [0, 0, 640, 640]


def test_scale_coords_nonuniform_letterbox():
    """坐标缩放必须抵消拉伸预处理带来的独立高度和宽度增益。."""
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops

    labels = {"img": np.zeros((320, 640, 3), dtype=np.uint8), "ratio_pad": (3.2, 3.2)}
    ratio_pad = LetterBox((640, 640), scale_fill=True)(labels)["ratio_pad"]
    boxes = np.array([[32.0, 64.0, 320.0, 384.0]])
    coords = torch.tensor([[160.0, 128.0]])
    assert ratio_pad == ((6.4, 3.2), (0, 0))
    assert np.allclose(ops.scale_boxes((640, 640), boxes, (100, 200), ratio_pad), [[10, 10, 100, 60]])
    assert torch.allclose(ops.scale_coords((640, 640), coords, (100, 200), ratio_pad), coords.new_tensor([[50, 20]]))

    boxes = np.array([[32.0, 192.0, 320.0, 352.0]])
    coords = torch.tensor([[160.0, 224.0]])
    assert np.allclose(ops.scale_boxes((640, 640), boxes, (100, 200)), [[10, 10, 100, 60]])
    assert torch.allclose(ops.scale_coords((640, 640), coords, (100, 200)), coords.new_tensor([[50, 20]]))


def test_nms_end2end_classes_before_max_det():
    """端到端 NMS 分支必须像基于 NMS 的分支一样，在截断到 max_det 前先过滤类别。."""
    from ultralytics.utils.nms import non_max_suppression

    # （2、4、6）个端到端预测结果按置信度降序排列：[x1, y1, x2, y2, conf, cls]
    pred = torch.tensor(
        [
            [[0, 0, 9, 9, 0.9, 5], [1, 1, 9, 9, 0.8, 0], [2, 2, 9, 9, 0.7, 0], [3, 3, 9, 9, 0.6, 0]],
            [[0, 0, 9, 9, 0.9, 0], [1, 1, 9, 9, 0.8, 5], [2, 2, 9, 9, 0.7, 5], [3, 3, 9, 9, 0.6, 0]],
        ],
        dtype=torch.float32,
    )
    outputs, indices = non_max_suppression(pred, conf_thres=0.25, classes=[0], max_det=2, return_idxs=True)
    for out, idx, confs, expected in zip(outputs, indices, ([0.8, 0.7], [0.9, 0.6]), ([1, 2], [0, 3])):
        assert out.shape[0] == 2 and (out[:, 5] == 0).all()  # 保留置信度最高的 2 个类别 0 边界框，不被截断
        assert torch.allclose(out[:, 4], torch.tensor(confs))
        assert idx.tolist() == expected
    out = non_max_suppression(pred, conf_thres=0.25, max_det=2)[0]  # 不指定 classes 时，整体前 2 个结果保持不变
    assert torch.allclose(out[:, 4], torch.tensor([0.9, 0.8]))


def test_process_mask_empty():
    """Process_mask/process_mask_native/scale_masks 必须在检测数量为 0 时正常处理而不崩溃。."""
    from ultralytics.utils import ops

    protos, coeffs, bboxes = torch.rand(32, 160, 160), torch.zeros(0, 32), torch.zeros(0, 4)
    assert ops.process_mask(protos, coeffs, bboxes, (640, 640), upsample=True).shape == (0, 640, 640)
    assert ops.process_mask(protos, coeffs, bboxes, (640, 640)).shape == (0, 160, 160)  # 不上采样时的原型分辨率
    assert ops.process_mask_native(protos, coeffs, bboxes, (640, 640)).shape == (0, 640, 640)
    assert ops.scale_masks(torch.zeros(1, 0, 160, 160), (640, 640)).shape == (1, 0, 640, 640)


def test_utils_files(tmp_path):
    """测试文件处理工具，包括文件年龄、日期以及包含空格的路径。."""
    from ultralytics.utils.files import file_age, file_date, get_latest_run, increment_path, spaces_in_path

    file_age(SOURCE)
    file_date(SOURCE)
    get_latest_run(ROOT / "runs")

    path = tmp_path / "path/with spaces"
    path.mkdir(parents=True, exist_ok=True)
    with spaces_in_path(path) as new_path:
        print(new_path)

    exp_dir = tmp_path / "runs" / "exp"
    exp_dir.mkdir(parents=True)
    assert increment_path(exp_dir) == tmp_path / "runs" / "exp-2"

    results_file = exp_dir / "results.txt"
    results_file.touch()
    assert increment_path(results_file) == exp_dir / "results-2.txt"


@pytest.mark.slow
def test_utils_patches_torch_save(tmp_path):
    """测试 _torch_save 抛出 RuntimeError 时 torch_save 的退避处理。."""
    from unittest.mock import MagicMock, patch

    from ultralytics.utils.patches import torch_save

    mock = MagicMock(side_effect=RuntimeError)

    with patch("ultralytics.utils.patches._torch_save", new=mock), pytest.raises(RuntimeError):
        torch_save(torch.zeros(1), tmp_path / "test.pt")

    assert mock.call_count == 4, "torch_save was not attempted the expected number of times"


def test_nn_modules_conv():
    """测试卷积神经网络模块，包括 CBAM、Conv2 和 ConvTranspose。."""
    from ultralytics.nn.modules.conv import CBAM, Conv2, ConvTranspose, DWConvTranspose2d, Focus

    c1, c2 = 8, 16  # input and output channels
    x = torch.zeros(4, c1, 10, 10)  # BCHW

    # 运行测试中尚未覆盖的所有模块
    DWConvTranspose2d(c1, c2)(x)
    ConvTranspose(c1, c2)(x)
    Focus(c1, c2)(x)
    CBAM(c1)(x)

    # 融合算子
    m = Conv2(c1, c2)
    m.fuse_convs()
    m(x)


def test_nn_modules_block():
    """测试各种神经网络块模块。."""
    from ultralytics.nn.modules.block import C1, C3TR, BottleneckCSP, C3Ghost, C3x

    c1, c2 = 8, 16  # input and output channels
    x = torch.zeros(4, c1, 10, 10)  # BCHW

    # 运行测试中尚未覆盖的所有模块
    C1(c1, c2)(x)
    C3x(c1, c2)(x)
    C3TR(c1, c2)(x)
    C3Ghost(c1, c2)(x)
    BottleneckCSP(c1, c2)(x)


def test_nn_detect_head_export_clamps_max_det():
    """检测导出后处理请求的候选数量不应超过可用锚点数量。."""
    from ultralytics.nn.modules.head import Detect

    head = Detect(nc=2, ch=(16,))
    head.export = True
    head.format = "onnx"
    anchors = 21
    assert head.postprocess(torch.rand(1, anchors, 4 + head.nc)).shape == (1, anchors, 6)


def _depth_head_feats():
    """返回一个与 Depth 检测头构造参数匹配的小型 P3/P4/P5 特征金字塔。."""
    return [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]


def test_nn_depth_head_export_upsamples_to_input():
    """深度导出会将结果上采样 4 倍至输入分辨率；推理返回检测头原生分辨率。."""
    from ultralytics.nn.modules.head import Depth

    head = Depth(c_mid=32, ch=(32, 64, 128)).eval()
    for fmt in ("onnx", "coreml"):
        head.export, head.format = True, fmt
        assert head(_depth_head_feats()).shape[-2:] == (256, 256)
    head.export = False
    assert head(_depth_head_feats()).shape[-2:] != (256, 256)  # inference returns native head resolution


def test_nn_depth_head_no_dead_parameters():
    """每个检测头参数都能获得梯度，因此 DDP 不需要 find_unused_parameters。."""
    from ultralytics.nn.modules.head import Depth

    head = Depth(c_mid=32, ch=(32, 64, 128)).train()
    head(_depth_head_feats())["depth"].sum().backward()
    unused = [n for n, p in head.named_parameters() if p.grad is None]
    assert not unused, f"parameters with no gradient: {unused}"


@pytest.fixture
def image():
    """从预定义源加载并返回图像（OpenCV BGR 格式）。."""
    return cv2.imread(str(SOURCE))


@pytest.mark.parametrize(
    "auto_augment, erasing, force_color_jitter",
    [
        (None, 0.0, False),
        ("randaugment", 0.5, True),
        ("augmix", 0.2, False),
        ("autoaugment", 0.0, True),
    ],
)
def test_classify_transforms_train(image, auto_augment, erasing, force_color_jitter):
    """在训练期间使用各种增强测试分类变换。."""
    from ultralytics.data.augment import classify_augmentations

    transform = classify_augmentations(
        size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip=0.5,
        vflip=0.5,
        auto_augment=auto_augment,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        force_color_jitter=force_color_jitter,
        erasing=erasing,
    )

    transformed_image = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    assert transformed_image.shape == (3, 224, 224)
    assert torch.is_tensor(transformed_image)
    assert transformed_image.dtype == torch.float32


@pytest.mark.slow
@pytest.mark.skipif(IS_RASPBERRYPI or IS_JETSON, reason="Edge devices not intended for tuning")
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_model_tune():
    """调整 YOLO 模型以提升性能。."""
    YOLO("yolo26n.pt").tune(
        data=["coco8.yaml", "coco8-grayscale.yaml"], plots=False, imgsz=32, epochs=1, iterations=2, device="cpu"
    )
    YOLO("yolo26n-pose.pt").tune(data="coco8-pose.yaml", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
    YOLO("yolo26n-cls.pt").tune(data="imagenet10", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")


@pytest.mark.slow
@pytest.mark.skipif(IS_RASPBERRYPI or IS_JETSON, reason="Edge devices not intended for tuning")
@pytest.mark.skipif(not ONLINE or not checks.IS_PYTHON_MINIMUM_3_10, reason="environment is offline")
@pytest.mark.skipif(not checks.check_requirements("ray", install=False), reason="ray[tune] not installed")
def test_model_tune_ray():
    """调整 YOLO 模型以提升性能。."""
    YOLO("yolo26n-cls.pt").tune(
        data="imagenet10",
        use_ray=True,
        plots=False,
        imgsz=32,
        epochs=1,
        iterations=2,
        search_alg="random",
        device="cpu",
    )


def test_model_embeddings():
    """测试 YOLO 模型提取嵌入向量的功能。."""
    model_detect = YOLO(MODEL)
    model_segment = YOLO(WEIGHTS_DIR / "yolo26n-seg.pt")

    for batch in [SOURCE], [SOURCE, SOURCE]:  # test batch size 1 and 2
        assert len(model_detect.embed(source=batch, imgsz=32)) == len(batch)
        assert len(model_segment.embed(source=batch, imgsz=32)) == len(batch)

    model_classify = YOLO(WEIGHTS_DIR / "yolo26n-cls.pt")
    assert model_classify.predict(SOURCE, imgsz=32)[0].probs is not None
    assert isinstance(model_classify.embed(SOURCE, imgsz=32)[0], torch.Tensor)
    assert model_classify.predict(SOURCE, imgsz=32)[0].probs is not None
    assert isinstance(model_classify.predict(SOURCE, imgsz=32, embed=[-2])[0], torch.Tensor)
    assert model_classify.predict(SOURCE, imgsz=32)[0].probs is not None


def test_process_mask_native_chunked():
    """分块原生上采样与一次性上采样所有掩码的结果相同。."""
    from ultralytics.utils import ops

    torch.manual_seed(0)
    protos, masks_in = torch.randn(32, 160, 160), torch.randn(70, 32)
    bboxes = torch.rand(70, 4) * 900 + 5  # fractional boxes exercise the crop edge handling
    bboxes[:, 2:] += bboxes[:, :2]
    out = ops.process_mask_native(protos, masks_in, bboxes, (1000, 1000))  # large shape forces multiple chunks
    ref = ops.scale_masks((masks_in @ protos.float().view(32, -1)).view(-1, 160, 160)[None], (1000, 1000))[0]
    ref = ops.crop_mask(ref, bboxes).gt_(0.0).byte()  # single-shot upsample-crop-threshold
    assert torch.equal(out, ref)


@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for CLIP-based models")
@pytest.mark.skipif(
    checks.IS_PYTHON_3_8 and LINUX and ARM64,
    reason="YOLOWorld with CLIP is not supported in Python 3.8 and aarch64 Linux",
)
def test_yolo_world():
    """测试支持 CLIP 的 YOLO World 模型。."""
    model = YOLO(WEIGHTS_DIR / "yolov8s-world.pt")  # no YOLO11n-world model yet
    model.set_classes(["tree", "window"])
    model(SOURCE, conf=0.01)

    model = YOLO(WEIGHTS_DIR / "yolov8s-worldv2.pt")  # no YOLO11n-world model yet
    # 从预训练模型开始训练。训练最后阶段包含评估。
    # 使用类别更少的 dota8.yaml，以减少 CLIP 模型的推理时间
    model.train(
        data="dota8.yaml",
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
    )

    # 测试 WorWorldTrainerFromScratch
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

    model = YOLO("yolov8s-worldv2.yaml")  # no YOLO11n-world model yet
    model.train(
        data={"train": {"yolo_data": ["dota8.yaml"]}, "val": {"yolo_data": ["dota8.yaml"]}},
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
        trainer=WorldTrainerFromScratch,
    )


@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for heavy CLIP-based models")
@pytest.mark.skipif(not TORCH_1_13, reason="YOLOE with CLIP requires torch>=1.13")
@pytest.mark.skipif(
    checks.IS_PYTHON_3_8 and LINUX and ARM64,
    reason="YOLOE with CLIP is not supported in Python 3.8 and aarch64 Linux",
)
def test_yoloe(tmp_path):
    """测试支持 MobileCLIP 的 YOLOE 模型。."""
    # 预测
    # 文本提示
    model = YOLO(WEIGHTS_DIR / "yoloe-11s-seg.pt")
    model.set_classes(["person", "bus"])
    model(SOURCE, conf=0.01)

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    # 视觉提示
    visuals = {
        "bboxes": np.array([[221.52, 405.8, 344.98, 857.54], [120, 425, 160, 445]]),
        "cls": np.array([0, 1]),
    }
    model.predict(
        SOURCE,
        visual_prompts=visuals,
        predictor=YOLOEVPSegPredictor,
    )

    # 验证
    model = YOLOE(WEIGHTS_DIR / "yoloe-11s-seg.pt")
    # 文本提示
    model.val(data="coco128-seg.yaml", imgsz=32)
    # 视觉提示
    model.val(data="coco128-seg.yaml", load_vp=True, imgsz=32)

    # 训练、微调
    from ultralytics.models.yolo.yoloe import YOLOEPEFreeTrainer, YOLOEPESegTrainer, YOLOESegTrainerFromScratch

    model = YOLOE("yoloe-11s-seg.pt")
    model.train(
        data="coco128-seg.yaml",
        epochs=1,
        close_mosaic=1,
        trainer=YOLOEPESegTrainer,
        imgsz=32,
    )
    # 从头训练
    data_dict = {"train": {"yolo_data": ["coco128-seg.yaml"]}, "val": {"yolo_data": ["coco128-seg.yaml"]}}
    data_yaml = tmp_path / "yoloe-data.yaml"
    YAML.save(data=data_dict, file=data_yaml)
    for data in [data_dict, data_yaml]:
        model = YOLOE("yoloe-11s-seg.yaml")
        model.train(
            data=data,
            epochs=1,
            close_mosaic=1,
            trainer=YOLOESegTrainerFromScratch,
            imgsz=32,
        )

    # 无提示
    # 预测
    model = YOLOE(WEIGHTS_DIR / "yoloe-11s-seg-pf.pt")
    model.predict(SOURCE)
    # 验证
    model = YOLOE("yoloe-11s-seg.pt")  # 也可以选择不同尺寸的 yoloe-m/l-seg.pt
    model.val(data="coco128-seg.yaml", imgsz=32)
    # 训练，冻结除分类分支之外的所有部分
    model = YOLOE("yoloe-11s-seg.pt")
    head = len(model.model.model) - 1
    freeze = [str(i) for i in range(head)]
    freeze += [f"{head}.{name}" for name, _ in model.model.model[-1].named_children() if "cv3" not in name]
    freeze += [f"{head}.cv3.{i}.{j}" for i in range(3) for j in (0, 1)]
    model.train(
        data={"train": {"yolo_data": ["coco128-seg.yaml"]}, "val": {"yolo_data": ["coco128-seg.yaml"]}},
        epochs=1,
        close_mosaic=1,
        trainer=YOLOEPEFreeTrainer,
        imgsz=32,
        freeze=freeze,
        single_cls=True,
    )
    assert "seg_loss" in model.trainer.loss_names  # 分割损失，而不是检测损失
    assert Path(model.trainer.best).exists()  # 训练结束时已执行验证并保存权重


def test_yoloe_visual_prompt_verbose_false(capfd):
    """验证 YOLOE 视觉提示遵守 verbose=False 设置。."""
    model = YOLO(WEIGHTS_DIR / "yoloe-11s-seg.pt")

    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    visuals = {
        "bboxes": np.array([[221.52, 405.8, 344.98, 857.54]]),
        "cls": np.array([0]),
    }

    # 忽略加载模型时产生的任何输出
    capfd.readouterr()

    model.predict(
        SOURCE,
        refer_image=SOURCE,
        visual_prompts=visuals,
        predictor=YOLOEVPSegPredictor,
        verbose=False,
    )

    captured = capfd.readouterr()
    output = captured.out + captured.err

    assert "Ultralytics" not in output


def test_yolov10():
    """测试 YOLOv10 模型的训练、验证和预测功能。."""
    model = YOLO("yolov10n.yaml")
    # train/val/predict
    model.train(data="coco8.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8.yaml", imgsz=32)
    model.predict(imgsz=32, save_txt=True, save_crop=True, augment=True)
    model(SOURCE)


def test_multichannel():
    """测试 YOLO 模型的多通道训练、验证和预测功能。."""
    model = YOLO("yolo26n.pt")
    model.train(data="coco8-multispectral.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8-multispectral.yaml")
    im = np.zeros((32, 32, 10), dtype=np.uint8)
    model.predict(source=im, imgsz=32, save_txt=True, save_crop=True, augment=True)
    model.export(format="onnx")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_grayscale(task: str, model: str, data: str, tmp_path) -> None:
    """测试 YOLO 模型的灰度训练、验证和预测功能。."""
    if IS_RASPBERRYPI and task == "semantic":
        skip_rpi_semantic()
    if task in {"classify", "depth"}:  # 分类或深度任务不支持灰度图
        return
    grayscale_data = tmp_path / f"{Path(data).stem}-grayscale.yaml"
    data = check_det_dataset(data)
    data["channels"] = 1  # 为灰度图添加 channels 键
    YAML.save(data=data, file=grayscale_data)
    # 如果 train/val 划分中存在 npy 文件则删除，它们可能由之前的测试创建
    for split in ("train", "val"):
        for npy_file in (Path(data["path"]) / data[split]).glob("*.npy"):
            npy_file.unlink()

    model = YOLO(model)
    model.train(data=grayscale_data, epochs=1, imgsz=32, close_mosaic=1, cache="ram")
    model.val(data=grayscale_data)
    im = np.zeros((32, 32, 1), dtype=np.uint8)
    model.predict(source=im, imgsz=32, save_txt=True, save_crop=True, augment=True)
    export_model = model.export(format="onnx")

    model = YOLO(export_model, task=task)
    model.predict(source=im, imgsz=32)


def test_semantic_polygon_data():
    """使用多边形数据测试 YOLO 语义分割模型。."""
    skip_rpi_semantic()
    model = YOLO("yolo26n-sem.pt")
    model.train(data="coco8-seg.yaml", epochs=1, imgsz=32, close_mosaic=1)
    model.val(data="coco8-seg.yaml")
