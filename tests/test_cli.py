# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODELS, TASK_MODEL_DATA
from ultralytics.utils import ARM64, ASSETS, DATASETS_DIR, IS_RASPBERRYPI, LINUX, WEIGHTS_DIR, checks
from ultralytics.utils.torch_utils import TORCH_1_11, TORCH_VERSION


def run(cmd: str) -> None:
    """使用 subprocess 执行 shell 命令。"""
    subprocess.run(cmd.split(), check=True)


def test_special_modes() -> None:
    """测试 YOLO 的各种特殊命令行模式。"""
    run("yolo help")
    run("yolo checks")
    run("yolo version")
    run("yolo settings reset")
    run(f"yolo settings weights_dir={WEIGHTS_DIR} datasets_dir={DATASETS_DIR}")
    run("yolo cfg")


@pytest.mark.parametrize("api_key", ["legacy_api_key", "ul_" + "a" * 40])
def test_settings_migration(tmp_path: Path, api_key: str) -> None:
    """验证架构迁移会保留用户设置，并且只保留 Platform API 密钥。"""
    from ultralytics.utils import SettingsManager

    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "settings_version": "0.0.6",
                "runs_dir": "/custom/runs",
                "api_key": api_key,
                "hub": True,
            }
        )
    )
    settings = SettingsManager(settings_file, version="0.0.7")

    assert settings["runs_dir"] == "/custom/runs"
    assert settings["api_key"] == (api_key if api_key.startswith("ul_") else "")
    assert settings["settings_version"] == "0.0.7"
    assert "hub" not in settings


def test_platform_login(monkeypatch) -> None:
    """验证 Platform 登录会保存有效密钥，退出登录会移除这些密钥。"""
    import requests

    from ultralytics import cfg

    class Response:
        status_code = 200

    settings = {"api_key": ""}
    monkeypatch.setattr(cfg, "SETTINGS", settings)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())

    cfg.handle_yolo_login(["login", "ul_valid"])
    assert settings["api_key"] == "ul_valid"
    cfg.handle_yolo_login(["logout"])
    assert settings["api_key"] == ""


def test_cli_imports_defer_torchvision() -> None:
    """验证启动导入不会加载 torchvision 或 SAM3 几何模块。"""
    code = (
        "import sys; "
        "from ultralytics import YOLO; "
        "from ultralytics.models.sam import Predictor; "
        "assert 'torchvision' not in sys.modules; "
        "assert 'ultralytics.models.sam.sam3.geometry_encoders' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train(task: str, model: str, data: str) -> None:
    """测试 YOLO 在不同任务、模型和数据集上的训练。"""
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 cache=disk")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_val(task: str, model: str, data: str) -> None:
    """使用 shell 命令测试指定任务、模型和数据的 YOLO 验证流程。"""
    for end2end in (False, True):
        run(f"yolo val {task} model={model} data={data} imgsz=32 end2end={end2end} max_det=100 agnostic_nms")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_predict(task: str, model: str, data: str) -> None:
    """使用给定的示例资源测试指定任务和模型的 YOLO 预测。"""
    for end2end in (False, True):
        run(f"yolo {task} predict model={model} source={ASSETS} imgsz=32 save end2end={end2end} max_det=100")


@pytest.mark.parametrize("model", MODELS)
def test_export(model: str, tmp_path: Path) -> None:
    """测试将 YOLO 模型导出为 TorchScript 格式。"""
    from ultralytics.utils.downloads import attempt_download_asset

    isolated = tmp_path / model
    shutil.copy(Path(attempt_download_asset(model)), isolated)
    for end2end in (False, True):
        run(f"yolo export model={isolated} format=torchscript imgsz=32 end2end={end2end} max_det=100")


@pytest.mark.parametrize(
    "task,data,student,teacher",
    [
        ("detect", "coco8.yaml", "yolo26n.yaml", WEIGHTS_DIR / "yolo26s.pt"),
        ("segment", "coco8-seg.yaml", "yolo26n-seg.yaml", WEIGHTS_DIR / "yolo26s-seg.pt"),
        ("pose", "coco8-pose.yaml", "yolo26n-pose.yaml", WEIGHTS_DIR / "yolo26s-pose.pt"),
        ("obb", "dota8.yaml", "yolo26n-obb.yaml", WEIGHTS_DIR / "yolo26s-obb.pt"),
    ],
)
def test_distill(task: str, data: str, student: str, teacher: Path) -> None:
    """通过 CLI 测试支持任务的 YOLO 知识蒸馏训练。"""
    run(f"yolo train {task} model={student} distill_model={teacher} data={data} imgsz=32 epochs=1")


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
@pytest.mark.skipif(
    LINUX and ARM64 and checks.IS_PYTHON_3_8 and "2.1.0a0" in TORCH_VERSION,
    reason="RTDETR CPU training produces NaN losses with JetPack 5 torch 2.1.0a0",
)
def test_rtdetr(task: str = "detect", model: Path = WEIGHTS_DIR / "rtdetr-l.pt", data: str = "coco8.yaml") -> None:
    """使用指定模型和数据测试 Ultralytics 中检测任务的 RTDETR 功能。"""
    # 添加逗号和空格，以测试 CLI 参数清理。
    run(f"yolo predict {task} model={model} source={ASSETS / 'bus.jpg'} imgsz=160 save")
    run(f"yolo train {task} model={model} data={data} --imgsz= 160 epochs =1, cache = disk")


@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for heavy FastSAM tests")
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="MobileSAM with CLIP is not supported in Python 3.12")
@pytest.mark.skipif(
    checks.IS_PYTHON_3_8 and LINUX and ARM64,
    reason="MobileSAM with CLIP is not supported in Python 3.8 and aarch64 Linux",
)
def test_fastsam(
    task: str = "segment", model: str = WEIGHTS_DIR / "FastSAM-s.pt", data: str = "coco8-seg.yaml"
) -> None:
    """在 Ultralytics 中使用各种提示测试 FastSAM 模型的图像目标分割。"""
    source = ASSETS / "bus.jpg"

    run(f"yolo segment val {task} model={model} data={data} imgsz=32")
    run(f"yolo segment predict model={model} source={source} imgsz=32 save")

    from ultralytics import FastSAM
    from ultralytics.models.sam import Predictor

    # 创建 FastSAM 模型
    sam_model = FastSAM(model)  # 也可以使用 FastSAM-x.pt

    # 对图像执行推理
    for s in (source, Image.open(source)):
        everything_results = sam_model(s, device="cpu", retina_masks=True, imgsz=160, conf=0.4, iou=0.9)

        # 移除较小区域
        _new_masks, _ = Predictor.remove_small_regions(everything_results[0].masks.data, min_area=20)

        # 同时使用边界框、点和文本提示执行推理
        sam_model(source, bboxes=[439, 437, 524, 709], points=[[200, 200]], labels=[1], texts="a photo of a dog")


def test_mobilesam() -> None:
    """使用 Ultralytics 和点提示、框提示测试 MobileSAM 分割。"""
    from ultralytics import SAM

    # 加载模型
    model = SAM(WEIGHTS_DIR / "mobile_sam.pt")

    # 输入源
    source = ASSETS / "zidane.jpg"

    # 根据一维点提示和一维标签预测分割结果。
    model.predict(source, points=[900, 370], labels=[1])

    # 根据三维点和二维标签预测分割结果（每个对象包含多个点）。
    model.predict(source, points=[[[900, 370], [1000, 100]]], labels=[[1, 1]])

    # 根据边界框提示预测分割结果
    model.predict(source, bboxes=[439, 437, 524, 709], save=True)

    # 预测全部结果
    # model(source)


# 慢速测试 -----------------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason="DDP is not available")
def test_train_gpu(task: str, model: str, data: str) -> None:
    """使用 GPU 测试 YOLO 在各种任务和模型上的训练。"""
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0")  # 单 GPU
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0,1")  # 多 GPU


@pytest.mark.parametrize(
    "solution",
    ["count", "blur", "workout", "heatmap", "isegment", "visioneye", "speed", "queue", "analytics", "trackzone"],
)
def test_solutions(solution: str) -> None:
    """测试 yolo solutions 命令行模式。"""
    run(f"yolo solutions {solution} verbose=False")
