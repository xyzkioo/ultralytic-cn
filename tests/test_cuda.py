# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from itertools import product
from pathlib import Path

import pytest
import torch

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, IS_JETSON, WEIGHTS_DIR
from ultralytics.utils.autodevice import GPUInfo
from ultralytics.utils.checks import check_amp, check_tensorrt
from ultralytics.utils.torch_utils import TORCH_1_13, parse_device

# 如果 CUDA 可用，尝试查找空闲设备
DEVICES = []
if CUDA_IS_AVAILABLE:
    if IS_JETSON:
        DEVICES = [0]  # NVIDIA Jetson 只有一个 GPU，且不完全支持 pynvml 库
    else:
        gpu_info = GPUInfo()
        gpu_info.print_status()
        autodevice_fraction = __import__("os").environ.get("YOLO_AUTODEVICE_FRACTION_FREE", 0.3)
        if idle_gpus := gpu_info.select_idle_gpu(
            count=2,
            min_memory_fraction=autodevice_fraction,
            min_util_fraction=autodevice_fraction,
        ):
            DEVICES = idle_gpus


def test_checks():
    """使用 torch 的 CUDA 函数验证 CUDA 设置。"""
    assert torch.cuda.is_available() == CUDA_IS_AVAILABLE
    assert torch.cuda.device_count() == CUDA_DEVICE_COUNT


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_amp():
    """测试 AMP 训练检查。"""
    model = YOLO("yolo26n.pt").model.to(f"cuda:{DEVICES[0]}")
    assert check_amp(model)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
@pytest.mark.parametrize(
    "task, dynamic, batch, simplify, nms",
    [  # 生成除排除项外的所有组合
        (task, dynamic, batch, simplify, nms)
        for task, dynamic, batch, simplify, nms in product(
            sorted(TASKS), [True, False], [1, 2], [True, False], [True, False]
        )
        if not ((task == "classify" and nms) or (task == "obb" and nms and (not TORCH_1_13 or IS_JETSON)))
    ],
)
def test_export_onnx_matrix(task, dynamic, batch, simplify, nms):
    """使用各种配置和参数测试 YOLO 导出为 ONNX 格式。"""
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx",
        imgsz=32,
        dynamic=dynamic,
        batch=batch,
        simplify=simplify,
        nms=nms,
        device=DEVICES[0],
        # opset=20 if nms else None,  # 修复使用 NMS 时的 ONNX Runtime 错误
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32, device=DEVICES[0])  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
@pytest.mark.parametrize(
    "task, dynamic, quantize, batch",
    [
        (task, dynamic, quantize, batch)
        # 限制 Jetson 任务覆盖范围以提升 CI 速度；完整任务覆盖仍在 GPU CI 中执行。
        # for task, dynamic, quantize, batch in product(TASKS, [True, False], [8, 16], [1, 2])
        for task, dynamic, quantize, batch in product(["detect"] if IS_JETSON else sorted(TASKS), [True], [8, 16], [2])
    ]
    + [("detect", False, 8, 2)],  # exercise TensorRT 7-10 implicit INT8 quantization on GPU CI
)
def test_export_engine_matrix(task, dynamic, quantize, batch):
    """使用各种配置测试 YOLO 模型导出为 TensorRT 格式，并运行推理。"""
    check_tensorrt()
    import tensorrt as trt

    is_trt11 = int(trt.__version__.split(".", 1)[0]) >= 11
    if not is_trt11 and quantize == 8 and dynamic:
        # TensorRT 7-10 的校准器路径无法量化动态形状模型；TensorRT 11 使用 ModelOpt 显式 Q/DQ
        pytest.skip("INT8 + dynamic export requires explicit quantization, available on TensorRT 11+")

    file = YOLO(TASK2MODEL[task]).export(
        format="engine",
        imgsz=32,
        dynamic=dynamic,
        quantize=quantize,
        batch=batch,
        data=TASK2DATA[task],  # 使用最小任务数据集以加快 INT8 校准
        workspace=1,  # 减少工作空间 GB 数，以降低测试期间的资源占用
        simplify=True,
        device=DEVICES[0],
    )
    model = YOLO(file)
    model([SOURCE] * batch, imgsz=64 if dynamic else 32, device=DEVICES[0])  # 导出模型推理
    model.val(data=TASK2DATA[task], imgsz=32, device=DEVICES[0], batch=batch)  # 导出模型验证
    Path(file).unlink()  # 清理
    if quantize == 8:
        Path(file).with_suffix(".cache").unlink(missing_ok=True)  # 清理 TensorRT 7-10 INT8 校准缓存
        Path(file).with_suffix(".int8.onnx").unlink(missing_ok=True)  # 清理 TensorRT 11 ModelOpt INT8 ONNX
    if quantize == 16:
        Path(file).with_suffix(".fp16.onnx").unlink(missing_ok=True)  # 清理 TensorRT 11 ModelOpt FP16 ONNX


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
@pytest.mark.parametrize("nc", [1, 3])
def test_semantic_loss_all_ignore_amp(nc):
    """当 sum() 在大 fp16 logits 上溢出为 inf 时，全 ignore 防护仍必须保持有限值（AMP 走 GPU 路径）。"""
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SemanticSegmentationModel
    from ultralytics.utils.loss import SemanticSegmentationLoss

    model = SemanticSegmentationModel(cfg="yolo26-sem.yaml", nc=nc, verbose=False)
    model.args = get_cfg()
    loss_fn = SemanticSegmentationLoss(model)
    preds = (torch.randn(1, nc, 64, 64, device=f"cuda:{DEVICES[0]}") + 50).half().requires_grad_()
    loss, items = loss_fn(preds, {"semantic_mask": torch.full((1, 64, 64), 255, dtype=torch.long)})
    assert torch.isfinite(loss).all() and all(torch.isfinite(x).all() for x in items.values())
    loss.backward()
    assert preds.grad is not None


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
@pytest.mark.skipif(IS_JETSON, reason="Edge devices not intended for training")
def test_train():
    """使用可用的 CUDA 设备在最小数据集上测试模型训练。"""
    device = tuple(DEVICES) if len(DEVICES) > 1 else DEVICES[0]
    expected = parse_device(device)  # canonical torch indices, e.g. physical ids translate under external CVD
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    results = YOLO(MODEL).train(data="coco8-grayscale.yaml", imgsz=64, epochs=1, device=DEVICES[0], batch=-1)
    model = YOLO(MODEL)
    results = model.train(data="coco8.yaml", imgsz=64, epochs=1, device=device, batch=15, compile=True)
    assert model.trainer.args.device == expected, "trained on wrong GPUs"
    assert model.trainer.device.index == int(expected.split(",")[0]), "trained on wrong GPU"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible, "CUDA_VISIBLE_DEVICES must never be mutated"
    results = YOLO(MODEL).train(data="coco128.yaml", imgsz=64, epochs=1, device=device, batch=15, val=False)
    # 单 GPU 和 DDP 都会返回指标（DDP 从保存的检查点中恢复指标）
    assert results is not None


@pytest.mark.skipif(not DEVICES or max(DEVICES) == 0, reason="requires an idle CUDA device with nonzero index")
@pytest.mark.skipif(IS_JETSON, reason="Edge devices not intended for training")
def test_train_cold_process_nonzero_device():
    """在 CUDA 冷启动状态下的新进程中使用非零 GPU 索引训练，复现真实 CLI 使用场景。

    已预热的 pytest 进程已经初始化 CUDA，因此只有不设置 CUDA_VISIBLE_DEVICES 的子进程，才能复现生产
    Pod（例如 Ultralytics Platform）中的冷启动设备选择。
    """
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    cmd = ["yolo", "train", f"model={MODEL}", "data=coco8.yaml", "imgsz=32", "epochs=1", f"device={max(DEVICES)}"]
    subprocess.run(cmd, check=True, env=env)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_predict_multiple_devices():
    """验证模型在 CPU 和 CUDA 设备上的预测一致性。"""
    model = YOLO("yolo26n.pt")

    # 测试 CPU
    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)
    assert str(model.device) == "cpu"

    # 测试 CUDA
    cuda_device = f"cuda:{DEVICES[0]}"
    model = model.to(cuda_device)
    assert str(model.device) == cuda_device
    _ = model(SOURCE)
    assert str(model.device) == cuda_device

    # 再次测试 CPU
    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)
    assert str(model.device) == "cpu"

    # 再次测试 CUDA
    model = model.to(cuda_device)
    assert str(model.device) == cuda_device
    _ = model(SOURCE)
    assert str(model.device) == cuda_device


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_track_exported_model():
    """在 GPU 上使用导出模型进行跟踪；导出后端会以单个 Tensor 返回原始预测结果。"""
    file = YOLO(MODEL).export(format="torchscript", imgsz=160, device=DEVICES[0])
    results = YOLO(file).track(SOURCE, imgsz=160, device=DEVICES[0])
    assert len(results[0].boxes)
    Path(file).unlink()  # cleanup


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_autobatch():
    """使用 autobatch 工具检查 YOLO 模型训练的最佳批次大小。"""
    from ultralytics.utils.autobatch import check_train_batch_size

    check_train_batch_size(YOLO(MODEL).model.to(f"cuda:{DEVICES[0]}"), imgsz=64, amp=True)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_utils_benchmarks(isolated_model):
    """分析 YOLO 模型性能，用于基准测试。"""
    from ultralytics.utils.benchmarks import ProfileModels

    # 预先导出动态引擎模型，用于动态推理
    YOLO(isolated_model).export(format="engine", imgsz=32, dynamic=True, batch=1, device=DEVICES[0])
    ProfileModels(
        [isolated_model],
        imgsz=32,
        quantize=32,
        min_time=1,
        num_timed_runs=3,
        num_warmup_runs=1,
        device=DEVICES[0],
    ).run()


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_predict_sam():
    """使用不同提示测试 SAM 模型预测。"""
    from ultralytics import SAM
    from ultralytics.models.sam import Predictor as SAMPredictor

    model = SAM(WEIGHTS_DIR / "sam2.1_b.pt")
    model.info()

    # 使用各种提示执行推理
    model(SOURCE, device=DEVICES[0])
    model(SOURCE, bboxes=[439, 437, 524, 709], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[900, 370], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[900, 370], labels=[1], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[[900, 370]], labels=[1], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[[[900, 370], [1000, 100]]], labels=[[1, 1]], device=DEVICES[0])

    # 测试预测器
    predictor = SAMPredictor(
        overrides={
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "imgsz": 1024,
            "model": WEIGHTS_DIR / "mobile_sam.pt",
            "device": DEVICES[0],
            "quantize": 16,
        }
    )
    predictor.set_image(ASSETS / "zidane.jpg")
    # predictor(bboxes=[439, 437, 524, 709])
    # predictor(points=[900, 370], labels=[1])
    predictor.reset_image()
