# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import os
import shutil
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from itertools import product
from pathlib import Path
from types import SimpleNamespace

if sys.platform == "win32":
    os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")

import pytest
import torch

from tests import MODEL, SOURCE
from tests.conftest import isolated_model_path
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS, _handle_deprecation, get_cfg
from ultralytics.engine.exporter import EXPORT_ENVS, Exporter, export_formats, validate_args
from ultralytics.utils import (
    ARM64,
    IS_RASPBERRYPI,
    LINUX,
    MACOS,
    MACOS_VERSION,
    WEIGHTS_DIR,
    WINDOWS,
    checks,
)
from ultralytics.utils.export.engine import modelopt_quantize_onnx, torch2onnx
from ultralytics.utils.torch_utils import (
    TORCH_1_10,
    TORCH_1_11,
    TORCH_1_13,
    TORCH_2_0,
    TORCH_2_1,
    TORCH_2_9,
)


def skip_rpi_semantic(task):
    """由于内存限制，在 Raspberry Pi 上跳过语义分割导出测试。."""
    if IS_RASPBERRYPI and task == "semantic":
        pytest.skip("Semantic segmentation export tests are skipped on Raspberry Pi due to memory constraints.")


@pytest.mark.parametrize("end2end", [False, True])
def test_export_torchscript(end2end, isolated_model):
    """测试 YOLO 模型导出为 TorchScript 格式的兼容性和正确性。."""
    file = YOLO(isolated_model).export(format="torchscript", imgsz=32, end2end=end2end)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx(end2end, isolated_model):
    """使用动态轴测试 YOLO 模型导出为 ONNX 格式。."""
    file = YOLO(isolated_model).export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
@pytest.mark.parametrize("precision", [{"int8": True}, {"quantize": 8}])
def test_export_onnx_int8(isolated_model, precision):
    """通过旧版 int8 别名和统一的 quantize 参数测试 INT8 ONNX 导出。."""
    file = YOLO(isolated_model).export(format="onnx", data=Path("coco8.yaml"), fraction=0.25, imgsz=32, **precision)
    assert Path(file).name.endswith("_int8.onnx")
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
    Path(file).unlink()  # cleanup


def test_onnx_int8_quantize_excludes_non_weighted_ops(monkeypatch):
    """检查 ONNX INT8 只量化带权算子，同时保持字符串返回值约定。."""
    import onnx
    import onnxruntime.quantization as ort_quantization

    from ultralytics.utils.export.onnx import onnx_int8_quantize

    calls = {}
    graph = SimpleNamespace(
        node=[
            SimpleNamespace(name="conv", op_type="Conv"),
            SimpleNamespace(name="pool", op_type="MaxPool"),
            SimpleNamespace(name="sigmoid", op_type="Sigmoid"),
        ]
    )

    monkeypatch.setattr(onnx, "load", lambda _: SimpleNamespace(graph=graph))
    monkeypatch.setattr(ort_quantization, "quantize_static", lambda *args, **kwargs: calls.update(kwargs))
    result = onnx_int8_quantize(Path("model.onnx"), Path("model_int8.onnx"), [], lambda x: x)
    assert result == "model_int8.onnx"
    assert calls["nodes_to_exclude"] == ["pool", "sigmoid"]


def test_quantize_canonicalization():
    """Quantize 接受 8/16/32（整数或字符串）及 w 表示法，并规范化为整数形式（未设置时保持 None）。."""
    for value, expected in [
        (8, 8),
        (16, 16),
        (32, 32),
        ("8", 8),
        ("int8", 8),
        ("INT8", 8),
        ("w8a8", 8),
        ("W8A8", 8),
        ("fp16", 16),
        ("Fp16", 16),
        ("w16a16", 16),
        ("fp32", 32),
        ("fP32", 32),
        ("w8a16", "w8a16"),
        ("W8a16", "w8a16"),
        ("w8a32", "w8a32"),
        ("W8A32", "w8a32"),
    ]:
        assert get_cfg(overrides={"quantize": value}).quantize == expected
    assert get_cfg().quantize is None  # 未设置时默认使用 FP32
    with pytest.raises(ValueError, match="quantize"):
        get_cfg(overrides={"quantize": "x4"})
    with pytest.raises(ValueError, match="quantize"):
        get_cfg(overrides={"quantize": "a8w8"})


def test_quantize_deprecation():
    """旧版 half/int8 参数在所有模式下都会转发到统一的 quantize 参数；发生冲突时 int8 优先。."""
    assert _handle_deprecation({"int8": True})["quantize"] == 8
    assert _handle_deprecation({"half": True})["quantize"] == 16
    assert _handle_deprecation({"half": True, "int8": True})["quantize"] == 8  # int8 wins
    assert "half" not in _handle_deprecation({"half": True})  # 旧标志转发后会被移除
    assert _handle_deprecation({"half": True, "quantize": None})["quantize"] is None  # explicit quantize wins
    assert _handle_deprecation({"half": True, "quantize": 8})["quantize"] == 8  # explicit quantize still wins


def test_benchmark_forwards_legacy_precision(monkeypatch):
    """model.benchmark(half=True) 必须以 quantize=16 传给基准测试调用，不能悄悄按 FP32 运行。."""
    import ultralytics.utils.benchmarks as bm

    captured = {}
    monkeypatch.setattr(bm, "benchmark", lambda **kw: captured.update(kw) or {})
    YOLO(MODEL).benchmark(half=True, format="onnx", data="coco8.yaml")
    assert captured["quantize"] == 16, f"legacy half was dropped: quantize={captured.get('quantize')}"


def test_qnn_quantize_requires_w8a16():
    """QNN 导出采用 W8A16；不支持显式的 INT8 激活量化。."""
    valid_args = ["batch", "data", "dynamic", "fraction", "keras", "nms"]
    validate_args("qnn", SimpleNamespace(quantize="w8a16"), valid_args)
    with pytest.raises(AssertionError, match=r"quantize=8 \(INT8\) is not supported"):
        validate_args("qnn", SimpleNamespace(quantize=8), valid_args)


def test_modelopt_quantize_onnx_requires_int8_dataset():
    """检查缺少校准数据时 INT8 ModelOpt 量化会提前失败。."""
    with pytest.raises(ValueError, match="requires a calibration dataset"):
        modelopt_quantize_onnx("model.onnx", quantize=8)


def test_int8_calibration_validates_split():
    """检查 INT8 校准会拒绝不存在的数据集划分。."""
    exporter = object.__new__(Exporter)
    exporter.model = SimpleNamespace(task="obb")
    exporter.args = SimpleNamespace(data="coco8.yaml", split="trainval")
    exporter.imgsz = [32]
    with pytest.raises(FileNotFoundError, match="trainval"):
        exporter.get_int8_calibration_dataloader()


def test_export_rknn_batch_expansion(monkeypatch, tmp_path):
    """检查 RKNN 会先以批次 1 进行校准，再由 Toolkit 扩展到请求的批次大小。."""
    calls = {}
    monkeypatch.setattr(
        "ultralytics.utils.export.rknn.onnx2rknn", lambda **kwargs: calls.update(kwargs) or kwargs["output_dir"]
    )
    monkeypatch.setattr("ultralytics.engine.exporter.file_size", lambda _: 1)

    image = tmp_path / "image.jpg"
    exporter = SimpleNamespace(
        args=SimpleNamespace(opset=None, quantize=8, name="rk3588", batch=8),
        im=torch.zeros(8, 3, 32, 32),
        file=tmp_path / "model.pt",
        metadata={},
        get_int8_calibration_dataloader=lambda prefix: SimpleNamespace(dataset=SimpleNamespace(im_files=[image])),
    )
    exporter.export_onnx = lambda: calls.update(onnx_batch=len(exporter.im)) or tmp_path / "model.onnx"
    Exporter.export_rknn(exporter)
    assert calls["onnx_batch"] == 1
    assert calls["batch"] == 8


def test_torch2onnx_serializes_concurrent_exports(monkeypatch, tmp_path):
    """确保不同 worker 线程的 ONNX 导出不会相互重叠。."""
    active = 0
    max_active = 0
    errors = []
    state_lock = threading.Lock()

    def fake_export(*args, **kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    def export_model(index: int):
        try:
            torch2onnx(torch.nn.Identity(), torch.zeros(1, 3, 8, 8), str(tmp_path / f"export-{index}.onnx"))
        except Exception as error:  # pragma: no cover - assertion handled below
            errors.append(error)

    threads = [threading.Thread(target=export_model, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"Concurrent export errors: {errors}"
    assert max_active == 1, f"Expected max 1 concurrent export, got {max_active}"


@pytest.mark.skipif(not TORCH_2_1, reason="OpenVINO requires torch>=2.1")
@pytest.mark.parametrize("end2end", [False, True])
def test_export_openvino(end2end, isolated_model):
    """测试 YOLO 导出为 OpenVINO 格式时的模型推理兼容性。."""
    file = YOLO(isolated_model).export(format="openvino", imgsz=32, end2end=end2end)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_2_1, reason="OpenVINO requires torch>=2.1")
@pytest.mark.parametrize(
    "task, dynamic, quantize, batch, nms, end2end",
    [  # 生成除排除项外的所有组合
        (task, dynamic, quantize, batch, nms, end2end)
        for task, dynamic, quantize, batch, nms, end2end in product(
            sorted(TASKS), [True, False], [8, 16], [1, 2], [True, False], [True]
        )
        if not ((task == "classify" and nms) or (end2end and nms))
    ],
)
# 暂时禁用 end2end=False 测试，因为 OpenVINO 测试期间 GitHub runner 会发生内存不足
def test_export_openvino_matrix(task, dynamic, quantize, batch, nms, end2end):
    """在各种配置矩阵条件下测试 YOLO 模型导出为 OpenVINO。."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(
        format="openvino",
        imgsz=32,
        dynamic=dynamic,
        quantize=quantize,
        batch=batch,
        data=TASK2DATA[task],  # 使用最小任务数据集以加快 INT8 校准
        nms=nms,
        end2end=end2end,
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32, batch=batch)  # exported model inference
    shutil.rmtree(file, ignore_errors=True)  # 重试，以防仍存在多线程文件使用错误


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, batch, simplify, nms, end2end",
    [  # 生成除排除项外的所有组合
        (task, dynamic, batch, simplify, nms, end2end)
        for task, dynamic, batch, simplify, nms, end2end in product(
            sorted(TASKS), [True, False], [1, 2], [True, False], [True, False], [True, False]
        )
        if not ((task == "classify" and nms) or (nms and not TORCH_1_13) or (end2end and nms))
    ],
)
def test_export_onnx_matrix(task, dynamic, batch, simplify, nms, end2end):
    """使用各种配置和参数测试 YOLO 导出为 ONNX 格式。."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx",
        imgsz=32,
        dynamic=dynamic,
        batch=batch,
        simplify=simplify,
        nms=nms,
        end2end=end2end,
    )
    r = YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference
    if task == "semantic":
        assert r[0].semantic_mask is not None
        assert r[0].semantic_mask.data.dtype in {torch.uint8, torch.int32}
    Path(file).unlink()  # cleanup


def test_export_onnx_semantic_dnn():
    """使用 OpenCV DNN 测试语义 ONNX 类别映射输出。."""
    skip_rpi_semantic("semantic")
    file = YOLO(TASK2MODEL["semantic"]).export(format="onnx", imgsz=32)
    r = YOLO(file).predict(SOURCE, imgsz=32, dnn=True)
    assert r[0].semantic_mask is not None
    Path(file).unlink()


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, batch, nms, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, batch, nms, end2end)
        for task, dynamic, batch, nms, end2end in product(
            sorted(TASKS), [False, True], [1, 2], [True, False], [True, False]
        )
        if not ((task == "classify" and nms) or (end2end and nms))
    ],
)
def test_export_torchscript_matrix(task, dynamic, batch, nms, end2end, tmp_path):
    """在不同配置下测试 YOLO 模型导出为 TorchScript 格式。."""
    skip_rpi_semantic(task)
    file = YOLO(isolated_model_path(tmp_path, WEIGHTS_DIR / TASK2MODEL[task])).export(
        format="torchscript", imgsz=32, dynamic=dynamic, batch=batch, nms=nms, end2end=end2end
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not MACOS, reason="CoreML inference only supported on macOS")
@pytest.mark.skipif(not TORCH_1_11, reason="CoreML export requires torch>=1.11")
@pytest.mark.skipif(
    MACOS and MACOS_VERSION and MACOS_VERSION >= "15", reason="CoreML YOLO26 matrix test crashes on macOS 15+"
)
@pytest.mark.parametrize(
    "task, dynamic, quantize, nms, batch, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, quantize, nms, batch, end2end)
        for task, dynamic, quantize, nms, batch, end2end in product(
            sorted(TASKS), [True, False], [8, 16], [True, False], [1], [True, False]
        )
        if not (task not in {"detect", "segment", "pose"} and nms)
        and not (dynamic and nms)
        and not (task == "classify" and dynamic)
        and not (end2end and nms)
    ],
)
def test_export_coreml_matrix(task, dynamic, quantize, nms, batch, end2end):
    """使用各种参数配置测试 YOLO 导出为 CoreML 格式。."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(
        format="coreml",
        imgsz=32,
        dynamic=dynamic,
        quantize=quantize,
        batch=batch,
        nms=nms,
        end2end=end2end,
    )
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    shutil.rmtree(file)  # cleanup


@pytest.mark.skipif(not TORCH_1_11, reason="CoreML export requires torch>=1.11")
@pytest.mark.skipif(WINDOWS, reason="CoreML not supported on Windows")  # RuntimeError: BlobWriter not loaded
@pytest.mark.skipif(LINUX and ARM64, reason="CoreML not supported on aarch64 Linux")
@pytest.mark.skipif(
    MACOS and checks.IS_PYTHON_MINIMUM_3_13,
    reason="coremltools deadlocks after OpenVINO on macOS Python 3.13 (conflicting OpenMP runtimes)",
)
@pytest.mark.parametrize("format", ["coreml", "mlmodel"])
def test_export_coreml(isolated_model, format, monkeypatch, tmp_path):
    """测试 YOLO 导出为 CoreML 格式并检查错误。."""
    from ultralytics.utils.export import coreml

    quantize, torch2coreml = [], coreml.torch2coreml

    def capture_quantize(*args, **kwargs):
        quantize.append(kwargs["quantize"])
        return torch2coreml(*args, **kwargs)

    monkeypatch.setattr(coreml, "torch2coreml", capture_quantize)
    # 捕获 stdout 和 stderr
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        file = YOLO(isolated_model_path(tmp_path, WEIGHTS_DIR / "yolo11n.pt")).export(
            format=format, nms=True, imgsz=32, iou=0.42, conf=0.24
        )
        import coremltools as ct

        spec = ct.utils.load_spec(str(file))
        metadata = spec.description.metadata
        assert metadata.author and metadata.shortDescription and metadata.license and metadata.versionString
        assert metadata.userDefined["IoU threshold"] == "0.42"
        assert metadata.userDefined["Confidence threshold"] == "0.24"
        assert all(key in metadata.userDefined for key in ("names", "stride", "task"))
        assert next(iter(spec.pipeline.models[1].nonMaximumSuppression.stringClassLabels.vector)) == "person"
        assert [output.name for output in spec.description.output] == ["confidence", "coordinates"]
        if MACOS:
            file = YOLO(isolated_model).export(format="coreml", imgsz=32)
            YOLO(file)(SOURCE, imgsz=32)  # model prediction only supported on macOS for nms=False models

    # 检查捕获的输出中是否存在错误
    output = stdout.getvalue() + stderr.getvalue()
    assert quantize[0] == (16 if format == "coreml" else None)
    assert "Error" not in output, f"CoreML export produced errors: {output}"
    assert "You will not be able to run predict()" not in output, "CoreML export has predict() error"


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR CoreML export requires torch>=1.11")
@pytest.mark.skipif(WINDOWS, reason="CoreML not supported on Windows")
@pytest.mark.skipif(LINUX and ARM64, reason="CoreML not supported on aarch64 Linux")
@pytest.mark.skipif(
    MACOS and checks.IS_PYTHON_MINIMUM_3_13,
    reason="coremltools deadlocks after OpenVINO on macOS Python 3.13 (conflicting OpenMP runtimes)",
)
def test_export_coreml_rtdetr():
    """测试 RT-DETR 导出为 CoreML 格式并检查错误。."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        file = YOLO(WEIGHTS_DIR / "rtdetr-l.pt").export(format="coreml", imgsz=160)
        import coremltools as ct

        shape = ct.models.MLModel(str(file)).get_spec().description.output[0].type.multiArrayType.shape
        assert shape[-2] == 300
        if MACOS:
            YOLO(file)(SOURCE, imgsz=160)

    output = stdout.getvalue() + stderr.getvalue()
    assert "Error" not in output, f"RTDETR CoreML export produced errors: {output}"
    assert "You will not be able to run predict()" not in output, "RTDETR CoreML export has predict() error"


@pytest.mark.skipif(True, reason="Test disabled")
@pytest.mark.skipif(not LINUX, reason="TF suffers from install conflicts on Windows and macOS")
def test_export_pb(isolated_model):
    """测试 YOLO 导出为 TensorFlow 的 Protobuf（*.pb）格式。."""
    model = YOLO(isolated_model)
    file = model.export(format="pb", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)


@pytest.mark.skipif(True, reason="Test disabled as Paddle protobuf and ONNX protobuf requirements conflict.")
def test_export_paddle(isolated_model):
    """测试 YOLO 导出为 Paddle 格式，并注明 protobuf 与 ONNX 存在冲突。."""
    YOLO(isolated_model).export(format="paddle", imgsz=32)


@pytest.mark.skipif(not TORCH_1_10, reason="MNN export requires torch>=1.10")
@pytest.mark.skipif(
    LINUX and checks.IS_PYTHON_MINIMUM_3_13,
    reason="MNN ONNX-parser protobuf conflicts with TensorFlow protobuf>=6.31.1 loaded earlier in the shared Python 3.13 test process",
)
def test_export_mnn(isolated_model):
    """测试 YOLO 导出为 MNN 格式（警告：MNN 测试必须先于 NCNN 测试，否则 Windows CI 会报错）。."""
    file = YOLO(isolated_model).export(format="mnn", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.parametrize(
    "model,kwargs,error",
    [
        ("yolo11n.yaml", {"batch": 2, "dynamic": True, "nms": True}, "combining"),
        ("yolo11n-seg.yaml", {"nms": True}, "only supports detect and pose"),
        ("yolo11n-obb.yaml", {"nms": True}, "only supports detect and pose"),
    ],
)
def test_export_mnn_rejects_unsupported_nms(model, kwargs, error):
    """测试 MNN 会拒绝运行时失败或丢失任务输出的 NMS 组合。."""
    with pytest.raises(ValueError, match=error):
        YOLO(model).export(format="mnn", imgsz=32, **kwargs)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model,task,kwargs",
    [
        ("yolo11n.yaml", "detect", {"batch": 2, "dynamic": True}),
        ("yolo11n.yaml", "detect", {"nms": True}),
        ("yolo11n-pose.yaml", "pose", {"nms": True}),
    ],
)
def test_export_mnn_options(model, task, kwargs):
    """通过推理测试 MNN 动态形状和支持的内嵌 NMS 任务。."""
    batch = kwargs.get("batch", 1)
    file = YOLO(model).export(format="mnn", imgsz=32, **kwargs)
    assert len(YOLO(file, task=task)([SOURCE] * batch, imgsz=64 if kwargs.get("dynamic") else 32)) == batch
    Path(file).unlink()


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_1_10, reason="MNN export requires torch>=1.10")
@pytest.mark.parametrize(
    "task, quantize, batch, end2end",
    [  # generate all combinations except for exclusion cases
        (task, quantize, batch, end2end)
        for task, quantize, batch, end2end in product(sorted(TASKS), [8, 16], [1, 2], [True, False])
    ],
)
def test_export_mnn_matrix(task, quantize, batch, end2end):
    """考虑各种导出配置，测试 YOLO 导出为 MNN 格式。."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(format="mnn", imgsz=32, quantize=quantize, batch=batch, end2end=end2end)
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.skipif(not TORCH_2_0, reason="NCNN inference causes segfault on PyTorch<2.0")
def test_export_ncnn(isolated_model):
    """测试 YOLO 导出为 NCNN 格式。."""
    file = YOLO(isolated_model).export(format="ncnn", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_2_0, reason="NCNN inference causes segfault on PyTorch<2.0")
@pytest.mark.parametrize("task, quantize, batch", list(product(sorted(TASKS), [16], [1])))
def test_export_ncnn_matrix(task, quantize, batch):
    """考虑各种导出配置，测试 YOLO 导出为 NCNN 格式。."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(format="ncnn", imgsz=32, quantize=quantize, batch=batch)
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    shutil.rmtree(file, ignore_errors=True)  # retry in case of potential lingering multi-threaded file usage errors


@pytest.mark.skipif(not TORCH_2_9, reason="IMX export requires torch>=2.9.0")
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_9, reason="IMX export requires Python>=3.9")
@pytest.mark.skipif(not LINUX, reason="IMX export only supported on Linux")
@pytest.mark.skipif(
    IS_RASPBERRYPI, reason="Test disabled as IMX export suffers from OOM (Out of Memory) on Raspberry Pi 5 16GB"
)
def test_export_imx():
    """测试 YOLO 导出为 IMX 格式。."""
    model = YOLO("yolo11n.pt")  # IMX 导出仅支持 YOLO11
    file = model.export(format="imx", imgsz=32, data="coco8.yaml")
    YOLO(file)(SOURCE, imgsz=32)


@pytest.mark.slow
@pytest.mark.skipif(not LINUX or ARM64, reason="RKNN export only supported on non-aarch64 Linux")
@pytest.mark.parametrize("quantize,batch", [(8, 8), (16, 1)])
def test_export_rknn(isolated_model, quantize, batch):
    """测试 YOLO 导出为 RKNN 格式。."""
    file = YOLO(isolated_model).export(format="rknn", imgsz=32, quantize=quantize, batch=batch, data="coco8.yaml")
    assert next(Path(file).rglob("*.rknn"), None), f"RKNN export failed, no RKNN model found in: {file}"
    shutil.rmtree(file, ignore_errors=True)


# @pytest.mark.skipif(True, reason="Disabled for debugging ruamel.yaml installation required by executorch")
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10 or not TORCH_2_9, reason="Requires Python>=3.10 and Torch>=2.9.0")
@pytest.mark.skipif(WINDOWS, reason="Skipping test on Windows")
def test_export_executorch(isolated_model):
    """测试 YOLO 模型导出为 ExecuTorch 格式。."""
    file = YOLO(isolated_model).export(format="executorch", imgsz=32)
    assert Path(file).exists(), f"ExecuTorch export failed, directory not found: {file}"
    # 检查导出目录中是否存在 .pte 文件
    pte_file = Path(file) / "model.pte"
    assert pte_file.exists(), f"ExecuTorch .pte file not found: {pte_file}"
    # 检查 metadata.yaml 是否存在
    metadata_file = Path(file) / "metadata.yaml"
    assert metadata_file.exists(), f"ExecuTorch metadata.yaml not found: {metadata_file}"
    # 注意：跳过推理测试，因为 ExecuTorch 需要特殊的运行时配置
    shutil.rmtree(file, ignore_errors=True)  # 清理


@pytest.mark.slow
@pytest.mark.skipif(not (MACOS or (LINUX and not ARM64)), reason="LiteRT export only supported on Linux x86 and macOS")
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="litert-torch requires Python>=3.10")
@pytest.mark.parametrize(
    "task, quantize",
    [(task, quantize) for task in sorted(TASKS) for quantize in (None, 8, "w8a16", "w8a32")],
)
def test_export_litert_matrix(task, quantize):
    """为各种任务测试 YOLO 导出为 LiteRT 格式（FP32、静态 INT8、静态 w8a16 和动态 w8a32）。."""
    file = Path(YOLO(TASK2MODEL[task]).export(format="litert", imgsz=32, quantize=quantize))
    assert file.is_file() and file.suffix == ".tflite", f"LiteRT export is not a single .tflite for '{task}': {file}"
    # 约定：导出结果保持 float32 图输入输出（int8/int16 仅在内部使用），因此下游运行时读写浮点数，
    # 不需要在边界处进行量化/反量化；int8/int16 输入输出回归会导致设备端使用方静默失效。
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(file))
    interpreter.allocate_tensors()
    io_details = interpreter.get_input_details() + interpreter.get_output_details()
    assert all(d["dtype"] == np.float32 for d in io_details), (
        f"LiteRT '{task}' quantize={quantize} must keep float32 I/O, got {[d['dtype'] for d in io_details]}"
    )
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference (also exercises the embedded metadata)
    file.unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10 or not TORCH_2_9, reason="Requires Python>=3.10 and Torch>=2.9.0")
@pytest.mark.skipif(WINDOWS, reason="Skipping test on Windows")
@pytest.mark.parametrize("task", sorted(TASKS))
def test_export_executorch_matrix(task):
    """为各种任务类型测试 YOLO 导出为 ExecuTorch 格式。."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(format="executorch", imgsz=32)
    assert Path(file).exists(), f"ExecuTorch export failed for task '{task}', directory not found: {file}"
    # 检查导出目录中是否存在 .pte 文件
    pte_file = Path(file) / "model.pte"
    assert pte_file.exists(), f"ExecuTorch .pte file not found for task '{task}': {pte_file}"
    # 检查是否存在 metadata.yaml
    metadata_file = Path(file) / "metadata.yaml"
    assert metadata_file.exists(), f"ExecuTorch metadata.yaml not found for task '{task}': {metadata_file}"
    # Note: Inference testing skipped as ExecuTorch requires special runtime setup
    shutil.rmtree(file, ignore_errors=True)  # cleanup


@pytest.mark.skipif(
    not (WINDOWS or LINUX) or sys.version_info < (3, 11),
    reason="onnxruntime-qnn ships prebuilt wheels only for Windows and Linux on Python>=3.11",
)
def test_export_qnn(isolated_model):
    """通过 ONNX Runtime QNN 执行提供程序测试 YOLO 导出为 Qualcomm QNN 格式。."""
    import importlib.util

    # QNN EP 以 'onnxruntime_qnn' 插件模块或 onnxruntime/capi 中捆绑的 provider 库形式提供。
    has_qnn = importlib.util.find_spec("onnxruntime_qnn") is not None
    if not has_qnn and importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime

        capi = Path(onnxruntime.__file__).parent / "capi"
        has_qnn = (capi / "libonnxruntime_providers_qnn.so").exists() or (
            capi / "onnxruntime_providers_qnn.dll"
        ).exists()
    if not has_qnn:
        pytest.skip("onnxruntime-qnn / QNN Execution Provider not available")
    file = YOLO(isolated_model).export(format="qnn", imgsz=32)
    assert Path(file).is_file() and file.endswith("_qnn.onnx"), f"QNN export failed, no context binary found: {file}"
    # 注意：此处不执行设备端推理，因为它需要 Qualcomm Snapdragon 硬件
    Path(file).unlink(missing_ok=True)  # cleanup


@pytest.mark.parametrize("env", [k for k, v in EXPORT_ENVS.items() if k != "base" or v["smoke"]])
def test_export_env_has_smoke(env):
    """确保每个非基础导出环境都声明了构建时导出冒烟测试。."""
    assert EXPORT_ENVS[env]["smoke"], f"export env '{env}' has no smoke command"


def test_every_format_env_is_registered():
    """确保每种导出格式都指向已注册的导出环境。."""
    for fmt, env in zip(export_formats()["Argument"], export_formats()["Env"]):
        assert env in EXPORT_ENVS, f"format '{fmt}' references unknown env '{env}'"
