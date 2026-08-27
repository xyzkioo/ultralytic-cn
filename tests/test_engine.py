# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests import MODEL, SOURCE, TASK_MODEL_DATA
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo import classify, depth, detect, obb, pose, segment, semantic
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import ASSETS, DEFAULT_CFG, IS_RASPBERRYPI, WEIGHTS_DIR
from ultralytics.utils.torch_utils import unwrap_model


def test_func(*args, **kwargs):
    """测试用于验证回调注册的回调函数桩。"""
    print("callback test passed")


def test_export(monkeypatch, tmp_path):
    """通过添加回调并验证其执行情况，测试模型导出功能。"""
    monkeypatch.chdir(tmp_path)
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "on_export_start callback not registered"
    f = exporter(model=YOLO("yolo26n.yaml").model)
    YOLO(f)(SOURCE)  # 导出模型推理


@pytest.mark.parametrize(
    "trainer_cls,validator_cls,predictor_cls,data,model,weights",
    [
        (
            detect.DetectionTrainer,
            detect.DetectionValidator,
            detect.DetectionPredictor,
            "coco8.yaml",
            "yolo26n.yaml",
            MODEL,
        ),
        (
            segment.SegmentationTrainer,
            segment.SegmentationValidator,
            segment.SegmentationPredictor,
            "coco8-seg.yaml",
            "yolo26n-seg.yaml",
            WEIGHTS_DIR / "yolo26n-seg.pt",
        ),
        (
            classify.ClassificationTrainer,
            classify.ClassificationValidator,
            classify.ClassificationPredictor,
            "imagenet10",
            "yolo26n-cls.yaml",
            None,
        ),
        (obb.OBBTrainer, obb.OBBValidator, obb.OBBPredictor, "dota8.yaml", "yolo26n-obb.yaml", None),
        (pose.PoseTrainer, pose.PoseValidator, pose.PosePredictor, "coco8-pose.yaml", "yolo26n-pose.yaml", None),
        (
            semantic.SemanticSegmentationTrainer,
            semantic.SemanticSegmentationValidator,
            semantic.SemanticSegmentationPredictor,
            "cityscapes8.yaml",
            "yolo26n-sem.yaml",
            None,
        ),
        (depth.DepthTrainer, depth.DepthValidator, depth.DepthPredictor, "depth8.yaml", "yolo26-depth.yaml", None),
    ],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_task(trainer_cls, validator_cls, predictor_cls, data, model, weights):
    """测试 YOLO 在各种任务上的训练、验证和预测。"""
    overrides = {
        "data": data,
        "model": model,
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "mask_ratio": 1,
        "overlap_mask": False,
    }

    # 训练器
    trainer = trainer_cls(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "on_train_start callback not registered"
    trainer.train()

    # 验证器
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data
    cfg.imgsz = 32
    val = validator_cls(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "on_val_start callback not registered"
    val(model=trainer.best)

    # 预测器
    pred = predictor_cls(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "on_predict_start callback not registered"

    # 确定预测使用的模型路径
    model_path = weights if weights else trainer.best
    if model == "yolo26n.yaml":  # 仅适用于检测
        # 确认 sys.argv 为空时不会出现问题
        with mock.patch.object(sys, "argv", []):
            result = pred(source=ASSETS, model=model_path)
            assert len(result) > 0, f"Predictor returned no results for {model}"
    else:
        result = pred(source=ASSETS, model=model_path)
        assert len(result) > 0, f"Predictor returned no results for {model}"

    # 测试恢复训练功能
    with pytest.raises(AssertionError):
        trainer_cls(overrides={**overrides, "resume": trainer.last}).train()


@pytest.mark.parametrize("task,weight,data", TASK_MODEL_DATA)
def test_resume_incomplete(task, weight, data, tmp_path):
    """测试从不完整的检查点恢复训练。"""
    train_args = {
        "data": data,
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": task,
        "imgsz": 32,
        "exist_ok": True,
    }

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    def disable_final_eval(trainer):
        trainer.final_eval = lambda: None

    model = YOLO(weight)
    model.add_callback("on_train_start", disable_final_eval)
    model.add_callback("on_train_epoch_end", stop_after_first_epoch)
    model.train(**train_args)
    last_path = model.trainer.last
    _, ckpt = load_checkpoint(last_path)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"

    # 使用检查点恢复训练
    resume_model = YOLO(last_path)
    resume_model.train(resume=True, **train_args)
    assert resume_model.trainer.start_epoch == resume_model.trainer.epoch == 1, "resume test failed"


def test_distill_resume(tmp_path: Path):
    """测试从不完整的检查点恢复知识蒸馏。"""
    overrides = {
        "data": "coco8.yaml",
        "model": "yolo26n.yaml",
        "distill_model": WEIGHTS_DIR / "yolo26s.pt",
        "imgsz": 32,
        "multi_scale": 0.5,  # 改变每个批次的图像尺寸，以测试动态蒸馏分数拆分
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": "distill",
        "exist_ok": True,
    }

    # 训练一个周期后中断，以生成可恢复的检查点
    trainer = detect.DetectionTrainer(overrides=overrides)

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    trainer.final_eval = lambda: None
    trainer.add_callback("on_train_epoch_end", stop_after_first_epoch)
    trainer.train()
    _, ckpt = load_checkpoint(trainer.last)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"
    assert isinstance(ckpt["ema"], DistillationModel), "distillation EMA wraps the student model"
    assert ckpt["ema"].teacher_model is None, "teacher should be stripped from the EMA/checkpoint"
    assert ckpt["ema"].projector is not None, "the distillation projector should be persisted in the EMA checkpoint"

    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.final_eval = lambda: None
    trainer.train()
    model = unwrap_model(trainer.model)
    assert isinstance(model, DistillationModel), "resume should rebuild the DistillationModel"
    assert model.teacher_model is not None, "resume should rebuild the teacher from the distill_model path"
    assert trainer.start_epoch == trainer.epoch == 1, "resume test failed"


def test_distill_grayscale(tmp_path: Path):
    """测试在单通道数据集上的知识蒸馏（https://github.com/ultralytics/ultralytics/issues/25066）。"""
    teacher = DetectionModel("yolo26n.yaml", ch=3, nc=80, verbose=False)
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model": teacher}, teacher_path)
    student = DetectionModel("yolo26n.yaml", ch=1, nc=80, verbose=False)
    student.args = SimpleNamespace(imgsz=32, dis=1.0)
    model = DistillationModel(teacher_model=teacher_path, student_model=student)
    assert isinstance(model, DistillationModel)
    assert model.teacher_model.yaml["channels"] == 1


@pytest.mark.parametrize(
    "ckpt",
    [
        {"model": OrderedDict([("a", torch.zeros(1))])},  # state_dict 保存在 "model" 键下
        {"model": {"a": torch.zeros(1)}},  # 普通字典形式的 "model" 值
        OrderedDict([("a", torch.zeros(1))]),  # 裸 state_dict，不包含 "model" 键
    ],
)
def test_load_checkpoint_state_dict_rejected(ckpt, tmp_path):
    """测试 state_dict 检查点会清晰地抛出 TypeError，而不是难以理解的 AttributeError/KeyError。"""
    weight = tmp_path / "bad.pt"
    torch.save(ckpt, weight)
    with pytest.raises(TypeError, match="supported Ultralytics checkpoint format"):
        load_checkpoint(weight)


def test_nan_recovery():
    """测试训练期间的 NaN 损失检测和恢复。"""
    nan_injected = [False]

    def inject_nan(trainer):
        """在批次处理期间向损失注入 NaN，以测试恢复机制。"""
        if trainer.epoch == 1 and trainer.tloss is not None and not nan_injected[0]:
            trainer.tloss[next(iter(trainer.tloss))] *= float("nan")
            nan_injected[0] = True

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 3}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", inject_nan)
    trainer.train()
    assert nan_injected[0], "NaN injection failed"


def test_checkpoint_fp16_overflow():
    """测试权重溢出 fp16 的有限模型仍会被保存为检查点（并进行截断），而不会被跳过。"""

    def inflate_ema(trainer):
        """将一个 EMA 权重推到 fp16 最大值（65504）以上，使其 fp16 快照原本会变成 Inf。"""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = 1.0e5

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 2}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", inflate_ema)
    trainer.train()
    assert trainer.last.exists(), "checkpoint not saved for a finite model with fp16-overflowing weights"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )
    # 验证必须保持实时 EMA 为 fp32 且不变；检查点序列化可以限制其 fp16 副本。
    ema_param = next(iter(trainer.ema.ema.parameters()))
    assert ema_param.dtype == torch.float32 and torch.isfinite(ema_param).all() and ema_param.flatten()[0] == 1.0e5, (
        "validation corrupted the live EMA"
    )


def test_checkpoint_nonfinite_ema_resync():
    """测试有限模型中的非有限 EMA 会重新同步而不是被跳过，使运行仍能生成检查点。"""

    def poison_ema(trainer):
        """让实时 fp32 EMA 真正变为非有限值，同时保持模型有限（有限损失运行中的持久 NaN）。"""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = float("inf")

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 2}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", poison_ema)
    trainer.train()
    assert trainer.last.exists(), "no checkpoint saved when the EMA went non-finite on a finite model"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )


def test_checkpoint_nonfinite_ema_and_model_sanitized():
    """测试 EMA 和模型中的非有限张量会被清理而不是跳过，使运行仍能生成检查点。"""

    def poison_ema_and_model(trainer):
        """强制实时 EMA 和模型中的第一个参数都变为非有限值（有限损失运行中的持久 NaN）。"""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = float("inf")
            next(iter(unwrap_model(trainer.model).parameters())).data.flatten()[0] = float("nan")

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 1}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", poison_ema_and_model)
    trainer.train()
    assert trainer.last.exists(), "no checkpoint saved when a tensor went non-finite in both EMA and model"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )


@pytest.mark.parametrize(
    "kwargs,uses_weights",
    [({}, True), ({"pretrained": True}, True), ({"pretrained": False}, False), ({"pretrained": MODEL}, True)],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_reuses_loaded_checkpoint_model(monkeypatch, kwargs, uses_weights):
    """测试训练会复用已加载的检查点配置，同时遵守 pretrained 参数。"""
    model = YOLO("yolo26n.yaml")
    model.ckpt = {"checkpoint": True}
    model.ckpt_path = "/tmp/fake.pt"
    model.overrides["model"] = "ul://glenn-jocher/m2/exp-14"
    model.overrides["pretrained"] = False
    original_model = model.model
    captured = {}

    class FakeTrainer:
        def __init__(self, overrides=None, _callbacks=None):
            self.overrides = overrides
            self.callbacks = _callbacks
            self.model = None
            self.validator = SimpleNamespace(metrics=None)
            self.best = MODEL.parent / "nonexistent-best.pt"
            self.last = MODEL
            captured["trainer"] = self

        def get_model(self, cfg=None, weights=None, verbose=True):
            captured["cfg"] = cfg
            captured["weights"] = weights
            return original_model

        def train(self):
            return None

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    monkeypatch.setattr(model, "_smart_load", lambda key: FakeTrainer)
    monkeypatch.setattr(
        "ultralytics.engine.model.load_checkpoint",
        lambda path: (original_model, {"checkpoint": True}),
    )

    model.train(data="coco8.yaml", epochs=1, **kwargs)

    assert captured["trainer"].model is original_model, "Trainer model does not match original"
    assert captured["cfg"] == original_model.yaml, f"Config mismatch: {captured['cfg']} != {original_model.yaml}"
    assert captured["weights"] is (original_model if uses_weights else None), "Unexpected weights loaded"


def test_train_multi_custom_trainer_metrics_and_failure_keys(monkeypatch, tmp_path):
    """测试自定义多数据集运行会保留内存指标和唯一失败键。"""
    model = YOLO(MODEL)
    calls = 0

    class FakeTrainer:
        def __init__(self, overrides=None, _callbacks=None):
            pass

        def get_model(self, cfg=None, weights=None, verbose=True):
            return model.model

        def train(self):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("failed repeated dataset")
            self.validator = SimpleNamespace(metrics=SimpleNamespace(results_dict={"fitness": 1.0}))

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    results = model.train(
        data=["coco8.yaml", "coco8.yaml"],
        project=tmp_path,
        plots=False,
        save=False,
        trainer=FakeTrainer,
    )

    assert model.trainer.trainer is FakeTrainer
    assert results == {"coco8": {"fitness": 1.0}, "coco8-2": None}


@pytest.mark.parametrize("pretrained,uses_weights", [(True, True), (False, False), (MODEL, True)])
def test_setup_model_respects_pretrained_arg_for_pt_models(monkeypatch, pretrained, uses_weights):
    """测试 .pt 模型会使用检查点配置，同时遵守 pretrained 参数。"""
    captured = {}
    checkpoint_model = SimpleNamespace(yaml={"nc": 80})
    trainer = object.__new__(BaseTrainer)
    trainer.model = "yolo26n.pt"
    trainer.args = SimpleNamespace(pretrained=pretrained)
    trainer.resume = False

    def fake_get_model(cfg=None, weights=None, verbose=True):
        captured["cfg"] = cfg
        captured["weights"] = weights
        return SimpleNamespace()

    trainer.get_model = fake_get_model
    monkeypatch.setattr(
        "ultralytics.engine.trainer.load_checkpoint", lambda path: (checkpoint_model, {"checkpoint": True})
    )

    trainer.setup_model()

    assert captured["cfg"] == checkpoint_model.yaml, "Checkpoint config was not used"
    assert captured["weights"] is (checkpoint_model if uses_weights else None), "Unexpected weights loaded"
