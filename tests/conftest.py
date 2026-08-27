# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
from pathlib import Path

import numpy.testing  # noqa: F401  # Pre-import before any test can corrupt numpy via in-place upgrade
import pytest


@pytest.fixture(scope="session")
def solution_assets():
    """按名称返回缓存的解决方案资源路径。"""
    from tests import SOLUTION_ASSETS
    from ultralytics.utils import ASSETS_URL, WEIGHTS_DIR
    from ultralytics.utils.downloads import safe_download

    cache_dir = WEIGHTS_DIR / "solution_assets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get_asset(name):
        asset_path = cache_dir / SOLUTION_ASSETS[name]
        if not asset_path.exists():
            safe_download(url=f"{ASSETS_URL}/{asset_path.name}", dir=cache_dir)
        return asset_path

    return get_asset


def pytest_addoption(parser):
    """为 pytest 添加自定义命令行选项。"""
    parser.addoption("--slow", action="store_true", default=False, help="Run slow tests")
    parser.addoption(
        "--export-env",
        default=None,
        help="Run only export tests assigned to this export environment id.",
    )


def _export_format_from_item(item, formats):
    """推断 tests/test_exports.py 测试项覆盖的导出格式。"""
    if Path(str(item.fspath)).name != "test_exports.py":
        return None
    name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
    if name == "test_torch2onnx_serializes_concurrent_exports":
        return "onnx"
    if not name.startswith("test_export_"):
        return None

    suffix = name[len("test_export_") :]
    return next(
        (fmt for fmt in sorted(formats, key=len, reverse=True) if suffix == fmt or suffix.startswith(f"{fmt}_")), None
    )


def pytest_collection_modifyitems(config, items):
    """当未指定 --slow 选项时，从测试项列表中排除标记为 slow 的测试。

    Args:
        config: 提供命令行选项访问权限的 pytest 配置对象。
        items (list): 已收集的 pytest 测试项对象列表，将根据是否存在 --slow 选项进行修改。
    """
    if not config.getoption("--slow"):
    # 如果测试项标记为 'slow'，则将其从测试项列表中完全移除
        items[:] = [item for item in items if "slow" not in item.keywords]

    export_env = config.getoption("--export-env")
    if not export_env:
        return

    from ultralytics.engine.exporter import export_formats

    env_by_format = dict(zip(export_formats()["Argument"], export_formats()["Env"]))
    for item in items:
        fmt = _export_format_from_item(item, env_by_format)
        if fmt and env_by_format.get(fmt) != export_env:
            item.add_marker(pytest.mark.skip(reason=f"export format '{fmt}' belongs to env '{env_by_format[fmt]}'"))


def isolated_model_path(tmp_path, model):
    """将模型复制到每个测试独立的路径，避免 pytest-xdist 下导出文件发生竞争。"""
    model = Path(model)
    if not model.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(attempt_download_asset(model.name), model)

    dst = tmp_path / model.name
    shutil.copy(model, dst)
    return str(dst)


def pytest_sessionstart(session):
    """初始化 pytest 的会话配置。

    pytest 会在创建“Session”对象后、收集测试前自动调用此函数。该函数会设置测试会话的初始随机种子。

    Args:
        session: pytest 会话对象。
    """
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds()


@pytest.fixture
def isolated_model(tmp_path):
    """提供测试模型的隔离副本，避免 pytest-xdist 下导出文件发生竞争。

    当多个 xdist worker 同时运行导出测试时，它们会根据模型路径（例如 model.onnx、model.torchscript）
    推导输出文件名。使用相同的 MODEL 路径会导致 worker 相互覆盖中间文件或导出文件。
    此 fixture 会将共享模型复制到每个测试独立的临时目录，使每个测试导出到唯一路径。
    """
    from tests import MODEL

    return isolated_model_path(tmp_path, MODEL)


def pytest_sessionfinish(session, exitstatus):
    """pytest 会话结束后的清理操作。

    仅在 pytest 控制器（或串行运行）上执行，并跳过 xdist worker，避免一个 worker 删除共享资源时另一个
    worker 仍在读取而产生竞争条件。
    """
    # 在 xdist worker 上跳过；只有控制器应清理共享资源
    if hasattr(session.config, "workerinput"):
        return

    from ultralytics.utils import WEIGHTS_DIR

    # 删除文件
    models = [path for x in ("*.onnx", "*.torchscript") for path in WEIGHTS_DIR.rglob(x)]
    for file in ["bus.jpg", "yolo26n.onnx", "yolo26n.torchscript", *models]:
        Path(file).unlink(missing_ok=True)

    # 删除目录
    for directory in [path for x in ("*.mlpackage", "*_openvino_model") for path in WEIGHTS_DIR.rglob(x)]:
        shutil.rmtree(directory, ignore_errors=True)
