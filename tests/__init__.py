# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks

# 模型、配置、数据源和环境信息的共享测试常量
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"  # 包含空格的路径，用于测试路径处理
CFG = "yolo26n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]  # 文件、目录和 glob 模式
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()
TASK_MODEL_DATA = sorted(
    [(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS]
)  # （任务、模型、数据）元组
MODELS = sorted([*list(TASK2MODEL.values()), "yolo11n-grayscale.pt"])  # 任务模型及灰度变体
SOLUTION_ASSETS = {
    "boats": "boats.jpg",
    "demo_video": "solutions_ci_demo.mp4",
    "crop_video": "decelera_landscape_min.mov",
    "pose_video": "solution_ci_pose_demo.mp4",
    "parking_video": "solution_ci_parking_demo.mp4",
    "vertical_video": "solution_vertical_demo.mp4",
    "track_video": "decelera_portrait_min.mov",
    "parking_areas": "solution_ci_parking_areas.json",
    "parking_model": "solutions_ci_parking_model.pt",
    "similarity_images": "4-imgs-similaritysearch.zip",
}

__all__ = (
    "CFG",
    "CUDA_DEVICE_COUNT",
    "CUDA_IS_AVAILABLE",
    "MODEL",
    "SOLUTION_ASSETS",
    "SOURCE",
    "SOURCES_LIST",
)
