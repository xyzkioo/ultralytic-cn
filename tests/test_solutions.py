# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# 测试 Ultralytics Solutions：https://docs.ultralytics.com/solutions
# 包含除 DistanceCalculation 和 Security Alarm System 外的所有解决方案。

import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from tests import MODEL
from ultralytics import solutions
from ultralytics.utils import IS_RASPBERRYPI, TORCH_VERSION, checks
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.torch_utils import TORCH_2_4

# 预定义参数值
SHOW = False
REGION = [(10, 200), (540, 200), (540, 180), (10, 180)]  # 用于目标计数、速度估计和队列管理
HORIZONTAL_LINE = [(10, 200), (540, 200)]  # 用于目标计数
VERTICAL_LINE = [(320, 0), (320, 400)]  # 用于目标计数


def process_video(solution, video_path: str, needs_frame_count: bool = False):
    """使用解决方案处理视频，将帧和可选的帧数传递给解决方案实例。"""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error reading video file {video_path}"

    frame_count = 0
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        frame_count += 1
        im_copy = im0.copy()
        args = [im_copy, frame_count] if needs_frame_count else [im_copy]
        _ = solution(*args)

    cap.release()


@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled for testing due to --slow test errors after YOLOE PR.")
@pytest.mark.parametrize(
    "name, solution_class, needs_frame_count, video_key, kwargs_update",
    [
        (
            "ObjectCounter",
            solutions.ObjectCounter,
            False,
            "demo_video",
            {"region": REGION, "model": MODEL, "show": SHOW},
        ),
        (
            "ObjectCounter",
            solutions.ObjectCounter,
            False,
            "demo_video",
            {"region": HORIZONTAL_LINE, "model": MODEL, "show": SHOW},
        ),
        (
            "ObjectCounterVertical",
            solutions.ObjectCounter,
            False,
            "vertical_video",
            {"region": VERTICAL_LINE, "model": MODEL, "show": SHOW},
        ),
        (
            "ObjectCounterwithOBB",
            solutions.ObjectCounter,
            False,
            "parking_video",  # yolo26n-obb.pt yields no tracks on demo_video, so it would not exercise the OBB path
            {"region": REGION, "model": "yolo26n-obb.pt", "show": SHOW},
        ),
        (
            "Heatmap",
            solutions.Heatmap,
            False,
            "demo_video",
            {"colormap": cv2.COLORMAP_PARULA, "model": MODEL, "show": SHOW, "region": None},
        ),
        (
            "HeatmapWithRegion",
            solutions.Heatmap,
            False,
            "demo_video",
            {"colormap": cv2.COLORMAP_PARULA, "region": REGION, "model": MODEL, "show": SHOW},
        ),
        (
            "SpeedEstimator",
            solutions.SpeedEstimator,
            False,
            "demo_video",
            {"region": REGION, "model": MODEL, "show": SHOW},
        ),
        (
            "QueueManager",
            solutions.QueueManager,
            False,
            "demo_video",
            {"region": REGION, "model": MODEL, "show": SHOW},
        ),
        (
            "LineAnalytics",
            solutions.Analytics,
            True,
            "demo_video",
            {"analytics_type": "line", "model": MODEL, "show": SHOW, "figsize": (6.4, 3.2)},
        ),
        (
            "PieAnalytics",
            solutions.Analytics,
            True,
            "demo_video",
            {"analytics_type": "pie", "model": MODEL, "show": SHOW, "figsize": (6.4, 3.2)},
        ),
        (
            "BarAnalytics",
            solutions.Analytics,
            True,
            "demo_video",
            {"analytics_type": "bar", "model": MODEL, "show": SHOW, "figsize": (6.4, 3.2)},
        ),
        (
            "AreaAnalytics",
            solutions.Analytics,
            True,
            "demo_video",
            {"analytics_type": "area", "model": MODEL, "show": SHOW, "figsize": (6.4, 3.2)},
        ),
        ("TrackZone", solutions.TrackZone, False, "demo_video", {"region": REGION, "model": MODEL, "show": SHOW}),
        (
            "ObjectCropper",
            solutions.ObjectCropper,
            False,
            "crop_video",
            {"temp_crop_dir": "cropped-detections", "model": MODEL, "show": SHOW},
        ),
        (
            "ObjectBlurrer",
            solutions.ObjectBlurrer,
            False,
            "demo_video",
            {"blur_ratio": 0.02, "model": MODEL, "show": SHOW},
        ),
        (
            "InstanceSegmentation",
            solutions.InstanceSegmentation,
            False,
            "demo_video",
            {"model": "yolo26n-seg.pt", "show": SHOW},
        ),
        ("VisionEye", solutions.VisionEye, False, "demo_video", {"model": MODEL, "show": SHOW}),
        (
            "RegionCounter",
            solutions.RegionCounter,
            False,
            "demo_video",
            {"region": REGION, "model": MODEL, "show": SHOW},
        ),
        ("AIGym", solutions.AIGym, False, "pose_video", {"kpts": [6, 8, 10], "show": SHOW}),
        (
            "ParkingManager",
            solutions.ParkingManagement,
            False,
            "parking_video",
            {"model": "parking_model", "show": SHOW, "json_file": "parking_areas"},
        ),
        (
            "StreamlitInference",
            solutions.Inference,
            False,
            None,  # Streamlit 应用不需要视频文件
            {},  # Streamlit 应用不接受参数
        ),
    ],
)
def test_solution(name, solution_class, needs_frame_count, video_key, kwargs_update, tmp_path, solution_assets):
    """通过视频处理和参数验证测试单个 Ultralytics 解决方案。"""
    video_path = str(solution_assets(video_key)) if video_key else None

    kwargs = {}
    for key, value in kwargs_update.items():
        if key.startswith("temp_"):
            kwargs[key.replace("temp_", "")] = str(tmp_path / value)
        elif value == "parking_model":
            kwargs[key] = str(solution_assets("parking_model"))
        elif value == "parking_areas":
            kwargs[key] = str(solution_assets("parking_areas"))
        else:
            kwargs[key] = value
    kwargs.setdefault("imgsz", 320)

    if name == "StreamlitInference":
        if checks.check_imshow():  # 不要与上面的 elif 合并
            solution_class(**kwargs).inference()  # 需要交互式 GUI 环境
        return

    process_video(
        solution=solution_class(**kwargs),
        video_path=video_path,
        needs_frame_count=needs_frame_count,
    )


@pytest.mark.parametrize(
    "region",
    [[(220, 140), (420, 140), (420, 340), (220, 340)], [(20, 360), (1080, 360), (1080, 400), (20, 400)]],
    ids=["square-zone", "wide-band"],
)
@pytest.mark.parametrize(
    "step, expected",
    [((8, 0), "IN"), ((-8, 0), "OUT"), ((0, 8), "IN"), ((0, -8), "OUT")],
    ids=["left-to-right", "right-to-left", "downward", "upward"],
)
def test_object_counter_polygon_direction(region, step, expected):
    """无论区域的宽高比如何，多边形计数都必须沿对象的运动轴进行。"""
    counter = solutions.ObjectCounter(region=region, show=False)
    counter.initialize_region()
    cx = sum(p[0] for p in region) / 4
    cy = sum(p[1] for p in region) / 4
    track = [(cx + (i - 5) * step[0], cy + (i - 5) * step[1]) for i in range(6)]  # ends at region center
    counter.track_history[1] = track
    counter.count_objects(track[-1], 1, track[-2], 0)
    counts = {"IN": counter.in_count, "OUT": counter.out_count}
    assert counts[expected] == 1 and counts["IN" if expected == "OUT" else "OUT"] == 0, (
        f"step {step} into {region} expected {expected}, got in={counter.in_count} out={counter.out_count}"
    )


def test_object_counter_polygon_reversal_at_entry():
    """轨迹先远离、转身再进入时，必须根据其穿越运动进行计数。"""
    counter = solutions.ObjectCounter(region=[(220, 140), (420, 140), (420, 340), (220, 340)], show=False)
    counter.initialize_region()
    # 从右边缘外向右移动，然后反向向左进入
    xs = [428, 436, 444, 436, 428, 420, 412]  # turn at x=444, entry at x=412
    track = [(float(x), 240.0) for x in xs]
    for i in range(2, len(track) + 1):
        counter.track_history[1] = track[:i]
        counter.count_objects(track[i - 1], 1, track[i - 2], 0)
    assert (counter.in_count, counter.out_count) == (0, 1), (
        f"entered moving left, expected OUT, got in={counter.in_count} out={counter.out_count}"
    )


def test_object_counter_polygon_reentry_after_inside_spawn():
    """轨迹在区域内部生成（第一帧不计数）、离开后再次进入时，必须根据其穿越运动进行计数。"""
    counter = solutions.ObjectCounter(region=[(220, 140), (420, 140), (420, 340), (220, 340)], show=False)
    counter.initialize_region()
    track = [(230.0, 240.0), (210.0, 240.0), (230.0, 240.0)]  # inside -> out the left edge -> re-enter rightward
    for i in range(1, len(track) + 1):
        counter.track_history[1] = track[:i]
        counter.count_objects(track[i - 1], 1, track[i - 2] if i > 1 else None, 0)
    assert (counter.in_count, counter.out_count) == (1, 0), (
        f"re-entered moving right, expected IN, got in={counter.in_count} out={counter.out_count}"
    )


def test_left_click_selection():
    """测试距离计算中的左键选择功能。"""
    dc = solutions.DistanceCalculation()
    dc.boxes, dc.track_ids = [[10, 10, 50, 50]], [1]
    dc.mouse_event_for_distance(cv2.EVENT_LBUTTONDOWN, 30, 30, None, None)
    assert 1 in dc.selected_boxes, f"Expected track_id 1 in selected_boxes, got {dc.selected_boxes}"


def test_left_click_selection_obb():
    """使用 (4, 2) OBB 角点框测试距离计算的左键选择。"""
    dc = solutions.DistanceCalculation()
    dc.boxes = [torch.tensor([[30.0, 10.0], [50.0, 30.0], [30.0, 50.0], [10.0, 30.0]])]  # rotated box corners
    dc.track_ids = [1]
    dc.mouse_event_for_distance(cv2.EVENT_LBUTTONDOWN, 30, 30, None, None)
    assert 1 in dc.selected_boxes, f"Expected track_id 1 in selected_boxes, got {dc.selected_boxes}"


@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled for testing due to --slow test errors after YOLOE PR.")
@pytest.mark.parametrize(
    "solution_class, extra_kwargs",
    [
        (solutions.Heatmap, {"colormap": cv2.COLORMAP_PARULA, "region": None}),
        (solutions.ObjectBlurrer, {"blur_ratio": 0.02}),
        (solutions.VisionEye, {}),
        (solutions.RegionCounter, {"region": REGION}),
        (solutions.ParkingManagement, {"json_file": "parking_areas"}),
    ],
    ids=["Heatmap", "ObjectBlurrer", "VisionEye", "RegionCounter", "ParkingManagement"],
)
def test_solution_obb_boxes(solution_class, extra_kwargs, solution_assets):
    """回归测试：使用 self.boxes 的解决方案必须能够处理 (4, 2) OBB 角点框而不崩溃。

    OBB 模型会用 (4, 2) 角点填充 self.boxes，而不是 xyxy 标量，get_enclosing_box 会对其进行归一化。
    单个 parking_video 帧就能生成 OBB 轨迹，因此无需处理完整视频即可覆盖每个崩溃位置。
    复用测试矩阵中已有的 parking_video 和 yolo26n-obb.pt 资源。
    """
    kwargs = {"model": "yolo26n-obb.pt", "show": SHOW, "imgsz": 320, **extra_kwargs}
    if kwargs.get("json_file") == "parking_areas":
        kwargs["json_file"] = str(solution_assets("parking_areas"))
    cap = cv2.VideoCapture(str(solution_assets("parking_video")))
    success, im0 = cap.read()
    cap.release()
    assert success, "Failed to read first frame of parking_video"
    results = solution_class(**kwargs)(im0)
    assert results.plot_im is not None, f"{solution_class.__name__} returned no plot_im on OBB input"


def test_object_blurrer_obb_outside_frame():
    """完全裁剪到画面外的 OBB 框会产生空 ROI，必须在调用 cv2.blur 前跳过。"""
    blurrer = solutions.ObjectBlurrer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    inside = torch.tensor([[100.0, 100], [200, 100], [200, 200], [100, 200]])  # (4, 2) OBB, in frame
    outside = torch.tensor([[700.0, 100], [760, 100], [760, 200], [700, 200]])  # (4, 2) OBB, fully off the right edge
    blurrer.boxes, blurrer.clss, blurrer.confs, blurrer.track_ids = [inside, outside], [0, 0], [0.9, 0.9], [1, 2]
    with patch.object(blurrer, "extract_tracks"), patch.object(blurrer, "display_output"):
        results = blurrer.process(frame)
    assert results.plot_im is not None


def test_right_click_reset():
    """测试距离计算的右键重置功能。"""
    dc = solutions.DistanceCalculation()
    dc.selected_boxes, dc.left_mouse_count = {1: [10, 10, 50, 50]}, 1
    dc.mouse_event_for_distance(cv2.EVENT_RBUTTONDOWN, 0, 0, None, None)
    assert not dc.selected_boxes, f"Expected empty selected_boxes after reset, got {dc.selected_boxes}"
    assert dc.left_mouse_count == 0, f"Expected left_mouse_count=0 after reset, got {dc.left_mouse_count}"


def test_parking_json_none():
    """测试 ParkingManagement 能够平稳处理缺失的 JSON。"""
    im0 = np.zeros((640, 480, 3), dtype=np.uint8)
    try:
        parkingmanager = solutions.ParkingManagement(json_path=None)
        parkingmanager(im0)
    except ValueError:
        pytest.skip("Skipping test due to missing JSON.")


def test_analytics_graph_not_supported():
    """测试不支持的分析类型会抛出 ValueError。"""
    try:
        analytics = solutions.Analytics(analytics_type="test")  # 'test' is unsupported
        analytics.process(im0=np.zeros((640, 480, 3), dtype=np.uint8), frame_number=0)
        assert False, "Expected ValueError for unsupported chart type"
    except ValueError as e:
        assert "Unsupported analytics_type" in str(e), f"Expected 'Unsupported analytics_type' in error, got: {e}"


def test_area_chart_padding():
    """测试面积图会使用动态类别填充逻辑进行更新。"""
    analytics = solutions.Analytics(analytics_type="area")
    analytics.update_graph(frame_number=1, count_dict={"car": 2}, plot="area")
    plot_im = analytics.update_graph(frame_number=2, count_dict={"car": 3, "person": 1}, plot="area")
    assert plot_im is not None, "Area chart plot returned None"


def test_config_update_method_with_invalid_argument():
    """测试 update() 会针对无效配置键抛出 ValueError。"""
    obj = solutions.config.SolutionConfig()
    try:
        obj.update(invalid_key=123)
        assert False, "Expected ValueError for invalid update argument"
    except ValueError as e:
        assert "is not a valid solution argument" in str(e), f"Expected validation error message, got: {e}"


def test_plot_with_no_masks():
    """测试实例分割能够处理没有掩码的情况。"""
    im0 = np.zeros((640, 480, 3), dtype=np.uint8)
    isegment = solutions.InstanceSegmentation(model="yolo26n-seg.pt")
    results = isegment(im0)
    assert results.plot_im is not None, "Instance segmentation plot returned None"


def test_streamlit_handle_video_upload_creates_file(tmp_path):
    """测试 Streamlit 视频上传逻辑能够正确保存文件。"""
    import io

    fake_file = io.BytesIO(b"fake video content")
    fake_file.read = fake_file.getvalue
    if fake_file is not None:
        g = io.BytesIO(fake_file.read())
        with open(tmp_path / "ultralytics.mp4", "wb") as out:
            out.write(g.read())
        output_path = str(tmp_path / "ultralytics.mp4")
    else:
        output_path = None
    assert output_path == str(tmp_path / "ultralytics.mp4"), (
        f"Expected output_path '{tmp_path / 'ultralytics.mp4'}', got {output_path}"
    )
    assert os.path.exists(tmp_path / "ultralytics.mp4"), "ultralytics.mp4 file not created"
    with open(tmp_path / "ultralytics.mp4", "rb") as f:
        content = f.read()
        assert content == b"fake video content", f"File content mismatch: {content}"


@pytest.mark.skipif(not TORCH_2_4, reason=f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})")
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled due to slow performance on Raspberry Pi.")
def test_similarity_search(tmp_path, solution_assets):
    """使用示例图像和文本查询测试相似度搜索解决方案。"""
    safe_download(solution_assets("similarity_images"), dir=tmp_path)  # 4 dog images for testing in a zip file
    searcher = solutions.VisualAISearch(data=str(tmp_path / "4-imgs-similaritysearch"))
    _ = searcher("a dog sitting on a bench")  # 以 "- 图像名称 | 相似度分数" 格式返回结果


@pytest.mark.skipif(not TORCH_2_4, reason=f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})")
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled due to slow performance on Raspberry Pi.")
def test_similarity_search_app_init():
    """测试 SearchApp 是否使用所需属性完成初始化。"""
    app = solutions.SearchApp(device="cpu")
    assert hasattr(app, "searcher")
    assert hasattr(app, "run")


@pytest.mark.skipif(not TORCH_2_4, reason=f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})")
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled due to slow performance on Raspberry Pi.")
def test_similarity_search_complete(tmp_path):
    """使用示例图像和查询测试 VisualAISearch 的端到端流程。"""
    from PIL import Image

    image_dir = tmp_path / "images"
    os.makedirs(image_dir, exist_ok=True)
    for i in range(2):
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        img.save(image_dir / f"test_image_{i}.jpg")
    searcher = solutions.VisualAISearch(data=str(image_dir))
    results = searcher("a red and white object")
    assert results, "Similarity search returned empty results"


def test_distance_calculation_process_method():
    """测试 DistanceCalculation.process() 能够计算所选框之间的距离。"""
    from ultralytics.solutions.solutions import SolutionResults

    dc = solutions.DistanceCalculation()
    dc.boxes, dc.track_ids, dc.clss, dc.confs = (
        [[100, 100, 200, 200], [300, 300, 400, 400]],
        [1, 2],
        [0, 0],
        [0.9, 0.95],
    )
    dc.selected_boxes = {1: dc.boxes[0], 2: dc.boxes[1]}
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch.object(dc, "extract_tracks"), patch.object(dc, "display_output"), patch("cv2.setMouseCallback"):
        result = dc.process(frame)
    assert isinstance(result, SolutionResults), f"Expected SolutionResults, got {type(result)}"
    assert result.total_tracks == 2, f"Expected 2 tracks, got {result.total_tracks}"
    assert result.pixels_distance > 0, f"Expected positive distance, got {result.pixels_distance}"


def test_object_crop_with_show_True():
    """使用 show=True 初始化 ObjectCropper，以覆盖显示警告逻辑。"""
    solutions.ObjectCropper(show=True)


def test_display_output_method():
    """测试启用 display_output 时会调用 imshow、waitKey 和 destroyAllWindows。"""
    counter = solutions.ObjectCounter(show=True)
    counter.env_check = True
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("cv2.imshow") as mock_imshow, patch("cv2.waitKey", return_value=ord("q")) as mock_wait, patch(
        "cv2.destroyAllWindows"
    ) as mock_destroy:
        counter.display_output(frame)
        mock_imshow.assert_called_once()
        mock_wait.assert_called_once()
        mock_destroy.assert_called_once()
