# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import requests
import yaml


def download_file(url: str, local_path: str) -> str:
    """Download a file from a URL to a local path.

    Args:
        url (str): URL of the file to download.
        local_path (str): Local path where the file will be saved.

    Returns:
        (str): Local path where the file was saved.
    """
    # 检查本地路径是否已存在
    if os.path.exists(local_path):
        print(f"File already exists at {local_path}. Skipping download.")
        return local_path
    # 从 URL 下载文件
    print(f"Downloading {url} to {local_path}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return local_path


class RTDETR:
    """RT-DETR (Real-Time Detection Transformer) object detection model for ONNX inference and visualization.

    This class implements the RT-DETR model for object detection tasks, supporting ONNX model inference and
    visualization of detection results with bounding boxes and class labels.

    Attributes:
        model_path (str): Path to the ONNX model file.
        img_path (str): Path to the input image.
        conf_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.
        session (ort.InferenceSession): ONNX runtime inference session.
        model_input (list): Model input metadata.
        input_width (int): Width dimension required by the model.
        input_height (int): Height dimension required by the model.
        classes (list[str]): List of class names from COCO dataset.
        color_palette (np.ndarray): Random color palette for visualization.
        img (np.ndarray): Loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.

    Methods:
        draw_detections: Draw bounding boxes and labels on the input image.
        preprocess: Preprocess the input image for model inference.
        bbox_cxcywh_to_xyxy: Convert bounding boxes from center format to corner format.
        postprocess: Postprocess model output to extract and visualize detections.
        main: Execute the complete object detection pipeline.

    Examples:
        Initialize RT-DETR detector and run inference
        >>> detector = RTDETR("rtdetr-l.onnx", "image.jpg", conf_thres=0.5)
        >>> output_image = detector.main()
        >>> cv2.imshow("Detections", output_image)
    """

    def __init__(
        self,
        model_path: str,
        img_path: str,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        class_names: str | None = None,
    ):
        """Initialize the RT-DETR object detection model.

        Args:
            model_path (str): Path to the ONNX model file.
            img_path (str): Path to the input image.
            conf_thres (float, optional): Confidence threshold for filtering detections.
            iou_thres (float, optional): IoU threshold for non-maximum suppression.
            class_names (Optional[str], optional): Path to a YAML file containing class names. If None, uses COCO
                dataset classes.
        """
        self.model_path = model_path
        self.img_path = img_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = class_names

    # 使用可用执行提供程序设置 ONNX Runtime 会话
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(model_path, providers=providers or available)

        self.model_input = self.session.get_inputs()
        self.input_width = self.model_input[0].shape[2]
        self.input_height = self.model_input[0].shape[3]

        if self.classes is None:
        # 从 COCO 数据集 YAML 文件加载类别名称
            self.classes = download_file(
                "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco8.yaml",
                "coco8.yaml",
            )

        # 解析 YAML 文件以获取类别名称
        with open(self.classes) as f:
            class_data = yaml.safe_load(f)
            self.classes = list(class_data["names"].values())

        # 确保类别是列表
        if not isinstance(self.classes, list):
            raise TypeError("Classes should be a list of class names.")

        # 生成用于绘制边界框的颜色调色板
        self.color_palette: np.ndarray = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, box: np.ndarray, score: float, class_id: int) -> None:
        """Draw bounding box and label on the input image for a detected object."""
        # 提取边界框坐标
        x1, y1, x2, y2 = box

        # 获取类别 ID 对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 使用类别名称和分数创建标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充矩形作为标签文本背景
        cv2.rectangle(
            self.img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # 在图像上绘制标签文本
        cv2.putText(
            self.img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    def preprocess(self) -> np.ndarray:
        """Preprocess the input image for model inference.

        Loads the image, converts color space from BGR to RGB, resizes to model input dimensions, and normalizes pixel
        values to [0, 1] range.

        Returns:
            (np.ndarray): Preprocessed image data with shape (1, 3, H, W) ready for inference.
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            raise FileNotFoundError(f"Image not found or unreadable: '{self.img_path}'")

        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 调整图像尺寸以匹配输入形状
        img = cv2.resize(img, (self.input_width, self.input_height))

        # 将图像数据除以 255.0 进行归一化
        image_data = np.array(img) / 255.0

        # 转置图像，使通道维度位于第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # 扩展图像数据维度，使其匹配预期输入形状
        image_data = image_data[None].astype(np.float32)

        return image_data

    def bbox_cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert bounding boxes from center format to corner format.

        Args:
            boxes (np.ndarray): Array of shape (N, 4) where each row represents a bounding box in (center_x, center_y,
                width, height) format.

        Returns:
            (np.ndarray): Array of shape (N, 4) with bounding boxes in (x_min, y_min, x_max, y_max) format.
        """
        # 计算边界框的半宽和半高
        half_width = boxes[:, 2] / 2
        half_height = boxes[:, 3] / 2

        # 计算边界框坐标
        x_min = boxes[:, 0] - half_width
        y_min = boxes[:, 1] - half_height
        x_max = boxes[:, 0] + half_width
        y_max = boxes[:, 1] + half_height

        # 以 (x_min, y_min, x_max, y_max) 格式返回边界框
        return np.column_stack((x_min, y_min, x_max, y_max))

    def postprocess(self, model_output: list[np.ndarray]) -> np.ndarray:
        """Postprocess model output to extract and visualize detections.

        Applies confidence thresholding, converts bounding box format, scales coordinates to original image dimensions,
        and draws detection annotations.

        Args:
            model_output (list[np.ndarray]): Output tensors from the model inference.

        Returns:
            (np.ndarray): Annotated image with detection bounding boxes and labels.
        """
        # 压缩模型输出以移除不必要的维度
        outputs = np.squeeze(model_output[0])

        # 从模型输出中提取边界框和分数
        boxes = outputs[:, :4]
        scores = outputs[:, 4:]

        # 获取每个检测结果的类别标签和分数
        labels = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        # 应用置信度阈值过滤低置信度检测结果
        mask = scores > self.conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        # 将边界框转换为 (x_min, y_min, x_max, y_max) 格式
        boxes = self.bbox_cxcywh_to_xyxy(boxes)

        # 缩放边界框以匹配原始图像尺寸
        boxes[:, 0::2] *= self.img_width
        boxes[:, 1::2] *= self.img_height

        # 应用非极大值抑制（RT-DETR 可选，但有助于过滤重叠框）
        xywh_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(xywh_boxes, scores.tolist(), self.conf_thres, self.iou_thres)
        indices = indices.flatten().tolist() if len(indices) else []

        # 在图像上绘制检测结果
        for i in indices:
            self.draw_detections(boxes[i], float(scores[i]), int(labels[i]))

        return self.img

    def main(self) -> np.ndarray:
        """Execute the complete object detection pipeline on the input image.

        Performs preprocessing, ONNX model inference, and postprocessing to generate annotated detection results.

        Returns:
            (np.ndarray): Output image with detection annotations including bounding boxes and class labels.
        """
        # 预处理图像作为模型输入
        image_data = self.preprocess()

        # 运行模型推理
        model_output = self.session.run(None, {self.model_input[0].name: image_data})

        # 处理并返回模型输出
        return self.postprocess(model_output)


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rtdetr-l.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--img", type=str, default="bus.jpg", help="Path to the input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold for object detection.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for non-maximum suppression.")
    args = parser.parse_args()

    # 使用指定参数创建检测器实例
    detection = RTDETR(args.model, args.img, args.conf_thres, args.iou_thres)

    # 执行检测并获取输出图像
    output_image = detection.main()

    # 显示标注后的输出图像
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
