# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import ASSETS, ROOT, YAML
from ultralytics.utils.checks import check_requirements


class YOLOv8:
    """YOLOv8 object detection model class for handling ONNX inference and visualization.

    This class provides functionality to load a YOLOv8 ONNX model, perform inference on images, and visualize the
    detection results with bounding boxes and labels.

    Attributes:
        onnx_model (str): Path to the ONNX model file.
        input_image (str): Path to the input image file.
        confidence_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.
        classes (list[str]): List of class names from the COCO dataset.
        color_palette (np.ndarray): Random color palette for visualizing different classes.
        input_width (int): Width dimension of the model input.
        input_height (int): Height dimension of the model input.
        img (np.ndarray): The loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.

    Methods:
        letterbox: Resize and reshape images while maintaining aspect ratio by adding padding.
        draw_detections: Draw bounding boxes and labels on the input image based on detected objects.
        preprocess: Preprocess the input image before performing inference.
        postprocess: Perform post-processing on the model's output to extract and visualize detections.
        main: Perform inference using an ONNX model and return the output image with drawn detections.

    Examples:
        Initialize YOLOv8 detector and run inference
        >>> detector = YOLOv8("yolov8n.onnx", "image.jpg", 0.5, 0.5)
        >>> output_image = detector.main()
    """

    def __init__(self, onnx_model: str, input_image: str, confidence_thres: float, iou_thres: float):
        """Initialize an instance of the YOLOv8 class.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 从 COCO 数据集加载类别名称
        self.classes = YAML.load(ROOT / "cfg/datasets/coco8.yaml")["names"]

        # 为各类别生成颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def letterbox(self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            pad (tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # 缩放比例（新尺寸 / 原尺寸）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # 计算填充量
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: list[float], score: float, class_id: int) -> None:
        """Draw bounding boxes and labels on the input image based on the detected objects."""
        # 提取边界框坐标
        x1, y1, w, h = box

        # 获取类别 ID 对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 使用类别名称和分数创建标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充矩形作为标签文本背景
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            image_data (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            pad (tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(self.input_image)

        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, (self.input_height, self.input_width))

        # 将图像数据除以 255.0 进行归一化
        image_data = np.array(img) / 255.0

        # 转置图像，使通道维度位于第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # 扩展图像数据维度，使其匹配预期输入形状
        image_data = image_data[None].astype(np.float32)

        # 返回预处理后的图像数据
        return image_data, pad

    def postprocess(self, input_image: np.ndarray, output: list[np.ndarray], pad: tuple[int, int]) -> np.ndarray:
        """Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs. It applies
        non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (list[np.ndarray]): The output arrays from the model.
            pad (tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # 转置并压缩输出，使其匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))

        # 获取输出数组中的行数
        rows = outputs.shape[0]

        # 用于保存检测结果边界框、分数和类别 ID 的列表
        boxes = []
        scores = []
        class_ids = []

        # 计算边界框坐标的缩放因子
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # 遍历输出数组中的每一行
        for i in range(rows):
            # 从当前行提取类别分数
            classes_scores = outputs[i][4:]

            # 找到类别分数中的最大值
            max_score = np.amax(classes_scores)

            # 如果最大分数高于置信度阈值
            if max_score >= self.confidence_thres:
                # 获取分数最高的类别 ID
                class_id = np.argmax(classes_scores)

                # 从当前行提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算缩放后的边界框坐标
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # 将类别 ID、分数和框坐标添加到对应列表
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 应用非极大值抑制，过滤重叠边界框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # 遍历非极大值抑制后选中的索引
        for i in np.array(indices).flatten():
        # 获取索引对应的框、分数和类别 ID
            box = boxes[int(i)]
            score = scores[int(i)]
            class_id = class_ids[int(i)]

        # 在输入图像上绘制检测结果
            self.draw_detections(input_image, box, score, class_id)

        # 返回修改后的输入图像
        return input_image

    def main(self) -> np.ndarray:
        """Perform inference using an ONNX model and return the output image with drawn detections.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        session = ort.InferenceSession(self.onnx_model, providers=providers or available)

        # 获取模型输入
        model_inputs = session.get_inputs()

        # 保存输入形状供后续使用；动态（非整数）轴回退到 640
        _, _, height, width = model_inputs[0].shape
        self.input_height = height if isinstance(height, int) else 640
        self.input_width = width if isinstance(width, int) else 640

        # 预处理图像数据
        img_data, pad = self.preprocess()

        # 使用预处理后的图像数据运行推理
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # 对输出执行后处理以获得输出图像
        return self.postprocess(self.img, outputs, pad)


if __name__ == "__main__":
    # 创建参数解析器以处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    # 检查依赖并选择合适的后端（CPU 或 GPU）
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # 使用指定参数创建 YOLOv8 类实例
    detection = YOLOv8(args.model, args.img, args.conf_thres, args.iou_thres)

    # 执行目标检测并获取输出图像
    output_image = detection.main()

    # 在窗口中显示输出图像
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)

    # 等待按键退出
    cv2.waitKey(0)
