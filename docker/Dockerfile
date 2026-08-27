# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# 在 Docker Hub 上构建 ultralytics/ultralytics:latest 镜像：https://hub.docker.com/r/ultralytics/ultralytics
# 此镜像针对 YOLO 单 GPU/多 GPU 训练与推理进行了 CUDA 优化

# 基于 PyTorch 镜像启动：https://hub.docker.com/r/pytorch/pytorch 或 nvcr.io/nvidia/pytorch:25.02-py3
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# 设置环境变量
# 避免 DDP 错误："MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library"
# 禁止 TensorFlow 的 cuDNN、cuBLAS、cuFFT 注册警告以及 PyTorch NNPACK 警告
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TORCH_CPP_LOG_LEVEL=ERROR

# 下载到用户配置目录
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# 安装 Linux 软件包
# 安装 Edge TPU 需要 gnupg
# libqxcb 创建基于 Qt 的可视化窗口需要 libsm6；可在 Docker 中设置 'QT_DEBUG_PLUGINS=1' 进行测试
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 gnupg libsm6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /ultralytics

# 复制内容并配置依赖项
COPY . .
RUN sed -i'' -e 's/"opencv-python/"opencv-python-headless/' pyproject.toml
ADD https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt .

# 安装 pip 软件包（基础镜像中已安装 uv）
RUN uv pip install --system --no-cache -e "." albumentations faster-coco-eval && \
    rm -rf tmp /root/.config/Ultralytics/persistent_cache.json

# 用法 --------------------------------------------------------------------------------------------------------------

# 正式构建：https://github.com/ultralytics/ultralytics/blob/main/.github/workflows/docker.yml
# 示例（构建）：t=ultralytics/ultralytics:latest && docker build -f docker/Dockerfile -t $t .
# 示例（推送）：docker push $t
# 示例（拉取）：t=ultralytics/ultralytics:latest && docker pull $t
# 示例（运行 GPU）：docker run -it --ipc=host --device nvidia.com/gpu=all $t
# 示例（运行指定 GPU）：docker run -it --ipc=host --device nvidia.com/gpu=2 --device nvidia.com/gpu=3 $t
# 注意：容器内的 CDI 设备 2、3 会显示为 cuda:0、1。
# 示例（挂载卷运行）：docker run -it --ipc=host --device nvidia.com/gpu=all -v "$PWD/shared/datasets:/datasets" $t
# CDI 需要 Docker >= 28.2.0 和 nvidia-container-toolkit >= 1.18。
# 旧版 --gpus all 在主机 systemd 守护进程重新加载后可能失去 GPU 访问权限（nvidia-container-toolkit#48）。
# 示例（标记发布）：t=ultralytics/ultralytics:latest tnew=ultralytics/ultralytics:vX.Y && docker tag $t $tnew && docker push $tnew
