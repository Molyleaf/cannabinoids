# 使用官方轻量 Python 3.11 镜像
FROM python:3.13-slim-trixie

# 从 official astral-sh/uv 镜像复制 uv 二进制工具
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录为 /app
WORKDIR /app

# 复制 app 目录所有文件到容器的 /app 目录
COPY . /app

# 使用 uv 进行快速依赖安装 (CPU 版本 PyTorch 优化镜像体积)
RUN uv pip install --system . --extra-index-url https://download.pytorch.org/whl/cpu

# 暴露 FastAPI (8000) 和 Streamlit (8001) 端口
EXPOSE 8000 8001

# 设置 Python 模块路径
ENV PYTHONPATH=/app

# 默认运行启动脚本以决定运行哪个服务
CMD ["python", "start.py"]
