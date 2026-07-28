FROM python:3.13-slim-trixie

WORKDIR /srv/cannabinoids

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    MAX_DATA_LEN=100 \
    PYTHONPATH=/srv/cannabinoids

COPY pyproject.toml ./

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends binutils; \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir .; \
    rm -f pyproject.toml; \
    find /usr/local/lib/python3.13/site-packages -type d \( -name "__pycache__" -o -name "tests" -o -name "test" \) -exec rm -rf {} + 2>/dev/null || true; \
    find /usr/local/lib/python3.13/site-packages -name "*.py[co]" -delete 2>/dev/null || true; \
    find /usr/local/lib/python3.13/site-packages -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    find /usr/local/lib/python3.13/site-packages -name "*.a" -delete 2>/dev/null || true; \
    groupadd --gid "${APP_GID}" appgroup; \
    useradd --uid "${APP_UID}" --gid "${APP_GID}" --no-create-home --home-dir /tmp --shell /usr/sbin/nologin appuser; \
    apt-get purge -y --auto-remove binutils; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /root/.cache; \
    pip cache purge 2>/dev/null || true


# Minified image: copy only the core app/ and common/ folders, do not copy tests/
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup common/ ./common/

# 预先创建 outputs 与 scratch 目录并赋予非 root 用户权限
RUN mkdir -p outputs scratch && chown -R appuser:appgroup outputs scratch

# 清除应用程序本身的字节码缓存
RUN find ./app ./common -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

EXPOSE 8000 8001

USER appuser

CMD ["python", "app/server.py"]
