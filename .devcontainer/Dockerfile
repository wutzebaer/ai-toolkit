FROM nvcr.io/nvidia/pytorch:25.10-py3



# Install dependencies
RUN --mount=target=/var/lib/apt/lists,type=cache,id=apt-lists \
    --mount=target=/var/cache/apt,type=cache,id=apt-cache \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt update && \
    apt install -y software-properties-common lsb-release ffmpeg npm
RUN apt update

# Setup virtual environment
RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install tools
RUN --mount=target=/root/.cache/pip,type=cache,id=pip-cache \
    pip install sageattention flash-attn

# docker buildx build -f Dockerfile ..