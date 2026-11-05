# Toolchain image for `kiwi real2simviz` (docs/replay.md#gaussian-scenes):
# the CUDA 12.8 toolkit, a C++ compiler, git, curl and uv that
# third_party/setup_real2sim.sh requires. The renderer itself is NOT baked in:
# the installer refuses to finish without a visible GPU, which `docker build`
# cannot provide, so it runs once in a GPU container and installs into
# build/3dgrut/ in the bind-mounted repo (the documented location):
#     docker compose run --rm real2sim bash third_party/setup_real2sim.sh
# 3DGRUT compiles its CUDA/Slang kernels on first render, so nvcc is needed at
# runtime too, not only at install time.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
ARG UV_VERSION=0.8.22
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${CUDA_IMAGE}

COPY --from=uv /uv /uvx /usr/local/bin/

# build-essential/ninja: torch C++/CUDA extension builds; libgl1/libglib2.0-0:
# opencv, which kiwi imports; ffmpeg: kiwi reads frame times with ffprobe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl ffmpeg git libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

# The installer's `uv venv --python 3.11` links the venv (on the host, under
# build/3dgrut) to a uv-managed interpreter. Install that interpreter into the
# image at a fixed path so the venv still resolves in every new container.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.11 && chmod -R a+rX /opt/uv-python

# Run as the host user so build/3dgrut and renders in the bind mount are not
# root-owned (same scheme as docker/Dockerfile).
ARG UID=1000
ARG GID=1000
ARG WORKSPACE=/workspace
RUN (getent group ${GID} || groupadd -g ${GID} kiwi) \
    && (getent passwd ${UID} && userdel -r "$(getent passwd ${UID} | cut -d: -f1)" || true) \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash kiwi \
    && mkdir -p ${WORKSPACE} /home/kiwi/.cache \
    && chown -R ${UID}:${GID} ${WORKSPACE} /home/kiwi
USER kiwi
WORKDIR ${WORKSPACE}
# The uv cache sits on the kiwi-cache volume, a different filesystem from the
# venv in the bind mount: copy rather than attempt hardlinks.
ENV UV_CACHE_DIR=/home/kiwi/.cache/uv \
    UV_LINK_MODE=copy

CMD ["bash"]
