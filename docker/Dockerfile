# Pixi runner image with the pixi environments installed at build time. The
# repo is bind-mounted over $WORKSPACE at runtime (see compose.yaml); the
# editable kiwi install points at $WORKSPACE/kiwi, so source edits are live
# without a rebuild. Rebuild after changing pixi.toml / pixi.lock.
ARG PIXI_VERSION=0.82.0
ARG DOCKER_VERSION=29.5.3
FROM docker:${DOCKER_VERSION}-cli AS docker-cli

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION}-noble

# The docker CLI only: kiwi calib launches Kalibr as a sibling container on
# the host daemon (socket mounted by compose.yaml).
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

# git: submodules + the slam-offline build; the rest is what pypi sdists and
# GUI-less OpenGL/X11 wheels expect to find on the system.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as the host user so files written into the bind mount (build/, data/)
# are not root-owned.
ARG UID=1000
ARG GID=1000
# The repo's path on the HOST. Sibling containers (Kalibr) get their bind
# mounts resolved by the host daemon, so the paths kiwi passes must be valid
# on the host; pixi's detached env name also hashes this path, so it must be
# the same at build time and at runtime.
ARG WORKSPACE=/workspace
RUN (getent group ${GID} || groupadd -g ${GID} kiwi) \
    && (getent passwd ${UID} && userdel -r "$(getent passwd ${UID} | cut -d: -f1)" || true) \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash kiwi \
    && mkdir -p ${WORKSPACE} /home/kiwi/.cache /home/kiwi/envs /home/kiwi/.pixi \
    && printf 'detached-environments = "/home/kiwi/envs"\n' > /home/kiwi/.pixi/config.toml \
    && chown -R ${UID}:${GID} ${WORKSPACE} /home/kiwi
USER kiwi
WORKDIR ${WORKSPACE}
# The base image puts /root/.pixi/bin on PATH; unreadable for a non-root user,
# it turns "command not found" into EACCES for every PATH lookup.
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Envs live outside $WORKSPACE (detached-environments above) so the bind
# mount does not shadow them. Only what the solve + the editable kiwi build
# need is copied, so source edits do not invalidate this layer.
COPY --chown=${UID}:${GID} pixi.toml pixi.lock pyproject.toml README.md LICENSE ./
COPY --chown=${UID}:${GID} kiwi ./kiwi
# Space-separated pixi environments to bake in (see [environments] in pixi.toml).
ARG PIXI_ENVIRONMENTS="default state viz tune replay"
RUN --mount=type=cache,target=/home/kiwi/.cache/rattler,uid=${UID},gid=${GID} \
    for env in ${PIXI_ENVIRONMENTS}; do pixi install --locked -e "$env" || exit 1; done

ENTRYPOINT ["pixi"]
CMD ["--help"]
