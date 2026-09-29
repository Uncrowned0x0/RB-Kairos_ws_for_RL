# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================

FROM osrf/ros:jazzy-desktop

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

ARG USERNAME=robot
ARG USER_UID=1000
ARG USER_GID=1000

# ── Copy dependency lists ─────────────────────────────────────────────────────
COPY docker/requirements/builder/packages.txt /tmp/requirements/builder-packages.txt
COPY docker/requirements/base/packages.txt    /tmp/requirements/base-packages.txt
COPY docker/requirements/python/requirements.txt /tmp/requirements/python-requirements.txt

# ── System packages & build tools ─────────────────────────────────────────────
RUN builder_packages="$(grep -vE '^(#|$)' /tmp/requirements/builder-packages.txt)" \
    && apt-get update \
    && apt-get install -y --no-install-recommends ${builder_packages} \
    && rm -rf /var/lib/apt/lists/*

# ── Official OSRF repository for Gazebo Harmonic ──────────────────────────────
RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# ── ROS 2 + Gazebo Harmonic + Controller packages ─────────────────────────────
RUN base_packages="$(sed "s/\${ROS_DISTRO}/${ROS_DISTRO}/g" /tmp/requirements/base-packages.txt | grep -vE '^(#|$)')" \
    && apt-get update \
    && apt-get install -y --no-install-recommends ${base_packages} \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies for Reinforcement Learning (PyTorch, SB3, Gymnasium) ──
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed \
    -r /tmp/requirements/python-requirements.txt

# ── Initialize rosdep ─────────────────────────────────────────────────────────
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi

# ── Create non-root user (robot) ──────────────────────────────────────────────
RUN if ! getent group "${USER_GID}" >/dev/null; then \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    fi \
    && if id -u "${USERNAME}" >/dev/null 2>&1; then \
        usermod --uid "${USER_UID}" --gid "${USER_GID}" \
                --home "/home/${USERNAME}" --shell /bin/bash "${USERNAME}"; \
    elif getent passwd "${USER_UID}" >/dev/null; then \
        existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
        usermod --login "${USERNAME}" --gid "${USER_GID}" \
                --home "/home/${USERNAME}" --move-home --shell /bin/bash "${existing_user}"; \
    else \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" \
                --create-home --shell /bin/bash "${USERNAME}"; \
    fi \
    && usermod -aG sudo,video,dialout "${USERNAME}" \
    && if getent group render >/dev/null; then usermod -aG render "${USERNAME}"; fi \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
    && mkdir -p "/home/${USERNAME}/ros2_ws/src" \
                "/home/${USERNAME}/ros2_ws/checkpoints" \
                "/home/${USERNAME}/ros2_ws/rl_level0" \
                "/home/${USERNAME}/ros2_ws/tb_logs" \
    && chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}"

# ── Copy entrypoint script ────────────────────────────────────────────────────
COPY docker/kairos-rl-entrypoint.sh /usr/local/bin/kairos-rl-entrypoint.sh
RUN chmod +x /usr/local/bin/kairos-rl-entrypoint.sh

# ── Automatic environment sourcing in bash profiles ───────────────────────────
RUN printf '%s\n' \
    'if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi' \
    'if [ -f /home/robot/ros2_ws/install/setup.bash ]; then source /home/robot/ros2_ws/install/setup.bash; fi' \
    > /etc/profile.d/kairos_rl_setup.sh \
    && chmod 0644 /etc/profile.d/kairos_rl_setup.sh

USER ${USERNAME}
WORKDIR /home/${USERNAME}/ros2_ws

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> "/home/${USERNAME}/.bashrc" \
    && echo 'if [ -f ~/ros2_ws/install/setup.bash ]; then source ~/ros2_ws/install/setup.bash; fi' >> "/home/${USERNAME}/.bashrc" \
    && rosdep update

# Switch back to root so entrypoint can configure host volume permissions
# and clean IPC before stepping down to robot via gosu
USER root

ENTRYPOINT ["/usr/local/bin/kairos-rl-entrypoint.sh"]
CMD ["sleep", "infinity"]
