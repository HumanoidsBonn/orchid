FROM osrf/ros:humble-desktop

ARG USERNAME=orchid
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=/home/${USERNAME}/cyclonedds.xml
ENV CARGO_HOME=/opt/cargo
ENV RUSTUP_HOME=/opt/rustup
ENV PATH=/opt/cargo/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    iputils-ping \
    x11-apps \
    python3-colcon-common-extensions \
    python3-argcomplete \
    python3-opencv \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    ros-humble-py-trees-ros-viewer \
    ros-humble-py-trees-ros \
    ros-humble-rosbridge-suite \
    sudo \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu's Cargo 1.75 cannot parse Rust 2024 dependency manifests. Keep this
# toolchain in sync with rust-toolchain.toml in the mounted UI repository.
RUN curl --proto '=https' --tlsv1.2 -fsS https://sh.rustup.rs -o /tmp/orchid-rustup.sh \
    && sh /tmp/orchid-rustup.sh -y --profile minimal --default-toolchain 1.85.1 --no-modify-path \
    && rm /tmp/orchid-rustup.sh \
    && cargo --version && rustc --version

RUN useradd -m -s /bin/bash ${USERNAME}
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /home/${USERNAME}/orchid_ui/ros2_ws/install/setup.bash" >> ~/.bashrc
WORKDIR /home/${USERNAME}/orchid_ui
CMD ["bash"]

