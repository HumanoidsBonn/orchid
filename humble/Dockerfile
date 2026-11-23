ARG BASE_IMAGE=registry.gitlab.uni-bonn.de:5050/s52sgupt/ovmm_images/orchid_base_image:latest
FROM ${BASE_IMAGE}

ARG USERNAME=orchid
SHELL ["/bin/bash", "-c"]
USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=/home/${USERNAME}/cyclonedds.xml

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && cp /root/.local/bin/uv /root/.local/bin/uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    git-lfs \
    ninja-build \
    vim \
    iputils-ping \
    x11-apps \
    python3-colcon-common-extensions \
    python3-argcomplete \
    libfcl-dev \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-rosidl-generator-dds-idl \
    ros-humble-py-trees-ros-viewer \
    ros-humble-py-trees-ros \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-nav2-bt-navigator \
    ros-humble-nav2-controller \
    ros-humble-nav2-dwb-controller \
    ros-humble-nav2-navfn-planner \
    ros-humble-nav2-planner \
    ros-humble-nav2-behaviors \
    ros-humble-nav2-waypoint-follower \
    ros-humble-nav2-simple-commander \
    ros-humble-xacro \
    ros-humble-moveit-msgs \
    ros-humble-pinocchio \
    ros-humble-geometric-shapes \
    ros-humble-pcl-ros \
    ros-humble-ros-gz \
    ros-humble-slam-toolbox \
    ros-humble-nav2-map-server \
    ros-humble-nav2-amcl \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    ros-humble-control-msgs \
    ros-humble-controller-interface \
    ros-humble-joint-trajectory-controller \
    ros-humble-gz-ros2-control \
    ros-humble-moveit-common \
    ros-humble-moveit-core \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-ros-planning \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-joint-state-publisher \
    ros-humble-rqt-joint-trajectory-controller \
    ros-humble-octomap \
    ros-humble-ruckig \
    ros-humble-srdfdom \
    ros-humble-urdfdom-py \
    ros-humble-sensor-msgs-py \
    ros-humble-turtlebot3-teleop \
    ros-humble-twist-mux \
    libeigen3-dev \
    libopencv-dev \
    libpcl-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends wget unzip \
    && wget -O zenoh-plugin-ros2dds.zip https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds/releases/download/1.5.0/zenoh-plugin-ros2dds-1.5.0-x86_64-unknown-linux-gnu-standalone.zip \
    && unzip zenoh-plugin-ros2dds.zip -d /usr/local/bin/ \
    && rm zenoh-plugin-ros2dds.zip \
    && apt-get purge -y wget unzip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# RUN touch /home/orchid/hsr/src/tmc_drivers/tmc_pgr_camera/COLCON_IGNORE

RUN id ${USERNAME} \
    && test -d /home/${USERNAME} \
    && test -d /home/${USERNAME}/ros2_ws/src \
    || (echo "Base image must provide user ${USERNAME} and /home/${USERNAME}/ros2_ws/src" && exit 1)

WORKDIR /home/${USERNAME}/orchid_server

COPY --chown=${USERNAME}:${USERNAME} submodules/gpd /home/${USERNAME}/gpd
RUN cd /home/${USERNAME}/gpd \
    && rm -rf build \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j"$(nproc)" \
    && make install

COPY --chown=${USERNAME}:${USERNAME} submodules/gpd_ros2/gpd_ros2 /home/${USERNAME}/ros2_ws/src/gpd_ros2
COPY --chown=${USERNAME}:${USERNAME} submodules/gpd_ros2/gpd_ros2_msgs /home/${USERNAME}/ros2_ws/src/gpd_ros2_msgs
COPY --chown=${USERNAME}:${USERNAME} submodules/orchid_ui/ros2_ws/src/ /home/${USERNAME}/ros2_ws/src/
COPY --chown=${USERNAME}:${USERNAME} submodules/mapping /home/${USERNAME}/ros2_ws/src/orchid_mapping_config
COPY --chown=${USERNAME}:${USERNAME} submodules/nav2_config /home/${USERNAME}/ros2_ws/src/orchid_nav2_config
COPY --chown=${USERNAME}:${USERNAME} submodules/orchid_moveit_config /home/${USERNAME}/ros2_ws/src/orchid_moveit_config
COPY --chown=${USERNAME}:${USERNAME} submodules/orchid_utils /home/${USERNAME}/ros2_ws/src/orchid_utils
COPY --chown=${USERNAME}:${USERNAME} submodules/orchid_hsr_pickup/ /home/${USERNAME}/ros2_ws/src/
RUN source /opt/ros/humble/setup.bash \
    && if [[ -f /home/${USERNAME}/hsr/install/setup.bash ]]; then source /home/${USERNAME}/hsr/install/setup.bash; fi \
    && cd /home/${USERNAME}/ros2_ws \
    && colcon build --symlink-install \
        --packages-select orchid_hsr_pickup_interfaces orchid_hsr_pickup orchid_ros_interfaces gpd_ros2_msgs gpd_ros2 orchid_mapping_config orchid_nav2_config orchid_moveit_config orchid_utils \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

# Install Orchid Python dependencies after the ROS workspace build so uv cannot
# replace ROS/colcon's setuptools while colcon is still invoking setup.py.
COPY pyproject.toml uv.lock ./
RUN echo "setuptools" > /tmp/orchid-uv-exclude.txt \
    && uv pip install --system -r pyproject.toml --excludes /tmp/orchid-uv-exclude.txt \
    && rm /tmp/orchid-uv-exclude.txt

CMD ["bash"]
