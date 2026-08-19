FROM ros:humble
SHELL ["bash", "-l", "-c"]

RUN apt-get update && \
    apt-get upgrade -y --autoremove && \
    apt-get install -y build-essential cmake git curl iproute2 jq openssh-client \
        python3-pip \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-rosidl-generator-dds-idl \
        ros-${ROS_DISTRO}-ament-cmake-clang-format \
        libyaml-cpp-dev libspdlog-dev libeigen3-dev libfmt-dev

RUN rosdep update

# Install Unitree Go2 ROS2 bindings
RUN mkdir -p /tmp/unitree_ros2
WORKDIR /tmp/unitree_ros2
RUN git clone https://github.com/unitreerobotics/unitree_ros2.git . -b v0.3.0
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --packages-select unitree_go unitree_api --install-base /opt/unitree_ros2

# Python dependencies
RUN pip3 install --no-cache-dir numpy torch

# Setup environment
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /opt/unitree_ros2/setup.bash" >> ~/.bashrc
RUN echo ". /opt/ros/${ROS_DISTRO}/setup.sh" >> ~/.profile && \
    echo ". /opt/unitree_ros2/setup.sh" >> ~/.profile

WORKDIR /
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" priority="default" multicast="default" /></Interfaces><AllowMulticast>spdp</AllowMulticast></General></Domain></CycloneDDS>'
ENV UNITREE_RL_CONTAINER=true
