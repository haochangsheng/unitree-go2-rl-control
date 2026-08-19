#!/bin/env bash
WORKDIR=$(dirname $0)
IMAGE="unitree-rl-dev"

DOCKER_FLAGS=(
    "--rm"
    "--privileged"
    "--network=host"
    "--mount=type=bind,source=$WORKDIR,target=/workspace"
)

COMMAND=("python3" "/workspace/go2_deploy_vel.py")

set -e

# Check if we are already inside the container
if [ -n "$UNITREE_RL_CONTAINER" ]; then
    ${COMMAND[*]}
    exit $?
fi

# Check if NVIDIA graphics are present
if [ $(lspci | grep -ci nvidia) -gt 0 ]; then
    if ! command -v nvidia-container-toolkit &> /dev/null ; then
        echo "NVIDIA GPU detected but nvidia-container-toolkit not installed."
        echo "See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        echo "Not enabling NVIDIA graphics in the container."
    else
        echo "Enabling NVIDIA GPU in the container."
        DOCKER_FLAGS+=("--gpus=all")
    fi
else
    echo "No NVIDIA graphics found on this system."
fi

echo "Starting ROS2 Go2 node."
docker run ${DOCKER_FLAGS[*]} -it ${IMAGE} bash -lc "${COMMAND[*]}"