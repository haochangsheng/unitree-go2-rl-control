#!/bin/env bash
WORKDIR=$(dirname $0)
IMAGE_NAME_DEV="unitree-rl-dev"

# Check if we are within the container
if [ -n "$UNITREE_RL_CONTAINER" ]; then
    colcon build
    exit $?
fi

docker build --pull -t "$IMAGE_NAME_DEV" "$WORKDIR"
