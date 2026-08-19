# Unitree Go2 Mujoco Simulation

This repository contains code used to run a Mujoco simulation of Unitree's Go2 robot in a Docker container.

Parts of this repository (incl. the Go2 Mujoco model and parts of the DDS translation code) were taken from [Unitree's Mujoco implementation](https://github.com/unitreerobotics/unitree_mujoco). See [unitree-license.txt](unitree-license.txt) for details on the license.

## Quickstart
To build the container, you need to have Docker installed on your system (see [install instructions](https://docs.docker.com/engine/install/)).
If you are running on a machine with NVIDIA graphics, you can use hardware acceleration to make the simulation significantly more performant. For that, you need a recent version of NVIDIAs official graphics driver as well as the NVIDIA Container Toolkit installed on your machine (see [install instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

Download and execute [start.sh](./start.sh). 
It will pull a pre-built image containing Mujoco with the Go2 plugin and run it.

## Building the Container yourself
Clone the repository to your machine and open it in a terminal.
There is a convenience script for building the container:
```sh
./build.sh
```

Once building is done, you can use the start script:
```sh
./start.sh
```
After a short load, Mujoco should open and show a simulated Go2 robot. 

## Developing
This repository contains a [development container specification](.devcontainer/devcontainer.json).
You can load the development container using Visual Studio's Dev Containers extension.

The Mujoco Go2 library can be built using ROS2's colcon:
```sh
colcon build
```
The build script will automatically install the compiled library in Mujoco's plugin directory.

You can start the simulator with the same start script (it detects if it is run inside the container):
```sh
./start.sh
```