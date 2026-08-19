"""
ROS2 deployment node for the Go2 RL velocity controller.

NN outputs joint velocities → PI controller → torque.

States: IDLE → STAND_UP → STANDING → RL
Keyboard: s/r/e/q

Usage:
    python go2_deploy_vel.py --policy_path policy_vel.pt
"""

import argparse
import sys
import tty
import termios
import threading
import enum
import time
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState, LowCmd, MotorCmd

NUM_MOTORS = 12

# MuJoCo: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
# RL:     FL(0-2), FR(3-5), RL(6-8), RR(9-11)
RL_TO_MJ = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
MJ_TO_RL = RL_TO_MJ

EFFORT_LIMIT = np.array([
    23.7, 23.7, 45.43,   # FL
    23.7, 23.7, 45.43,   # FR
    23.7, 23.7, 45.43,   # RL
    23.7, 23.7, 45.43,   # RR
], dtype=np.float32)

DEFAULT_JOINT_ANGLES = np.array([
     0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,   # FR
     0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,   # RR
], dtype=np.float32)

OBS_LEN = 48
BASE_LIN_VEL = slice(0, 3)
BASE_ANG_VEL = slice(3, 6)
JOINT_POS = slice(6, 18)
JOINT_VEL = slice(18, 30)
PROJECTED_GRAVITY = slice(30, 33)
COMMANDS = slice(33, 36)
ACTIONS = slice(36, 48)

OBS_SCALE = np.ones(OBS_LEN, dtype=np.float32)
OBS_SCALE[BASE_LIN_VEL] = 2.0
OBS_SCALE[BASE_ANG_VEL] = 0.25
OBS_SCALE[JOINT_POS] = 1.0
OBS_SCALE[JOINT_VEL] = 0.05
OBS_SCALE[PROJECTED_GRAVITY] = 1.0
OBS_SCALE[COMMANDS][:2] = 2.0
OBS_SCALE[COMMANDS.start + 2] = 0.25
OBS_SCALE[ACTIONS] = 1.0


class State(enum.Enum):
    IDLE = 0
    STAND_UP = 1
    STANDING = 2
    RL = 3


def projected_gravity(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(w * w - x * x - y * y + z * z),
    ], dtype=np.float32)


def quat_apply_forward(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.array([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dtype=np.float32)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f'Network structure:\n{ckpt["network_structure"]}')

    state_dict = ckpt['network_state_dict']
    net = rebuild_mlp(state_dict, device)
    net.eval()

    obs_mask = ckpt.get('actor_obs_mask')
    if obs_mask is not None:
        obs_mask = obs_mask.to(device)
        print(f'Actor obs mask: {obs_mask.sum().int().item()} / {obs_mask.shape[0]} dims')

    return net, obs_mask


def rebuild_mlp(state_dict, device):
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix('model.')] = v
    
    layers = []
    weight_keys = sorted([k for k in cleaned if 'weight' in k])
    for i, wk in enumerate(weight_keys):
        w = cleaned[wk]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(weight_keys) - 1:
            layers.append(nn.ReLU())
    
    net = nn.Sequential(*layers).to(device)
    net.load_state_dict(cleaned)
    return net


class Go2DeployNode(Node):
    def __init__(self, policy_net, obs_mask, device, control_dt, standup_duration,
                 n_intermediate_steps=4, standup_kp=40.0, standup_kd=1.0):
        super().__init__('go2_deploy')
        self.policy_net = policy_net
        self.obs_mask = obs_mask
        self.device = device
        self.control_dt = control_dt
        self.standup_duration = standup_duration
        self.n_intermediate_steps = n_intermediate_steps

        # Stand-up uses simple PD (not the PI controller)
        self.standup_kp = standup_kp
        self.standup_kd = standup_kd

        # PI controller state
        self.integral_error = np.zeros(NUM_MOTORS, dtype=np.float32)

        self.state = State.IDLE
        self.latest_msg = None
        self.prev_raw_action = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.cmd_vel = np.zeros(3, dtype=np.float32)
        self.cmd_vel[0] = 0.7

        # Circular walking: yaw_rate = v / R
        self.desired_yaw_rate = self.cmd_vel[0] / 1.0  # radius = 2m

        # Yaw heading correction
        self.target_heading = 0.0

        # RL step counter
        self.step_counter = 0
        self.current_raw_action = np.zeros(NUM_MOTORS, dtype=np.float32)

        # Stand-up interpolation
        self.standup_start_pos = None
        self.standup_start_time = None

        sim_dt = control_dt / n_intermediate_steps
        self.state_sub = self.create_subscription(
            LowState, '/lowstate', self._state_callback, 10)
        self.cmd_pub = self.create_publisher(LowCmd, '/lowcmd', 10)
        self.timer = self.create_timer(sim_dt, self._control_loop)

        self.get_logger().info(
            f'Vel controller: policy dt={control_dt}s, sim dt={sim_dt}s, ')
        self._print_status()

    def _print_status(self):
        labels = {
            State.IDLE: 'IDLE       — press [s] to stand up',
            State.STAND_UP: 'STAND_UP   — interpolating to nominal...',
            State.STANDING: 'STANDING   — press [r] to start RL',
            State.RL: 'RL         — policy running (vel PI)',
        }
        self.get_logger().info(f'State: {labels[self.state]}')

    def transition(self, key):
        if key == 's' and self.state == State.IDLE:
            if self.latest_msg is None:
                self.get_logger().warn('No state received yet')
                return
            self.standup_start_pos = self._get_joint_pos_rl(self.latest_msg)
            self.standup_start_time = time.monotonic()
            self.state = State.STAND_UP

        elif key == 'r' and self.state == State.STANDING:
            self.prev_raw_action = np.zeros(NUM_MOTORS, dtype=np.float32)
            self.current_raw_action = np.zeros(NUM_MOTORS, dtype=np.float32)
            self.integral_error = np.zeros(NUM_MOTORS, dtype=np.float32)
            self.step_counter = 0
            quat = np.array(self.latest_msg.imu_state.quaternion, dtype=np.float32)
            fwd = quat_apply_forward(quat)
            self.target_heading = np.arctan2(fwd[1], fwd[0])
            self.state = State.RL

        elif key == 'e':
            self.state = State.IDLE
            self.integral_error = np.zeros(NUM_MOTORS, dtype=np.float32)
            self.get_logger().warn('EMERGENCY STOP')

        elif key == 'q':
            self.get_logger().info('Quit requested')
            raise SystemExit

        else:
            return
        self._print_status()

    def _state_callback(self, msg):
        self.latest_msg = msg

    def _get_joint_pos_rl(self, msg):
        return np.array([msg.motor_state[MJ_TO_RL[i]].q for i in range(NUM_MOTORS)], dtype=np.float32)

    def _get_joint_state_rl(self, msg):
        pos = np.array([msg.motor_state[MJ_TO_RL[i]].q for i in range(NUM_MOTORS)], dtype=np.float32)
        vel = np.array([msg.motor_state[MJ_TO_RL[i]].dq for i in range(NUM_MOTORS)], dtype=np.float32)
        return pos, vel

    def _compute_torque_pi(self, action, joint_vel):
        """PI velocity controller with nominal torque and anti-windup."""
        current_error = action * 3. - joint_vel

        torques = 2. * current_error + 0.2 * self.integral_error

        # Anti-windup: only integrate when not saturated
        not_sat = np.abs(torques) < EFFORT_LIMIT
        torques = np.clip(torques, -EFFORT_LIMIT, EFFORT_LIMIT)

        # Integral error update with sign-flip reset
        sign_flipped = (self.integral_error * current_error) < 0
        self.integral_error[sign_flipped] = current_error[sign_flipped]
        self.integral_error[not_sat & ~sign_flipped] += current_error[not_sat & ~sign_flipped]

        return torques

    def _compute_torque_pd(self, target_pos, joint_pos, joint_vel):
        """Simple PD for stand-up phase."""
        torques = self.standup_kp * (target_pos - joint_pos) - self.standup_kd * joint_vel
        return np.clip(torques, -EFFORT_LIMIT, EFFORT_LIMIT)

    def _make_torque_cmd(self, torques_rl):
        cmd = LowCmd()
        for i in range(NUM_MOTORS):
            motor = MotorCmd()
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = float(torques_rl[i])
            cmd.motor_cmd[RL_TO_MJ[i]] = motor
        return cmd

    def _build_observation(self, msg):
        obs = np.zeros(OBS_LEN, dtype=np.float32)
        obs[BASE_LIN_VEL] = 0.0
        obs[BASE_ANG_VEL] = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        for i in range(NUM_MOTORS):
            obs[JOINT_POS.start + i] = msg.motor_state[MJ_TO_RL[i]].q
            obs[JOINT_VEL.start + i] = msg.motor_state[MJ_TO_RL[i]].dq
        quat = np.array(msg.imu_state.quaternion, dtype=np.float32)
        obs[PROJECTED_GRAVITY] = projected_gravity(quat)

        # Yaw heading correction
        fwd = quat_apply_forward(quat)
        heading = np.arctan2(fwd[1], fwd[0])
        self.cmd_vel[2] = np.clip(0.5 * wrap_to_pi(self.target_heading - heading), -1.0, 1.0)

        obs[COMMANDS] = self.cmd_vel
        obs[ACTIONS] = self.prev_raw_action

        obs[JOINT_POS] -= DEFAULT_JOINT_ANGLES
        obs *= OBS_SCALE
        obs = np.clip(obs, -100.0, 100.0)
        return obs

    def _control_loop(self):
        if self.latest_msg is None:
            return

        if self.state == State.IDLE:
            return

        elif self.state == State.STAND_UP:
            elapsed = time.monotonic() - self.standup_start_time
            alpha = min(elapsed / self.standup_duration, 1.0)
            alpha = 3 * alpha**2 - 2 * alpha**3
            target = (1 - alpha) * self.standup_start_pos + alpha * DEFAULT_JOINT_ANGLES
            joint_pos, joint_vel = self._get_joint_state_rl(self.latest_msg)
            torques = self._compute_torque_pd(target, joint_pos, joint_vel)
            self.cmd_pub.publish(self._make_torque_cmd(torques))
            if elapsed >= self.standup_duration:
                self.state = State.STANDING
                self._print_status()

        elif self.state == State.STANDING:
            joint_pos, joint_vel = self._get_joint_state_rl(self.latest_msg)
            torques = self._compute_torque_pd(DEFAULT_JOINT_ANGLES, joint_pos, joint_vel)
            self.cmd_pub.publish(self._make_torque_cmd(torques))

        elif self.state == State.RL:
            if self.step_counter % self.n_intermediate_steps == 0:
                self.target_heading += self.desired_yaw_rate * self.control_dt
                obs = self._build_observation(self.latest_msg)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                if self.obs_mask is not None:
                    obs_t = obs_t[:, self.obs_mask.bool()]
                with torch.no_grad():
                    self.current_raw_action = self.policy_net(obs_t).squeeze(0).cpu().numpy()
                self.prev_raw_action = self.current_raw_action

            # PI torque computed every sim step
            joint_pos, joint_vel = self._get_joint_state_rl(self.latest_msg)
            torques = self._compute_torque_pi(self.current_raw_action, joint_vel)
            self.cmd_pub.publish(self._make_torque_cmd(torques))
            self.step_counter += 1


def keyboard_thread(node):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while rclpy.ok():
            ch = sys.stdin.read(1)
            if ch:
                node.transition(ch.lower())
    except SystemExit:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_path', type=str, default="/workspace/policy_vel.pt")
    parser.add_argument('--control_dt', type=float, default=0.02)
    parser.add_argument('--n_intermediate_steps', type=int, default=4)
    parser.add_argument('--standup_duration', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f'Loading policy from {args.policy_path}')
    policy_net, obs_mask = load_policy(args.policy_path, args.device)

    rclpy.init()
    node = Go2DeployNode(
        policy_net, obs_mask, args.device,
        args.control_dt, args.standup_duration,
        args.n_intermediate_steps)

    print('\n--- Keyboard Controls ---')
    print('  s : stand up')
    print('  r : start RL policy')
    print('  e : emergency stop')
    print('  q : quit')
    print('-------------------------\n')

    kb = threading.Thread(target=keyboard_thread, args=(node,), daemon=True)
    kb.start()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.get_logger().info('Shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()