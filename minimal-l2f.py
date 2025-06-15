import asyncio
import socket
import struct
import numpy as np
import pygame
import time
from functools import reduce

from copy import copy
import l2f
from l2f import vector8 as vector
from foundation_model import QuadrotorPolicy

# UDP Ports
PORT_PWM = 9002    # Receive RPMs (from Betaflight)
PORT_STATE = 9003  # Send state (to Betaflight)
PORT_RC = 9004     # Send RC input (to Betaflight)
UDP_IP = "127.0.0.1"
SIMULATOR_MAX_RC_CHANNELS=16 # https://github.com/betaflight/betaflight/blob/a94083e77d6258bbf9b8b5388a82af9498c923e9/src/platform/SIMULATOR/target/SITL/target.h#L238

policy = QuadrotorPolicy()
device = l2f.Device()
rng = vector.VectorRng()
env = vector.VectorEnvironment()
params = vector.VectorParameters()
state = vector.VectorState()
next_state = vector.VectorState()
observation = np.zeros((env.N_ENVIRONMENTS, env.OBSERVATION_DIM), dtype=np.float32)
vector.initialize_rng(device, rng, 0)
vector.initialize_environment(device, env)
vector.sample_initial_parameters(device, env, params, rng)
vector.initial_state(device, env, params, state)

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
else:
    joystick = None

gamepad_mapping = {
    "throttle": {"axis": 1, "invert": True},
    "yaw": {"axis": 0, "invert": False},
    "roll": {"axis": 2, "invert": False},
    "pitch": {"axis": 3, "invert": True},
    "arm": {"button": 10, "invert": False},
}

betaflight_order = ["roll", "pitch", "throttle", "yaw", "arm"] # AETR

initial_axes = None
def test_rc_channels():
    global initial_axes
    pygame.event.pump()
    axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
    if initial_axes is None:
        initial_axes = copy(axes)
    if sum(np.abs(np.array(axes) - initial_axes) > 0.25) == 1:
        diffs = np.abs(np.array(axes) - initial_axes)
        print("Axis: ", np.argmax(diffs), "Value: ", axes[np.argmax(diffs)])
    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
    for i, button in enumerate(buttons):
        if button:
            print(f"Button {i} pressed")

def get_rc_channels():
    pygame.event.pump()
    if joystick is None:
        return [1500] * SIMULATOR_MAX_RC_CHANNELS
    axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
    rc = []
    for key in betaflight_order:
        cfg = gamepad_mapping[key]
        if "axis" in cfg:
            idx = cfg["axis"]
            v = axes[idx] if idx < len(axes) else 0.0
        else:
            idx = cfg["button"]
            v = 1.0 if idx < len(buttons) and buttons[idx] else 0.0
        if cfg.get("invert", False):
            v = -v
        rc.append(int((v + 1) * 500 + 1000))
    rc = rc[:SIMULATOR_MAX_RC_CHANNELS]
    while len(rc) < SIMULATOR_MAX_RC_CHANNELS:
        rc.append(1000)
    return rc

def parse_rpm_packet(data):
    # 4 float32 = 16 bytes
    if len(data) >= 16:
        return struct.unpack('<4f', data[:16])
    return [0.0, 0.0, 0.0, 0.0]

def make_fdm_packet(state, drone_id=0):
    s = state.states[drone_id]
    timestamp = time.time()
    imu_angular_velocity_rpy = np.asarray(s.angular_velocity, dtype=np.float64)
    imu_linear_acceleration_xyz = np.zeros(3, dtype=np.float64) # todo
    imu_orientation_quat = np.asarray(s.orientation, dtype=np.float64)
    velocity_xyz = np.asarray(s.linear_velocity, dtype=np.float64)
    position_xyz = np.asarray(s.position, dtype=np.float64)
    pressure = 1013.25  # Dummy value for now
    fmt = '<d3d3d4d3d3dd'
    packet = struct.pack(fmt,
        timestamp,
        *imu_angular_velocity_rpy,
        *imu_linear_acceleration_xyz,
        *imu_orientation_quat,
        *velocity_xyz,
        *position_xyz,
        pressure
    )
    return packet

async def udp_recv(loop, sock):
    return await loop.sock_recv(sock, 1024)

async def main():
    loop = asyncio.get_running_loop()
    udp_pwm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_pwm_sock.bind((UDP_IP, PORT_PWM))
    udp_pwm_sock.setblocking(False)
    udp_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_rc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    policy.reset()
    armed = False
    while True:
        rc_channels = get_rc_channels()
        timestamp = time.time()
        rc_packet = struct.pack(f'<d{SIMULATOR_MAX_RC_CHANNELS}h', timestamp, *rc_channels)
        udp_rc_sock.sendto(rc_packet, (UDP_IP, PORT_RC))

        if rc_channels[betaflight_order.index("arm")] > 1500:
            policy.reset()
            vector.initial_state(device, env, params, state)
            armed = True

        try:
            data = await asyncio.wait_for(udp_recv(loop, udp_pwm_sock), timeout=0.01)
            rpms = parse_rpm_packet(data)
        except asyncio.TimeoutError:
            rpms = [0.0, 0.0, 0.0, 0.0]
        
        vector.observe(device, env, params, state, observation, rng)
        action = policy.evaluate_step(observation[:, :22])
        action[0] = np.array(rpms) * 2 - 1
        dts = vector.step(device, env, params, state, action, next_state, rng)
        if armed:
            state.assign(next_state)

        state_packet = make_fdm_packet(state)
        udp_state_sock.sendto(state_packet, (UDP_IP, PORT_STATE))


        print(f"RPMs: {rpms} ")
        await asyncio.sleep(dts[-1] if hasattr(dts, '__getitem__') else 0.01)
        # test_rc_channels()

if __name__ == "__main__":
    asyncio.run(main())
