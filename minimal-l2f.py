import asyncio
import socket
import struct
import numpy as np
import pygame
import time
import websockets
import json
from scipy.spatial.transform import Rotation as R
import subprocess
import os

from copy import copy
import l2f
from l2f import vector8 as vector
from foundation_model import QuadrotorPolicy


subprocess.Popen(["./betaflight/obj/betaflight_4.6.0_SITL"], cwd=os.path.dirname(os.path.abspath(__file__)))
time.sleep(2)

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
ui = l2f.UI()
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
# gamepad_mapping = {
#     "throttle": {"axis": 1, "invert": True},
#     "yaw": {"axis": 0, "invert": False},
#     "roll": {"axis": 3, "invert": False},
#     "pitch": {"axis": 4, "invert": True},
#     "arm": {"button": 5, "invert": False},
# }

betaflight_order = ["roll", "pitch", "throttle", "yaw", "arm"] # AETR
# crazyflie: top-right, bottom-right, bottom-left, top-left
# betaflight: front-left, front-right, rear-right, rear-left
# betaflight-sitl (uses PX4 mapping): top-right, bottom-left, top-lef, bottom-right,  # https://github.com/betaflight/betaflight/blob/a94083e77d6258bbf9b8b5388a82af9498c923e9/src/platform/SIMULATOR/sitl.c#L602
# crazyflie_from_betaflight_motors = [1, 0, 2, 3]
# crazyflie_from_betaflight_motors = [1, 0, 2, 3]
crazyflie_from_betaflight_motors = [0, 3, 1, 2]

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

def flu_to_frd(vec):
    if len(vec) == 3:
        return np.array([vec[0], -vec[1], -vec[2]], dtype=np.float64)
    elif len(vec) == 4: # Quaternion (w, x, y, z)
        return np.array([vec[0], vec[1], -vec[2], -vec[3]], dtype=np.float64)
def make_fdm_packet(state, accelerometer, drone_id=0):
    s = state.states[drone_id]
    timestamp = time.time()
    imu_angular_velocity_rpy = np.asarray(s.angular_velocity, dtype=np.float64)
    imu_orientation_quat = np.asarray(s.orientation, dtype=np.float64)
    velocity_xyz = np.asarray(s.linear_velocity, dtype=np.float64)
    position_xyz = np.asarray(s.position, dtype=np.float64)
    pressure = 101325  # Dummy value for now
    fmt = '<d3d3d4d3d3dd'
    packet = struct.pack(fmt,
        timestamp,
        *flu_to_frd(imu_angular_velocity_rpy),
        *-accelerometer,
        *imu_orientation_quat,
        *[0, 0, 0],
        *[0, 0, 0],
        pressure
    )
    return packet

async def udp_recv(loop, sock):
    return await loop.sock_recv(sock, 1024)

async def main():
    uri = "ws://localhost:13337/backend"
    async with websockets.connect(uri) as websocket:
        handshake = json.loads(await websocket.recv(uri))
        assert(handshake["channel"] == "handshake")
        namespace = handshake["data"]["namespace"]
        ui.ns = namespace
        ui_message = vector.set_ui_message(device, env, ui)
        parameters_message = vector.set_parameters_message(device, env, params, ui)
        await websocket.send(ui_message)
        await websocket.send(parameters_message)

        loop = asyncio.get_running_loop()
        udp_pwm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_pwm_sock.bind((UDP_IP, PORT_PWM))
        udp_pwm_sock.setblocking(False)
        udp_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_rc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        policy.reset()
        armed = False
        simulation_dts = []
        previous_time = time.time()
        while True:
            rc_channels = get_rc_channels()
            timestamp = time.time()
            rc_packet = struct.pack(f'<d{SIMULATOR_MAX_RC_CHANNELS}h', timestamp, *rc_channels)
            udp_rc_sock.sendto(rc_packet, (UDP_IP, PORT_RC))

            if rc_channels[betaflight_order.index("arm")] > 1500:
                if not armed:
                    policy.reset()
                    vector.initial_state(device, env, params, state)
                armed = True
            else:
                armed = False

            try:
                data = await asyncio.wait_for(udp_recv(loop, udp_pwm_sock), timeout=0.01)
                rpms = parse_rpm_packet(data)
                rpms = np.clip(rpms, -1.0, 1.0)
                assert all(rpms >= 0)
            except asyncio.TimeoutError:
                rpms = [0.0, 0.0, 0.0, 0.0]

            simulation_dt = time.time() - previous_time
            previous_time = time.time()
            simulation_dts.append(simulation_dt)
            simulation_dts = simulation_dts[-100:]

            for i in range(env.N_ENVIRONMENTS):
                parameters_string = l2f.parameters_to_json(device, env.environments[i], params.parameters[i])
                parameters = json.loads(parameters_string)
                parameters["integration"]["dt"] = simulation_dt
                l2f.parameters_from_json(device, env.environments[i], json.dumps(parameters), params.parameters[i])
            
            vector.observe(device, env, params, state, observation, rng)
            action = policy.evaluate_step(observation[:, :22])
            action[0] = np.array(rpms)[crazyflie_from_betaflight_motors] * 2 - 1
            dts = vector.step(device, env, params, state, action, next_state, rng)
            acceleration = (next_state.states[0].linear_velocity - state.states[0].linear_velocity) / simulation_dt
            r = R.from_quat([*state.states[0].orientation[1:], state.states[0].orientation[0]])
            R_wb = r.as_matrix()
            accelerometer = R_wb.T @ (acceleration - np.array([0, 0, -9.81], dtype=np.float64))
            if armed:
                state.assign(next_state)
            

            state.states[0].position[:] = 0
            state.states[0].linear_velocity[:] = 0
            # if state.states[0].position[2] <= -0.05:
            #     state.states[0].position[2] = 0
            #     state.states[0].linear_velocity[:] = 0
            #     state.states[0].orientation = [1, 0, 0, 0]
            #     state.states[0].angular_velocity[:] = 0

            ui_state = copy(state)
            for i, s in enumerate(ui_state.states):
                s.position[0] += i * 0.1 # Spacing for visualization
            state_action_message = vector.set_state_action_message(device, env, params, ui, ui_state, action)
            await websocket.send(state_action_message)

            state_packet = make_fdm_packet(state, accelerometer)
            udp_state_sock.sendto(state_packet, (UDP_IP, PORT_STATE))


            print(f"RPMs: {rpms} dt: {np.mean(simulation_dts):.4f} s, action: {action[0].tolist()}")
            # await asyncio.sleep(0.01)
            # test_rc_channels()

if __name__ == "__main__":
    asyncio.run(main())
