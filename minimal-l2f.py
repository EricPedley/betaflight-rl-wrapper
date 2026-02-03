from sitl.simulator import L2F
from sitl.gamepad import Gamepad
import asyncio
import os
import time
import argparse


async def main():
    print(f"PID: {os.getpid()}")
    parser = argparse.ArgumentParser(description="Run the L2F simulator with a gamepad.")
    parser.add_argument("gamepad_mapping", type=str, help="Path to the gamepad mapping JSON file.")
    parser.add_argument("--parameters", "-p", type=str, default='meteor75_parameters.json', help="Path to the quadrotor parameters JSON file.")
    args = parser.parse_args()
    time.sleep(1)
    simulator = L2F(START_SITL=False, parameters_file=args.parameters)
    gamepad = Gamepad(args.gamepad_mapping, simulator.set_joystick_channels)

    await asyncio.gather(simulator.run(), gamepad.run())
def sync_main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())


if __name__ == "__main__":
    sync_main()
