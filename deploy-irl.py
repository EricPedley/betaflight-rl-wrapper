"""
Deploy RL policy on real hardware via ELRS and optionally Vicon.

Usage:
    # Gamepad-only mode (no Vicon)
    uv run deploy-irl.py edgetx

    # With Vicon motion capture
    uv run deploy-irl.py edgetx --vicon-ip 192.168.30.153 --vicon-object meteor75

    # With custom ELRS port
    uv run deploy-irl.py edgetx --elrs-port /dev/ttyUSB1
"""

from deployment.real_deployment import RealDeployment
from sitl.gamepad import Gamepad
import asyncio
import argparse


async def main():
    parser = argparse.ArgumentParser(
        description="Deploy RL policy on real hardware via ELRS and optionally Vicon."
    )
    parser.add_argument(
        "gamepad_mapping",
        type=str,
        help="Path to gamepad mapping JSON file, or 'edgetx' for EdgeTX controller"
    )
    parser.add_argument(
        "--elrs-port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for ELRS module (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--elrs-baud",
        type=int,
        default=921600,
        help="Baud rate for ELRS (default: 921600)"
    )
    parser.add_argument(
        "--vicon-ip",
        type=str,
        default=None,
        help="Vicon DataStream server IP (omit for gamepad-only mode)"
    )
    parser.add_argument(
        "--vicon-object",
        type=str,
        default="meteor75",
        help="Name of tracked object in Vicon (default: meteor75)"
    )
    parser.add_argument(
        "--velocity-filter",
        type=float,
        default=0.3,
        help="Velocity filter alpha (0-1, higher = more responsive, default: 0.3)"
    )
    parser.add_argument(
        "--loop-rate",
        type=int,
        default=100,
        help="Main loop rate in Hz (default: 100)"
    )
    args = parser.parse_args()

    # Create deployment instance
    deployment = RealDeployment(
        elrs_port=args.elrs_port,
        elrs_baud=args.elrs_baud,
        vicon_ip=args.vicon_ip,
        vicon_object_name=args.vicon_object,
        velocity_filter_alpha=args.velocity_filter,
        loop_rate_hz=args.loop_rate,
    )

    # Create gamepad with callback
    gamepad = Gamepad(args.gamepad_mapping, deployment.set_joystick_channels)

    print("Starting deployment...")
    print(f"  Gamepad: {args.gamepad_mapping}")
    print(f"  ELRS: {args.elrs_port}")
    if args.vicon_ip:
        print(f"  Vicon: {args.vicon_ip} (object: {args.vicon_object})")
    else:
        print("  Vicon: disabled (gamepad-only mode)")
    print()

    # Run both concurrently
    await asyncio.gather(deployment.run(), gamepad.run())


def sync_main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())


if __name__ == "__main__":
    sync_main()
