#!/usr/bin/env python3
"""
Script to analyze Betaflight blackbox .bbl files and visualize in Rerun.

This script:
1. Copies the .bbl file to /tmp
2. Runs blackbox_decode to generate CSV
3. Logs all fields to Rerun with:
   - Raw RC channel values
   - Semantic RC channel names and transformed values (as passed to policy)
   - Drone pose visualization using log_drone_pose
   - All other blackbox fields

Usage:
    python analyze_blackbox.py <path_to_bbl_file>
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import rerun as rr
    import numpy as np
    from scipy.spatial.transform import Rotation
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install rerun-sdk numpy scipy")
    sys.exit(1)

# Import log_drone_pose from sitl package
sys.path.insert(0, str(Path(__file__).parent.parent / "sitl"))
try:
    from sitl.logging_utils import log_drone_pose
except ImportError:
    print("Warning: Could not import log_drone_pose from sitl package")
    print("Drone pose visualization will not be available")
    log_drone_pose = None


# RC channel mapping based on policy.cpp
# Channels 7-15 carry state information from simulator to firmware
RC_CHANNEL_MAPPING = {
    7: "position_x",      # World frame X position
    8: "position_y",      # World frame Y position
    9: "position_z",      # World frame Z position
    10: "velocity_x",     # World frame X velocity
    11: "velocity_y",     # World frame Y velocity
    12: "velocity_z",     # World frame Z velocity
    13: "rotation_vec_x", # Rotation vector X
    14: "rotation_vec_y", # Rotation vector Y
    15: "rotation_vec_z", # Rotation vector Z
}


def from_channel(rc_value):
    """
    Convert RC channel value to [-1, 1] range.
    Based on policy.cpp from_channel function.
    RC channels are typically in range [1000, 2000].
    """
    return (rc_value - 1500.0) / 500.0


def rotation_vector_to_quaternion(rv):
    """
    Convert rotation vector to quaternion [w, x, y, z].
    Using scipy's Rotation class.
    """
    angle = np.linalg.norm(rv)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    rot = Rotation.from_rotvec(axis * angle)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix (body to world).
    """
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy expects [x, y, z, w]
    return rot.as_matrix()


def transform_to_body_frame(world_vec, rotation_matrix):
    """
    Transform a world frame vector to body frame using R^T.
    """
    return rotation_matrix.T @ world_vec


def compute_policy_observations(rc_data, gyro_adc):
    """
    Compute the 12-dimensional observation vector that would be passed to the policy.
    Based on policy.cpp lines 352-392.

    Returns:
        dict with policy observation components
    """
    # Extract state from RC channels
    position = np.array([
        from_channel(rc_data.get(7, 1500)),
        from_channel(rc_data.get(8, 1500)),
        from_channel(rc_data.get(9, 1500)),
    ])

    velocity_world = np.array([
        from_channel(rc_data.get(10, 1500)),
        from_channel(rc_data.get(11, 1500)),
        from_channel(rc_data.get(12, 1500)),
    ])

    rotation_vec = np.array([
        from_channel(rc_data.get(13, 1500)),
        from_channel(rc_data.get(14, 1500)),
        from_channel(rc_data.get(15, 1500)),
    ])

    # Convert rotation vector to quaternion
    quat = rotation_vector_to_quaternion(rotation_vec)

    # Get rotation matrix (body to world)
    R = quaternion_to_rotation_matrix(quat)

    # Transform velocity to body frame
    body_linear_velocity = transform_to_body_frame(velocity_world, R)

    # Body frame angular velocity from gyro (convert deg/s to rad/s)
    GYRO_CONVERSION_FACTOR = np.pi / 180.0
    body_angular_velocity = np.array([
        gyro_adc.get(0, 0.0) * GYRO_CONVERSION_FACTOR,
        gyro_adc.get(1, 0.0) * GYRO_CONVERSION_FACTOR,
        gyro_adc.get(2, 0.0) * GYRO_CONVERSION_FACTOR,
    ])

    # Body-projected gravity vector
    gravity_world = np.array([0.0, 0.0, -1.0])
    body_gravity = transform_to_body_frame(gravity_world, R)

    # Body frame position setpoint (assuming target is [0, 0, 1])
    target_position = np.array([0.0, 0.0, 1.0])
    position_error_world = target_position - position
    body_position_setpoint = transform_to_body_frame(position_error_world, R)

    return {
        'position_world': position,
        'velocity_world': velocity_world,
        'velocity_body': body_linear_velocity,
        'rotation_vector': rotation_vec,
        'quaternion': quat,
        'rotation_matrix': R,
        'angular_velocity_body': body_angular_velocity,
        'gravity_body': body_gravity,
        'position_setpoint_body': body_position_setpoint,
    }


def decode_blackbox(bbl_path, blackbox_decode_path):
    """
    Decode .bbl file to CSV using blackbox_decode.
    Returns path to generated CSV file.
    """
    bbl_path = Path(bbl_path)

    # Create temp directory and copy .bbl file
    temp_dir = Path(tempfile.mkdtemp(prefix="blackbox_analysis_"))
    temp_bbl = temp_dir / bbl_path.name
    shutil.copy2(bbl_path, temp_bbl)

    print(f"Copied {bbl_path} to {temp_bbl}")

    # Run blackbox_decode
    print(f"Running blackbox_decode...")
    try:
        result = subprocess.run(
            [str(blackbox_decode_path), str(temp_bbl)],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running blackbox_decode: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)

    # Find generated CSV file (should have same name with .csv extension)
    csv_files = list(temp_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV file generated in {temp_dir}")
        sys.exit(1)

    csv_path = csv_files[0]
    print(f"Generated CSV: {csv_path}")

    return csv_path


def log_to_rerun(csv_path):
    """
    Read decoded CSV and log all fields to Rerun.
    """
    csv_path = Path(csv_path)

    # Initialize Rerun
    rr.init("blackbox_analyzer", spawn=True)

    print(f"Loading CSV: {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)

        # Read header
        header = next(reader)
        columns = [col.strip() for col in header]

        print(f"Found {len(columns)} columns")

        # Process each row
        row_count = 0
        for row in reader:
            if len(row) != len(columns):
                continue

            # Parse row data
            data = {}
            for col_name, value_str in zip(columns, row):
                value_str = value_str.strip()
                if not value_str:
                    data[col_name] = None
                else:
                    try:
                        if '.' in value_str:
                            data[col_name] = float(value_str)
                        else:
                            data[col_name] = int(value_str)
                    except ValueError:
                        data[col_name] = value_str

            # Set timeline
            time_us = data.get('time (us)')
            if time_us is None:
                continue

            rr.set_time("time_us", timestamp=time_us * 1e-6)
            rr.set_time("loop_iteration", sequence=data.get('loopIteration', row_count))

            # ========== RAW RC CHANNELS ==========
            rc_data = {}
            for i in range(18):  # Betaflight typically has up to 18 RC channels
                key = f'rcCommand[{i}]'
                if key in data and data[key] is not None:
                    rr.log(f"rc/raw/channel_{i}", rr.Scalars(data[key]))
                    rc_data[i] = data[key]

            # ========== SEMANTIC RC CHANNELS ==========
            # Log the semantic meaning of RC channels 7-15 (state from simulator)
            for ch_num, semantic_name in RC_CHANNEL_MAPPING.items():
                key = f'rcCommand[{ch_num}]'
                if key in data and data[key] is not None:
                    raw_value = data[key]
                    transformed_value = from_channel(raw_value)

                    rr.log(f"rc/semantic/{semantic_name}/raw", rr.Scalars(raw_value))
                    rr.log(f"rc/semantic/{semantic_name}/transformed", rr.Scalars(transformed_value))

            # ========== POLICY OBSERVATIONS ==========
            # Compute the full observation vector as it would be passed to the policy
            gyro_adc = {}
            for i in range(3):
                key = f'gyroADC[{i}]'
                if key in data and data[key] is not None:
                    gyro_adc[i] = data[key]

            if len(rc_data) >= 16 and len(gyro_adc) == 3:
                try:
                    obs = compute_policy_observations(rc_data, gyro_adc)

                    # Log body frame velocity (NN input [0:3])
                    rr.log("policy_obs/body_linear_velocity/x", rr.Scalars(obs['velocity_body'][0]))
                    rr.log("policy_obs/body_linear_velocity/y", rr.Scalars(obs['velocity_body'][1]))
                    rr.log("policy_obs/body_linear_velocity/z", rr.Scalars(obs['velocity_body'][2]))

                    # Log body frame angular velocity (NN input [3:6])
                    rr.log("policy_obs/body_angular_velocity/x", rr.Scalars(obs['angular_velocity_body'][0]))
                    rr.log("policy_obs/body_angular_velocity/y", rr.Scalars(obs['angular_velocity_body'][1]))
                    rr.log("policy_obs/body_angular_velocity/z", rr.Scalars(obs['angular_velocity_body'][2]))

                    # Log body-projected gravity (NN input [6:9])
                    rr.log("policy_obs/body_gravity/x", rr.Scalars(obs['gravity_body'][0]))
                    rr.log("policy_obs/body_gravity/y", rr.Scalars(obs['gravity_body'][1]))
                    rr.log("policy_obs/body_gravity/z", rr.Scalars(obs['gravity_body'][2]))

                    # Log body frame position setpoint (NN input [9:12])
                    rr.log("policy_obs/body_position_setpoint/x", rr.Scalars(obs['position_setpoint_body'][0]))
                    rr.log("policy_obs/body_position_setpoint/y", rr.Scalars(obs['position_setpoint_body'][1]))
                    rr.log("policy_obs/body_position_setpoint/z", rr.Scalars(obs['position_setpoint_body'][2]))

                    # Log world frame data for reference
                    rr.log("state/position_world/x", rr.Scalars(obs['position_world'][0]))
                    rr.log("state/position_world/y", rr.Scalars(obs['position_world'][1]))
                    rr.log("state/position_world/z", rr.Scalars(obs['position_world'][2]))

                    rr.log("state/velocity_world/x", rr.Scalars(obs['velocity_world'][0]))
                    rr.log("state/velocity_world/y", rr.Scalars(obs['velocity_world'][1]))
                    rr.log("state/velocity_world/z", rr.Scalars(obs['velocity_world'][2]))

                    # Log quaternion
                    rr.log("state/quaternion/w", rr.Scalars(obs['quaternion'][0]))
                    rr.log("state/quaternion/x", rr.Scalars(obs['quaternion'][1]))
                    rr.log("state/quaternion/y", rr.Scalars(obs['quaternion'][2]))
                    rr.log("state/quaternion/z", rr.Scalars(obs['quaternion'][3]))

                    # ========== DRONE POSE VISUALIZATION ==========
                    if log_drone_pose is not None:
                        # Convert quaternion from [w, x, y, z] to [x, y, z, w] for scipy
                        quat_xyzw = np.array([
                            obs['quaternion'][1],
                            obs['quaternion'][2],
                            obs['quaternion'][3],
                            obs['quaternion'][0]
                        ])
                        log_drone_pose(obs['position_world'], quat_xyzw)

                except Exception as e:
                    print(f"Warning: Error computing policy observations at row {row_count}: {e}")

            # ========== ALL OTHER BLACKBOX FIELDS ==========
            # Log PID values
            for axis, axis_name in enumerate(['roll', 'pitch', 'yaw']):
                for term in ['P', 'I', 'D', 'F']:
                    key = f'axis{term}[{axis}]'
                    if key in data and data[key] is not None:
                        rr.log(f"pid/{term}/{axis_name}", rr.Scalars(data[key]))

            # Log gyro data
            for i, axis in enumerate(['roll', 'pitch', 'yaw']):
                for gyro_type in ['gyroADC', 'gyroUnfilt']:
                    key = f'{gyro_type}[{i}]'
                    if key in data and data[key] is not None:
                        rr.log(f"sensors/{gyro_type}/{axis}", rr.Scalars(data[key]))

            # Log accelerometer
            for i, axis in enumerate(['x', 'y', 'z']):
                key = f'accSmooth[{i}]'
                if key in data and data[key] is not None:
                    rr.log(f"sensors/acc/{axis}", rr.Scalars(data[key]))

            # Log setpoints
            for i, name in enumerate(['roll', 'pitch', 'yaw', 'throttle']):
                key = f'setpoint[{i}]'
                if key in data and data[key] is not None:
                    rr.log(f"setpoint/{name}", rr.Scalars(data[key]))

            # Log motors
            for i in range(4):
                key = f'motor[{i}]'
                if key in data and data[key] is not None:
                    rr.log(f"motors/output/m{i}", rr.Scalars(data[key]))

            # Log eRPM
            for i in range(4):
                key = f'eRPM[{i}]'
                if key in data and data[key] is not None:
                    rr.log(f"motors/erpm/m{i}", rr.Scalars(data[key]))

            # Log battery
            if 'vbatLatest (V)' in data and data['vbatLatest (V)'] is not None:
                rr.log("battery/voltage", rr.Scalars(data['vbatLatest (V)']))
            if 'amperageLatest (A)' in data and data['amperageLatest (A)'] is not None:
                rr.log("battery/current", rr.Scalars(data['amperageLatest (A)']))
            if 'energyCumulative (mAh)' in data and data['energyCumulative (mAh)'] is not None:
                rr.log("battery/energy", rr.Scalars(data['energyCumulative (mAh)']))

            # Log debug values
            for i in range(8):
                key = f'debug[{i}]'
                if key in data and data[key] is not None:
                    rr.log(f"debug/{i}", rr.Scalars(data[key]))

            # Log RSSI
            if 'rssi' in data and data['rssi'] is not None:
                rr.log("radio/rssi", rr.Scalars(data['rssi']))

            # Log flags as text
            if 'flightModeFlags (flags)' in data and data['flightModeFlags (flags)'] is not None:
                rr.log("status/flight_mode", rr.TextLog(str(data['flightModeFlags (flags)'])))
            if 'failsafePhase (flags)' in data and data['failsafePhase (flags)'] is not None:
                rr.log("status/failsafe", rr.TextLog(str(data['failsafePhase (flags)'])))

            row_count += 1
            if row_count % 1000 == 0:
                print(f"Processed {row_count} rows...")

    print(f"\nCompleted! Logged {row_count} rows to Rerun.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Betaflight blackbox .bbl files and visualize in Rerun"
    )
    parser.add_argument(
        "bbl_file",
        type=str,
        help="Path to the .bbl blackbox file"
    )
    parser.add_argument(
        "--blackbox-decode",
        type=str,
        default=None,
        help="Path to blackbox_decode binary (default: analysis/blackbox-tools/obj/blackbox_decode)"
    )

    args = parser.parse_args()

    # Find blackbox_decode binary
    if args.blackbox_decode:
        blackbox_decode_path = Path(args.blackbox_decode)
    else:
        # Default location
        script_dir = Path(__file__).parent
        blackbox_decode_path = script_dir / "blackbox-tools" / "obj" / "blackbox_decode"

    if not blackbox_decode_path.exists():
        print(f"Error: blackbox_decode not found at {blackbox_decode_path}")
        print("\nPlease build blackbox-tools first:")
        print("  cd analysis/blackbox-tools")
        print("  make")
        sys.exit(1)

    # Decode blackbox file
    csv_path = decode_blackbox(args.bbl_file, blackbox_decode_path)

    # Log to Rerun
    log_to_rerun(csv_path)


if __name__ == "__main__":
    main()
