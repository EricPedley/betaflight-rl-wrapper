#!/usr/bin/env python3
"""
Script to analyze Betaflight blackbox .bbl files and visualize in Rerun.

This script:
1. Copies the .bbl file to /tmp
2. Runs blackbox_decode to generate CSV
3. Logs all fields to Rerun with:
   - Raw RC channel values
   - Drone pose visualization from debug values
   - All other blackbox fields

Uses rr.send_columns for efficient batch logging instead of per-row rr.log calls.

Usage:
    python analyze_blackbox.py <path_to_bbl_file>
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
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

# Debug field mapping for RL_TOOLS debug mode
# These fields are populated in policy.cpp when debug_mode = RL_TOOLS
RL_TOOLS_DEBUG_MAPPING = {
    0: ("body_setpoint_error_x", 0.001),   # Scaled by 1000 in firmware
    1: ("body_setpoint_error_y", 0.001),
    2: ("body_setpoint_error_z", 0.001),
    3: ("quaternion_w", 0.0001),            # Scaled by 10000 in firmware
    4: ("quaternion_x", 0.0001),
    5: ("quaternion_y", 0.0001),
    6: ("quaternion_z", 0.0001),
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
    csv_files = sorted(list(temp_dir.glob("*.csv")))
    if not csv_files:
        print(f"Error: No CSV file generated in {temp_dir}")
        sys.exit(1)

    csv_path = csv_files[-1]
    print(f"Generated CSV: {csv_path}")

    return csv_path


def load_csv_data(csv_path):
    """Load CSV file into a dictionary of numpy arrays (float64) or lists (strings)."""
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = [col.strip() for col in header]

        raw_data = {col: [] for col in columns}

        for row in reader:
            if len(row) != len(columns):
                continue
            for col_name, value_str in zip(columns, row):
                value_str = value_str.strip()
                if not value_str:
                    raw_data[col_name].append(np.nan)
                else:
                    try:
                        raw_data[col_name].append(float(value_str))
                    except ValueError:
                        raw_data[col_name].append(value_str)

    # Convert to numpy arrays where possible
    data = {}
    for col in raw_data:
        vals = raw_data[col]
        # Check if any value is a string (not just the first)
        has_strings = any(isinstance(v, str) for v in vals)
        if has_strings:
            data[col] = vals  # Keep as list for string columns
        else:
            data[col] = np.array(vals, dtype=np.float64)

    return data


def send_scalar_column(entity_path, times_sec, loop_iterations, values):
    """Send a 1D scalar column using send_columns, filtering NaN values."""
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return
    rr.send_columns(
        entity_path,
        indexes=[
            rr.TimeColumn("time_us", timestamp=times_sec[valid_mask]),
            rr.TimeColumn("loop_iteration", sequence=loop_iterations[valid_mask]),
        ],
        columns=rr.Scalars.columns(scalars=values[valid_mask]),
    )


def log_to_rerun(csv_path):
    """
    Read decoded CSV and log all fields to Rerun using send_columns for speed.
    """
    csv_path = Path(csv_path)

    # Initialize Rerun
    rr.init("blackbox_analyzer", spawn=True)

    print(f"Loading CSV: {csv_path}")

    # ===== PHASE 1: Load CSV data =====
    t_start = time.time()
    data = load_csv_data(csv_path)
    t_load = time.time()

    time_us = data.get('time (us)')
    if time_us is None or not isinstance(time_us, np.ndarray):
        print("Error: No 'time (us)' column found")
        sys.exit(1)

    N = len(time_us)
    print(f"Loaded {N} rows in {t_load - t_start:.3f}s")

    # Build timeline arrays
    times_sec = time_us * 1e-6
    loop_iter_col = data.get('loopIteration')
    if loop_iter_col is not None and isinstance(loop_iter_col, np.ndarray):
        loop_iterations = loop_iter_col.astype(np.int64)
    else:
        loop_iterations = np.arange(N, dtype=np.int64)

    # ===== PHASE 2: Send all data to Rerun via send_columns =====
    t_send_start = time.time()

    # --- Raw RC channels ---
    for i in range(18):
        key = f'rcCommand[{i}]'
        if key in data and isinstance(data[key], np.ndarray):
            send_scalar_column(f"rc/raw/channel_{i}", times_sec, loop_iterations, data[key])

    # --- PID values ---
    for axis, axis_name in enumerate(['roll', 'pitch', 'yaw']):
        for term in ['P', 'I', 'D', 'F']:
            key = f'axis{term}[{axis}]'
            if key in data and isinstance(data[key], np.ndarray):
                send_scalar_column(f"pid/{term}/{axis_name}", times_sec, loop_iterations, data[key])

    # --- Gyro data ---
    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        for gyro_type in ['gyroADC', 'gyroUnfilt']:
            key = f'{gyro_type}[{i}]'
            if key in data and isinstance(data[key], np.ndarray):
                send_scalar_column(f"sensors/{gyro_type}/{axis}", times_sec, loop_iterations, data[key])

    # --- Accelerometer ---
    for i, axis in enumerate(['x', 'y', 'z']):
        key = f'accSmooth[{i}]'
        if key in data and isinstance(data[key], np.ndarray):
            send_scalar_column(f"sensors/acc/{axis}", times_sec, loop_iterations, data[key])

    # --- Setpoints ---
    for i, name in enumerate(['roll', 'pitch', 'yaw', 'throttle']):
        key = f'setpoint[{i}]'
        if key in data and isinstance(data[key], np.ndarray):
            send_scalar_column(f"setpoint/{name}", times_sec, loop_iterations, data[key])

    # --- Motors ---
    for i in range(4):
        key = f'motor[{i}]'
        if key in data and isinstance(data[key], np.ndarray):
            send_scalar_column(f"motors/output/m{i}", times_sec, loop_iterations, data[key])

    # --- eRPM ---
    for i in range(4):
        key = f'eRPM[{i}]'
        if key in data and isinstance(data[key], np.ndarray):
            send_scalar_column(f"motors/erpm/m{i}", times_sec, loop_iterations, data[key])

    # --- Battery ---
    for col_name, entity in [
        ('vbatLatest (V)', 'battery/voltage'),
        ('amperageLatest (A)', 'battery/current'),
        ('energyCumulative (mAh)', 'battery/energy'),
    ]:
        if col_name in data and isinstance(data[col_name], np.ndarray):
            send_scalar_column(entity, times_sec, loop_iterations, data[col_name])

    # --- Debug values ---
    for i in range(8):
        key = f'debug[{i}]'
        if key in data and isinstance(data[key], np.ndarray):
            send_scalar_column(f"debug/{i}", times_sec, loop_iterations, data[key])

    # --- RL_TOOLS debug (scaled semantic values) ---
    has_debug = all(
        f'debug[{i}]' in data and isinstance(data[f'debug[{i}]'], np.ndarray)
        for i in range(7)
    )
    if has_debug:
        for i, (name, scale) in RL_TOOLS_DEBUG_MAPPING.items():
            send_scalar_column(
                f"rl_state/{name}",
                times_sec, loop_iterations, data[f'debug[{i}]'] * scale,
            )

    # --- RSSI ---
    if 'rssi' in data and isinstance(data['rssi'], np.ndarray):
        send_scalar_column("radio/rssi", times_sec, loop_iterations, data['rssi'])

    t_send = time.time()
    print(f"Sent scalar columns in {t_send - t_send_start:.3f}s")

    # ===== PHASE 3: Per-row logging (text logs, drone pose) =====
    t_perrow_start = time.time()

    # Flight mode flags (only log on change)
    if 'flightModeFlags (flags)' in data:
        flags = data['flightModeFlags (flags)']
        prev = None
        for i, flag in enumerate(flags if isinstance(flags, list) else flags.tolist()):
            if flag != prev and not (isinstance(flag, float) and np.isnan(flag)):
                rr.set_time("time_us", timestamp=times_sec[i])
                rr.set_time("loop_iteration", sequence=int(loop_iterations[i]))
                rr.log("status/flight_mode", rr.TextLog(str(flag)))
                prev = flag

    if 'failsafePhase (flags)' in data:
        flags = data['failsafePhase (flags)']
        prev = None
        for i, flag in enumerate(flags if isinstance(flags, list) else flags.tolist()):
            if flag != prev and not (isinstance(flag, float) and np.isnan(flag)):
                rr.set_time("time_us", timestamp=times_sec[i])
                rr.set_time("loop_iteration", sequence=int(loop_iterations[i]))
                rr.log("status/failsafe", rr.TextLog(str(flag)))
                prev = flag

    # Drone pose visualization (batched via send_columns from debug values)
    drone_model = "drone/drone_model"
    drone_positions = None
    drone_quaternions = None

    if has_debug:
        drone_positions = np.column_stack([
            data['debug[0]'] * 0.001,
            data['debug[1]'] * 0.001,
            data['debug[2]'] * 0.001,
        ])
        drone_quaternions = np.column_stack([
            data['debug[4]'] * 0.0001,  # x
            data['debug[5]'] * 0.0001,  # y
            data['debug[6]'] * 0.0001,  # z
            data['debug[3]'] * 0.0001,  # w
        ])

    if drone_positions is not None:
        # Log static assets (Asset3D model + camera) once
        drone_obj_path = Path(__file__).parent.parent / "sitl" / "sitl" / "Drone.obj"
        if drone_obj_path.exists():
            rr.log(drone_model, rr.Asset3D(path=drone_obj_path), static=True)
            rr.log(
                f"{drone_model}/camera",
                rr.Transform3D(
                    translation=[0.03, 0, 0.04],
                    quaternion=Rotation.from_euler('y', -20, degrees=True).as_quat(),
                ),
                rr.Pinhole(
                    fov_y=1.2,
                    aspect_ratio=1.7777778,
                    camera_xyz=rr.ViewCoordinates.FLU,
                    image_plane_distance=0.1,
                    color=[255, 128, 0],
                    line_width=0.003,
                ),
            )

        # Batch send Transform3D + axes for all timesteps
        # Normalize quaternions for rerun (scipy's as_quat may not be unit)
        quats_for_rr = Rotation.from_quat(drone_quaternions).as_quat()  # re-normalize

        rr.send_columns(
            drone_model,
            indexes=[
                rr.TimeColumn("time_us", timestamp=times_sec),
                rr.TimeColumn("loop_iteration", sequence=loop_iterations),
            ],
            columns=[
                *rr.Transform3D.columns(
                    translation=drone_positions,
                    quaternion=quats_for_rr,
                ),
                *rr.TransformAxes3D.columns(
                    axis_length=np.full(N, 0.1, dtype=np.float32),
                ),
            ],
        )

    t_perrow = time.time()
    print(f"Per-row logging (text + drone pose) in {t_perrow - t_perrow_start:.3f}s")

    t_total = t_perrow - t_start
    print(f"\nCompleted! Logged {N} rows to Rerun.")
    print(f"  CSV loading:      {t_load - t_start:.3f}s")
    print(f"  send_columns:     {t_send - t_send_start:.3f}s")
    print(f"  Per-row logging:  {t_perrow - t_perrow_start:.3f}s")
    print(f"  Total:            {t_total:.3f}s")


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
