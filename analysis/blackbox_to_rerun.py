#!/usr/bin/env python3
"""
Script to visualize flight controller CSV logs using Rerun.

Uses rr.send_columns for efficient batch logging instead of per-row rr.log calls.

Usage:
    python blackbox_to_rerun.py <path_to_csv_file>
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import rerun as rr
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install rerun-sdk numpy")
    sys.exit(1)


def parse_csv_header(header_row):
    """Parse the CSV header to extract column names."""
    columns = [col.strip() for col in header_row]
    return columns


def convert_value(value_str):
    """Convert a string value to appropriate numeric type."""
    value_str = value_str.strip()
    if not value_str:
        return np.nan
    try:
        if '.' not in value_str:
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        return np.nan


def load_csv_data(csv_path):
    """Load CSV file into a dictionary of numpy arrays."""
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = parse_csv_header(header)

        # Initialize lists for each column
        data = {col: [] for col in columns}

        for row in reader:
            if len(row) != len(columns):
                continue
            for col_name, value_str in zip(columns, row):
                data[col_name].append(convert_value(value_str))

    # Convert to numpy arrays
    for col in data:
        data[col] = np.array(data[col])

    return data


def send_scalar_columns(entity_path, times, values, timeline_name="time"):
    """Send scalar data using send_columns."""
    # Handle NaN values by masking
    valid_mask = ~np.isnan(values) if values.ndim == 1 else ~np.any(np.isnan(values), axis=1)
    if not np.any(valid_mask):
        return

    valid_times = times[valid_mask]
    valid_values = values[valid_mask]

    rr.send_columns(
        entity_path,
        indexes=[rr.TimeColumn(timeline_name, timestamp=valid_times)],
        columns=rr.Scalars.columns(scalars=valid_values),
    )


def log_to_rerun(csv_path):
    """Read CSV file and log all data to Rerun using send_columns."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Initialize Rerun
    rr.init("flight_log_viewer", spawn=True)

    print(f"Loading log file: {csv_path}")
    data = load_csv_data(csv_path)
    row_count = len(data.get('time (us)', []))
    print(f"Loaded {row_count} rows")

    # Get time column (convert microseconds to seconds)
    if 'time (us)' not in data:
        print("Error: No 'time (us)' column found")
        sys.exit(1)

    times = data['time (us)'] * 1e-6  # Convert to seconds

    print("Sending data to Rerun...")

    # PID P values
    if all(f'axisP[{i}]' in data for i in range(3)):
        values = np.column_stack([data[f'axisP[{i}]'] for i in range(3)])
        send_scalar_columns("pid/P", times, values)

    # PID I values
    if all(f'axisI[{i}]' in data for i in range(3)):
        values = np.column_stack([data[f'axisI[{i}]'] for i in range(3)])
        send_scalar_columns("pid/I", times, values)

    # PID D values (only roll and pitch typically)
    if all(f'axisD[{i}]' in data for i in range(2)):
        values = np.column_stack([data[f'axisD[{i}]'] for i in range(2)])
        send_scalar_columns("pid/D", times, values)

    # PID F values
    if all(f'axisF[{i}]' in data for i in range(3)):
        values = np.column_stack([data[f'axisF[{i}]'] for i in range(3)])
        send_scalar_columns("pid/F", times, values)

    # Gyro ADC
    if all(f'gyroADC[{i}]' in data for i in range(3)):
        values = np.column_stack([data[f'gyroADC[{i}]'] for i in range(3)])
        send_scalar_columns("sensors/gyro_adc", times, values)

    # Gyro unfiltered
    if all(f'gyroUnfilt[{i}]' in data for i in range(3)):
        values = np.column_stack([data[f'gyroUnfilt[{i}]'] for i in range(3)])
        send_scalar_columns("sensors/gyro_unfilt", times, values)

    # Accelerometer
    if all(f'accSmooth[{i}]' in data for i in range(3)):
        values = np.column_stack([data[f'accSmooth[{i}]'] for i in range(3)])
        send_scalar_columns("sensors/acc", times, values)

    # RC commands
    if all(f'rcCommand[{i}]' in data for i in range(4)):
        values = np.column_stack([data[f'rcCommand[{i}]'] for i in range(4)])
        send_scalar_columns("rc/command", times, values)

    # Setpoints
    if all(f'setpoint[{i}]' in data for i in range(4)):
        values = np.column_stack([data[f'setpoint[{i}]'] for i in range(4)])
        send_scalar_columns("setpoint", times, values)

    # Motor outputs
    if all(f'motor[{i}]' in data for i in range(4)):
        values = np.column_stack([data[f'motor[{i}]'] for i in range(4)])
        send_scalar_columns("motors/output", times, values)

    # eRPM
    if all(f'eRPM[{i}]' in data for i in range(4)):
        values = np.column_stack([data[f'eRPM[{i}]'] for i in range(4)])
        send_scalar_columns("motors/erpm", times, values)

    # Battery voltage
    if 'vbatLatest (V)' in data:
        send_scalar_columns("battery/voltage", times, data['vbatLatest (V)'])

    # Battery current
    if 'amperageLatest (A)' in data:
        send_scalar_columns("battery/current", times, data['amperageLatest (A)'])

    # Battery energy
    if 'energyCumulative (mAh)' in data:
        send_scalar_columns("battery/energy", times, data['energyCumulative (mAh)'])

    # RSSI
    if 'rssi' in data:
        send_scalar_columns("radio/rssi", times, data['rssi'])

    # Debug values
    debug_cols = [f'debug[{i}]' for i in range(8) if f'debug[{i}]' in data]
    if debug_cols:
        values = np.column_stack([data[col] for col in debug_cols])
        send_scalar_columns("debug", times, values)

    # RC data channels (for NN state observation)
    rc_data_cols = [f'rcData[{i}]' for i in range(7, 16) if f'rcData[{i}]' in data]
    if rc_data_cols:
        values = np.column_stack([data[col] for col in rc_data_cols])
        send_scalar_columns("rc/data", times, values)

    # Flight mode and failsafe as text logs (these need per-row logging)
    if 'flightModeFlags (flags)' in data:
        mode_data = data['flightModeFlags (flags)']
        # Only log when mode changes to reduce overhead
        prev_mode = None
        for i, mode in enumerate(mode_data):
            if mode != prev_mode and not np.isnan(mode) if isinstance(mode, float) else mode is not None:
                rr.set_time("time", timestamp=times[i])
                rr.log("status/flight_mode", rr.TextLog(str(int(mode) if isinstance(mode, float) else mode)))
                prev_mode = mode

    if 'failsafePhase (flags)' in data:
        failsafe_data = data['failsafePhase (flags)']
        prev_failsafe = None
        for i, failsafe in enumerate(failsafe_data):
            if failsafe != prev_failsafe and not np.isnan(failsafe) if isinstance(failsafe, float) else failsafe is not None:
                rr.set_time("time", timestamp=times[i])
                rr.log("status/failsafe", rr.TextLog(str(int(failsafe) if isinstance(failsafe, float) else failsafe)))
                prev_failsafe = failsafe

    print(f"\nCompleted! Logged {row_count} rows to Rerun.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize flight controller CSV logs using Rerun"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV log file"
    )

    args = parser.parse_args()
    log_to_rerun(args.csv_file)


if __name__ == "__main__":
    main()
