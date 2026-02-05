# Plan: Save .rrd from real_deployment + sync with blackbox in analyze_blackbox

## Overview

Two changes:
1. **real_deployment.py**: Always save an `.rrd` file alongside the live viewer
2. **analyze_blackbox.py**: Accept optional `--rrd` arg, load vicon data from it using `rr.server.Server` query API, cross-correlate RC throttle to find time offset, re-log vicon data on the blackbox timeline

## Changes

### 1. `deployment/real_deployment.py`

**In `run()` after `rr.init("RealDeployment", spawn=True)`:**
- Generate timestamped path: `deployment/logs/deployment_YYYYMMDD_HHMMSS.rrd`
- `os.makedirs` the directory
- Call `rr.save(rrd_path)` — this adds a file sink alongside the spawn viewer
- Print the save path

That's it — the existing `_flush_log_buffer()` and `_log_to_rerun()` already write all data to the active recording, which will go to both the viewer and the file.

### 2. `analysis/analyze_blackbox.py`

**Add `--rrd` CLI argument** (optional path to `.rrd` file from real_deployment).

**Add `load_vicon_from_rrd(rrd_path)` function:**
1. `with rr.server.Server(datasets={"vicon": [rrd_path]}) as server:`
2. Query `dataset.reader(index="time", fill_latest_at=True).to_pandas()`
3. Extract columns:
   - Timestamps: `df["time"]` → convert datetime64 to epoch float seconds
   - Joystick: `df["/rc/joystick:Scalars:scalars"]` → stack to (N, 8) array, extract throttle `[:, 2]`
   - World position: `df["/world/position:Scalars:scalars"]` → (N, 3)
   - World velocity: `df["/world/velocity:Scalars:scalars"]` → (N, 3)
   - Body setpoint error: `df["/body/setpoint_error:Scalars:scalars"]` → (N, 3)
   - Body velocity: `df["/body/velocity:Scalars:scalars"]` → (N, 3)
   - Quaternion xyz: `df["/world/quaternion_xyz:Scalars:scalars"]` → (N, 3)
   - Vicon valid: `df["/status/vicon_valid:Scalars:scalars"]` → (N,)
4. Return dict with all arrays

**Add `compute_time_offset(blackbox_throttle, blackbox_times, vicon_throttle, vicon_times)` function:**
1. Resample both throttle signals to a common 100Hz grid via `np.interp`
2. Normalize both (subtract mean, divide by std)
3. Cross-correlate via `np.correlate(a, b, mode="full")` to find lag
4. Convert peak lag from samples to seconds
5. Return offset such that `vicon_time + offset ≈ blackbox_time`

**Add `log_vicon_data(vicon_data, time_offset, times_sec, loop_iterations)` function:**
1. Shift vicon timestamps: `synced_times = vicon_epoch_times + time_offset`
2. Create a `loop_iteration` array for vicon data by interpolating from blackbox timestamps
3. Use `send_scalar_column` to log under `vicon/` prefix:
   - `vicon/world_position/{x,y,z}`
   - `vicon/world_velocity/{x,y,z}`
   - `vicon/body_setpoint_error/{x,y,z}`
   - `vicon/body_velocity/{x,y,z}`
   - `vicon/vicon_valid`
4. Batch-log drone pose via `rr.send_columns` with `Transform3D.columns()`

**In `log_to_rerun()`:** Accept `rrd_path` parameter. After all blackbox data is sent, if `rrd_path` was provided:
1. Call `load_vicon_from_rrd(rrd_path)`
2. Extract blackbox throttle from `data["rcCommand[2]"]`
3. Call `compute_time_offset(...)`
4. Call `log_vicon_data(...)` to re-log on synced timeline

## Files to modify
- `deployment/real_deployment.py` — add `rr.save()` call + directory creation
- `analysis/analyze_blackbox.py` — add `--rrd` arg, vicon loading, cross-correlation sync, vicon logging

## Verification
- `uv run analysis/analyze_blackbox.py logs/run_7.bbl` — still works without `--rrd`
- Can't fully test vicon sync without real deployment data, but can verify the rrd loading path doesn't crash
