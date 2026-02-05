"""
Real-world deployment class that replaces L2F simulator.

Provides same interface as L2F:
- set_joystick_channels(joystick_values): Callback for gamepad input
- run(): Async main loop

Forwards gamepad RC channels (0-7) to ELRS transmitter.
Optionally populates state channels (7-15) from Vicon motion capture:
- Channels 7-9: Body frame setpoint error (direct NN input)
- Channels 10-12: Body frame velocity (direct NN input)
- Channels 13-15: Quaternion xyz components (qw recovered via unit constraint)
"""

import asyncio
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Callable
from elrs import ELRS
from elrs.elrs import RC_CHANNEL_MAX, RC_CHANNEL_MIN
# betaflight source:
# /* conversion from RC value to PWM
# * for 0x16 RC frame
# *       RC     PWM
# * min  172 ->  988us
# * mid  992 -> 1500us
# * max 1811 -> 2012us
# * scale factor = (2012-988) / (1811-172) = 0.62477120195241
# * offset = 988 - 172 * 0.62477120195241 = 880.53935326418548
# */
# return (channelScale * (float)crsfChannelData[chan]) + 881;
from datetime import datetime

# Optional Vicon import
try:
    from pyvicon_datastream import tools
    VICON_AVAILABLE = True
except ImportError:
    VICON_AVAILABLE = False

import rerun as rr


class RealDeployment:
    """
    Real-world deployment class that replaces L2F simulator for IRL flights.

    Always forwards gamepad channels 0-7 via ELRS.
    Optionally reads Vicon data to populate state channels 7-15 with body frame
    values (setpoint error and velocity) that are direct NN inputs.
    """

    def __init__(
        self,
        elrs_port: str = "/dev/ttyUSB0",
        elrs_baud: int = 921600,
        elrs_rate: int = 100,
        vicon_ip: Optional[str] = None,
        vicon_object_name: Optional[str] = None,
        velocity_filter_alpha: float = 0.3,
        loop_rate_hz: int = 100,
    ):
        """
        Initialize the deployment system.

        Args:
            elrs_port: Serial port for ELRS module (e.g., "/dev/ttyUSB0")
            elrs_baud: Baud rate for ELRS (default 921600)
            elrs_rate: ELRS telemetry polling rate in Hz
            vicon_ip: Vicon DataStream server IP (None = gamepad-only mode)
            vicon_object_name: Name of tracked object in Vicon
            velocity_filter_alpha: Low-pass filter coefficient for velocity (0-1)
            loop_rate_hz: Main loop rate in Hz
        """
        self.elrs_port = elrs_port
        self.elrs_baud = elrs_baud
        self.elrs_rate = elrs_rate
        self.vicon_ip = vicon_ip
        self.vicon_object_name = vicon_object_name
        self.velocity_filter_alpha = velocity_filter_alpha
        self.loop_rate_hz = loop_rate_hz

        # Joystick state (8 channels, PWM 1000-2000)
        self.joystick_values = [1500] * 8

        # ELRS transmitter
        self.elrs: Optional[ELRS] = None

        # Vicon tracker
        self.vicon_tracker = None
        self.vicon_connected = False

        # State estimation
        self._prev_position: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._velocity: np.ndarray = np.zeros(3)
        self._position: np.ndarray = np.zeros(3)
        self._quaternion_xyz: np.ndarray = np.zeros(3)

        # Body frame values (computed from world frame)
        self._body_setpoint_error: np.ndarray = np.zeros(3)
        self._body_velocity: np.ndarray = np.zeros(3)

        # Status flags
        self._running = False
        self._vicon_data_valid = False
        self._last_reconnect_attempt = 0.0

        # Rerun logging buffer (for send_columns batching)
        self._log_buffer_size = 50  # Flush every 50 samples (0.5s at 100Hz)
        self._log_buffer: dict = {}
        self._reset_log_buffer()

    def _reset_log_buffer(self) -> None:
        """Reset the logging buffer for send_columns batching."""
        self._log_buffer = {
            'time': [],
            'joystick': [],
            'rc_state': [],
            'world_position': [],
            'world_velocity': [],
            'world_quaternion_xyz': [],
            'body_setpoint_error': [],
            'body_velocity': [],
            'vicon_valid': [],
        }

    def set_joystick_channels(self, joystick_values: list) -> None:
        """
        Callback interface matching L2F simulator.

        Args:
            joystick_values: List of 8 PWM values (1000-2000) for channels 0-7
        """
        self.joystick_values = list(joystick_values[:8])
        # Pad to 8 channels if needed
        while len(self.joystick_values) < 8:
            self.joystick_values.append(1500)

    def _init_vicon(self) -> bool:
        """Initialize Vicon connection."""
        if not VICON_AVAILABLE:
            print("pyvicon_datastream not installed, running in gamepad-only mode")
            return False

        if self.vicon_ip is None:
            return False

        try:
            print(f"Attempting vicon connection to {self.vicon_ip}")
            self.vicon_tracker = tools.ObjectTracker(self.vicon_ip)
            self.vicon_connected = True
            print(f"Connected to Vicon at {self.vicon_ip}")
            return True
        except Exception as e:
            print(f"Failed to connect to Vicon: {e}")
            self.vicon_connected = False
            return False

    def _get_vicon_pose(self) -> Optional[tuple]:
        """
        Get pose from Vicon.

        Returns:
            Tuple of (position_meters, rotation_vector) or None if unavailable
        """
        if not self.vicon_connected or self.vicon_tracker is None:
            return None

        try:
            result = self.vicon_tracker.get_position(self.vicon_object_name)
            if result is False or result is None:
                return None

            latency, frame_no, objects = result
            if len(objects) == 0:
                return None

            # Extract first matching object
            # Format: [subject_name, segment_name, x_mm, y_mm, z_mm, euler_x, euler_y, euler_z]
            obj = objects[0]

            position = np.array([obj[2], obj[3], obj[4]])

            # Orientation: convert Euler XYZ (degrees) to quaternion xyz components
            euler_xyz_deg = np.array([obj[5], obj[6], obj[7]])

            # Convert Euler to quaternion using scipy
            rotation = R.from_euler('xyz', euler_xyz_deg, degrees=True)
            quat_xyzw = rotation.as_quat()  # scipy returns (x, y, z, w)

            # Convert to (w, x, y, z) and ensure canonical form (w >= 0)
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            if quat_wxyz[0] < 0:
                quat_wxyz = -quat_wxyz
            quaternion_xyz = quat_wxyz[1:4]  # (x, y, z) - always in [-1, 1]

            return position, quaternion_xyz

        except Exception as e:
            print(f"Vicon read error: {e}")
            self._handle_vicon_disconnect()
            return None

    def _handle_vicon_disconnect(self) -> None:
        """Handle Vicon disconnection gracefully."""
        self.vicon_connected = False
        self._vicon_data_valid = False
        # Decay velocity toward zero
        self._velocity *= 0.95

    def _update_velocity(self, position: np.ndarray, current_time: float) -> np.ndarray:
        """
        Compute velocity from position using filtered numerical differentiation.

        Args:
            position: Current position in meters [x, y, z]
            current_time: Current timestamp in seconds

        Returns:
            Filtered velocity estimate [vx, vy, vz]
        """
        if self._prev_position is None or self._prev_time is None:
            # First measurement - no velocity yet
            self._prev_position = position.copy()
            self._prev_time = current_time
            return np.zeros(3)

        dt = current_time - self._prev_time

        # Avoid division by very small dt (< 1ms)
        if dt < 0.001:
            return self._velocity

        # Raw velocity from finite difference
        raw_velocity = (position - self._prev_position) / dt

        # Exponential moving average filter
        alpha = self.velocity_filter_alpha
        self._velocity = alpha * raw_velocity + (1 - alpha) * self._velocity

        # Update history
        self._prev_position = position.copy()
        self._prev_time = current_time

        return self._velocity

    @staticmethod
    def _rescale_to_crsf(value: float) -> int:
        """
        Rescale value from [-1, 1] to CRSF RC range [172, 1811].

        CRSF channel values map to PWM as follows:
            172 -> 988us, 992 -> 1500us, 1811 -> 2012us
        """
        # Linear mapping: [-1, 1] -> [172, 1811]
        # -1 -> 172, 0 -> 991.5, 1 -> 1811
        crsf_range = RC_CHANNEL_MAX - RC_CHANNEL_MIN  # 1811 - 172 = 1639
        scaled = RC_CHANNEL_MIN + (value + 1) * crsf_range / 2
        return max(RC_CHANNEL_MIN, min(RC_CHANNEL_MAX, round(scaled)))

    @staticmethod
    def _pwm_to_crsf(pwm: int) -> int:
        """
        Convert PWM (1000-2000) to CRSF RC range [172, 1811].

        PWM 1000 -> 172, PWM 1500 -> ~992, PWM 2000 -> 1811
        """
        # Linear mapping: [1000, 2000] -> [172, 1811]
        crsf_range = RC_CHANNEL_MAX - RC_CHANNEL_MIN  # 1639
        scaled = RC_CHANNEL_MIN + (pwm - 1000) * crsf_range / 1000
        return max(RC_CHANNEL_MIN, min(RC_CHANNEL_MAX, round(scaled)))

    def _build_rc_channels(self) -> list:
        """
        Build 16 RC channels for ELRS transmission.

        Channels 0-6:  Gamepad input (PWM converted to CRSF range)
        Channels 7-9:  Body frame setpoint error (direct NN input)
        Channels 10-12: Body frame velocity (direct NN input)
        Channels 13-15: Quaternion xyz (qx, qy, qz) scaled to CRSF range
        """
        channels = [0] * 16

        # Channels 0-6: Gamepad input (convert PWM 1000-2000 to CRSF 172-1811)
        for i in range(7):
            pwm = self.joystick_values[i] if i < len(self.joystick_values) else 1500
            channels[i] = self._pwm_to_crsf(pwm)

        if self._vicon_data_valid:
            # Reconstruct quaternion from xyz (qw = sqrt(1 - x^2 - y^2 - z^2))
            qxyz = self._quaternion_xyz
            w_squared = 1.0 - np.sum(qxyz**2)
            qw = np.sqrt(max(0, w_squared))
            quat_xyzw = np.array([qxyz[0], qxyz[1], qxyz[2], qw])  # scipy format
            rotation = R.from_quat(quat_xyzw)

            # Compute body frame setpoint error: R^T * (target - position)
            # Target is [0, 0, 1] (hover at 1m height)
            target_position = np.array([0.0, 0.0, 1.0])
            position_error_world = target_position - self._position
            body_setpoint_error = rotation.inv().apply(position_error_world)

            # Compute body frame velocity: R^T * world_velocity
            body_velocity = rotation.inv().apply(self._velocity)

            # Channels 7-9: Body frame setpoint error (direct NN input)
            channels[7] = self._rescale_to_crsf(np.clip(body_setpoint_error[0], -1, 1))
            channels[8] = self._rescale_to_crsf(np.clip(body_setpoint_error[1], -1, 1))
            channels[9] = self._rescale_to_crsf(np.clip(body_setpoint_error[2], -1, 1))

            # Channels 10-12: Body frame velocity (direct NN input)
            channels[10] = self._rescale_to_crsf(np.clip(body_velocity[0], -1, 1))
            channels[11] = self._rescale_to_crsf(np.clip(body_velocity[1], -1, 1))
            channels[12] = self._rescale_to_crsf(np.clip(body_velocity[2], -1, 1))

            # Channels 13-15: Quaternion xyz (always in [-1, 1] for unit quaternions)
            channels[13] = self._rescale_to_crsf(self._quaternion_xyz[0])
            channels[14] = self._rescale_to_crsf(self._quaternion_xyz[1])
            channels[15] = self._rescale_to_crsf(self._quaternion_xyz[2])

            # Store body frame values for logging
            self._body_setpoint_error = body_setpoint_error
            self._body_velocity = body_velocity
        else:
            # No valid Vicon data - send neutral values (zero error/velocity/rotation)
            neutral = self._rescale_to_crsf(0.0)
            for i in range(7, 16):
                channels[i] = neutral
            self._body_setpoint_error = np.zeros(3)
            self._body_velocity = np.zeros(3)

        return channels

    def _log_to_rerun(self, channels: list, current_time: float) -> None:
        """Buffer data for Rerun visualization, flush when buffer is full."""
        # Append to buffers
        self._log_buffer['time'].append(current_time)
        self._log_buffer['joystick'].append([float(v) for v in self.joystick_values])
        self._log_buffer['rc_state'].append([float(channels[i]) for i in range(7, 16)])
        self._log_buffer['world_position'].append(self._position.tolist())
        self._log_buffer['world_velocity'].append(self._velocity.tolist())
        self._log_buffer['world_quaternion_xyz'].append(self._quaternion_xyz.tolist())
        self._log_buffer['body_setpoint_error'].append(self._body_setpoint_error.tolist())
        self._log_buffer['body_velocity'].append(self._body_velocity.tolist())
        self._log_buffer['vicon_valid'].append(float(self._vicon_data_valid))

        # Flush when buffer is full
        if len(self._log_buffer['time']) >= self._log_buffer_size:
            self._flush_log_buffer()

    def _flush_log_buffer(self) -> None:
        """Flush buffered data to Rerun using send_columns for efficiency."""
        if not self._log_buffer['time']:
            return

        times = np.array(self._log_buffer['time'])

        # Joystick channels (8 values: roll, pitch, throttle, yaw, arm, mode, aux3, aux4)
        joystick = np.array(self._log_buffer['joystick'])
        rr.send_columns(
            "rc/joystick",
            indexes=[rr.TimeColumn("time", timestamp=times)],
            columns=rr.Scalars.columns(scalars=joystick),
        )

        # RC state channels (9 values: body_setpoint_error xyz, body_velocity xyz, rotation xyz)
        rc_state = np.array(self._log_buffer['rc_state'])
        rr.send_columns(
            "rc/state",
            indexes=[rr.TimeColumn("time", timestamp=times)],
            columns=rr.Scalars.columns(scalars=rc_state),
        )

        # World frame position
        world_pos = np.array(self._log_buffer['world_position'])
        rr.send_columns(
            "world/position",
            indexes=[rr.TimeColumn("time", timestamp=times)],
            columns=rr.Scalars.columns(scalars=world_pos),
        )

        # World frame velocity
        world_vel = np.array(self._log_buffer['world_velocity'])
        rr.send_columns(
            "world/velocity",
            indexes=[rr.TimeColumn("time", timestamp=times)],
            columns=rr.Scalars.columns(scalars=world_vel),
        )

        # World frame quaternion xyz
        world_quat = np.array(self._log_buffer['world_quaternion_xyz'])
        rr.send_columns(
            "world/quaternion_xyz",
            indexes=[rr.TimeColumn("time", timestamp=times)],
            columns=rr.Scalars.columns(scalars=world_quat),
        )

        # Body frame setpoint error
        body_err = np.array(self._log_buffer['body_setpoint_error'])
        rr.send_columns(
            "body/setpoint_error",
            indexes=[rr.TimeColumn("time", timestamp=times)],
            columns=rr.Scalars.columns(scalars=body_err),
        )

        # Body frame velocity
        body_vel = np.array(self._log_buffer['body_velocity'])
        rr.send_columns(
            "body/velocity",
            indexes=[rr.TimeColumn("time", timestamp=times)],
            columns=rr.Scalars.columns(scalars=body_vel),
        )

        # Vicon valid status
        vicon_valid = np.array(self._log_buffer['vicon_valid'])
        rr.send_columns(
            "status/vicon_valid",
            indexes=[rr.TimeColumn("time", timestamp=times)],
            columns=rr.Scalars.columns(scalars=vicon_valid),
        )

        self._reset_log_buffer()

    def _telemetry_callback(self, ftype: int, decoded) -> None:
        """Handle ELRS telemetry data."""
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{ts}] ELRS telemetry: {ftype:02X} {decoded}")

    def _try_vicon_reconnect(self) -> None:
        """Attempt to reconnect to Vicon."""
        if time.time() - self._last_reconnect_attempt > 5.0:  # Try every 5 seconds
            self._last_reconnect_attempt = time.time()
            print("Attempting Vicon reconnection...")
            self._init_vicon()

    async def run(self) -> None:
        """
        Async main loop matching L2F interface.

        Continuously:
        1. Reads Vicon data (if configured)
        2. Computes velocity from position
        3. Encodes state to RC channels
        4. Sends all 16 channels via ELRS
        """
        self._running = True
        loop_period = 1.0 / self.loop_rate_hz

        # Initialize ELRS
        self.elrs = ELRS(
            self.elrs_port,
            baud=self.elrs_baud,
            rate=self.elrs_rate,
            telemetry_callback=self._telemetry_callback,
        )

        # Start ELRS background task
        elrs_task = asyncio.create_task(self.elrs.start())

        # Initialize Vicon (optional)
        vicon_enabled = self._init_vicon()

        # Initialize Rerun
        rr.init("RealDeployment", spawn=True)

        print(f"RealDeployment started:")
        print(f"  ELRS: {self.elrs_port} @ {self.elrs_baud} baud")
        print(f"  Vicon: {'enabled' if vicon_enabled else 'disabled (gamepad-only mode)'}")
        if vicon_enabled:
            print(f"  Vicon object: {self.vicon_object_name}")
        print(f"  Loop rate: {self.loop_rate_hz} Hz")

        try:
            while self._running:
                loop_start = time.time()

                # 1. Read Vicon data (if enabled)
                if vicon_enabled or self.vicon_ip is not None:
                    pose = self._get_vicon_pose()
                    if pose is not None:
                        position, quaternion_xyz = pose
                        self._position = position
                        self._quaternion_xyz = quaternion_xyz
                        self._update_velocity(position, loop_start)
                        self._vicon_data_valid = True
                    else:
                        # Vicon data unavailable this frame
                        self._vicon_data_valid = False
                        # Try to reconnect periodically
                        if not self.vicon_connected and self.vicon_ip is not None:
                            self._try_vicon_reconnect()

                # 2. Build RC channels
                channels = self._build_rc_channels()

                # 3. Log to Rerun (buffered)
                self._log_to_rerun(channels, loop_start)

                # 4. Send to ELRS
                self.elrs.set_channels(channels)

                # 5. Sleep to maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, loop_period - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            print("RealDeployment loop cancelled")
        finally:
            # Flush any remaining buffered log data
            self._flush_log_buffer()
            self._running = False
            if self.elrs:
                self.elrs.stop()
            elrs_task.cancel()
            try:
                await elrs_task
            except asyncio.CancelledError:
                pass

    def stop(self) -> None:
        """Stop the deployment loop."""
        self._running = False
