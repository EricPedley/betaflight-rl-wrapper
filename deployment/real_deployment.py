"""
Real-world deployment class that replaces L2F simulator.

Provides same interface as L2F:
- set_joystick_channels(joystick_values): Callback for gamepad input
- run(): Async main loop

Forwards gamepad RC channels (0-7) to ELRS transmitter.
Optionally populates state channels (7-15) from Vicon motion capture:
- Channels 7-9: Body frame setpoint error (direct NN input)
- Channels 10-12: Body frame velocity (direct NN input)
- Channels 13-15: Rotation vector (for quaternion reconstruction)
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
        self._rotation_vector: np.ndarray = np.zeros(3)

        # Body frame values (computed from world frame)
        self._body_setpoint_error: np.ndarray = np.zeros(3)
        self._body_velocity: np.ndarray = np.zeros(3)

        # Status flags
        self._running = False
        self._vicon_data_valid = False
        self._last_reconnect_attempt = 0.0

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

            # Orientation: convert Euler XYZ (degrees) to rotation vector
            euler_xyz_deg = np.array([obj[5], obj[6], obj[7]])

            # Convert Euler to rotation object using scipy
            rotation = R.from_euler('xyz', euler_xyz_deg, degrees=True)
            rotation_vector = rotation.as_rotvec()

            return position, rotation_vector

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
        Channels 13-15: Rotation vector (rx, ry, rz) scaled to CRSF range
        """
        channels = [0] * 16

        # Channels 0-6: Gamepad input (convert PWM 1000-2000 to CRSF 172-1811)
        for i in range(7):
            pwm = self.joystick_values[i] if i < len(self.joystick_values) else 1500
            channels[i] = self._pwm_to_crsf(pwm)

        if self._vicon_data_valid:
            # Get rotation from rotation vector
            rotation = R.from_rotvec(self._rotation_vector)

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

            # Channels 13-15: Rotation vector (radians, clamped to [-1, 1])
            channels[13] = self._rescale_to_crsf(np.clip(self._rotation_vector[0], -1, 1))
            channels[14] = self._rescale_to_crsf(np.clip(self._rotation_vector[1], -1, 1))
            channels[15] = self._rescale_to_crsf(np.clip(self._rotation_vector[2], -1, 1))

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

    def _log_to_rerun(self, channels: list) -> None:
        """Log data to Rerun for visualization."""
        # Log joystick values (PWM 1000-2000)
        rr.log("rc/joystick/roll", rr.Scalars(float(self.joystick_values[0])))
        rr.log("rc/joystick/pitch", rr.Scalars(float(self.joystick_values[1])))
        rr.log("rc/joystick/throttle", rr.Scalars(float(self.joystick_values[2])))
        rr.log("rc/joystick/yaw", rr.Scalars(float(self.joystick_values[3])))
        rr.log("rc/joystick/arm", rr.Scalars(float(self.joystick_values[4])))
        rr.log("rc/joystick/mode", rr.Scalars(float(self.joystick_values[5])))
        rr.log("rc/joystick/aux3", rr.Scalars(float(self.joystick_values[6])))
        rr.log("rc/joystick/aux4", rr.Scalars(float(self.joystick_values[7])))

        # Log encoded RC state channels (11-bit values) with body frame semantics
        rr.log("rc/state/body_setpoint_error_x", rr.Scalars(float(channels[7])))
        rr.log("rc/state/body_setpoint_error_y", rr.Scalars(float(channels[8])))
        rr.log("rc/state/body_setpoint_error_z", rr.Scalars(float(channels[9])))
        rr.log("rc/state/body_velocity_x", rr.Scalars(float(channels[10])))
        rr.log("rc/state/body_velocity_y", rr.Scalars(float(channels[11])))
        rr.log("rc/state/body_velocity_z", rr.Scalars(float(channels[12])))
        rr.log("rc/state/rotation_x", rr.Scalars(float(channels[13])))
        rr.log("rc/state/rotation_y", rr.Scalars(float(channels[14])))
        rr.log("rc/state/rotation_z", rr.Scalars(float(channels[15])))

        # Log world frame values for reference
        rr.log("world/position", rr.Scalars([
            float(self._position[0]),
            float(self._position[1]),
            float(self._position[2])
        ]))
        rr.log("world/velocity", rr.Scalars([
            float(self._velocity[0]),
            float(self._velocity[1]),
            float(self._velocity[2])
        ]))
        rr.log("world/rotation_vector", rr.Scalars([
            float(self._rotation_vector[0]),
            float(self._rotation_vector[1]),
            float(self._rotation_vector[2])
        ]))

        # Log body frame values (direct NN inputs)
        rr.log("body/setpoint_error", rr.Scalars([
            float(self._body_setpoint_error[0]),
            float(self._body_setpoint_error[1]),
            float(self._body_setpoint_error[2])
        ]))
        rr.log("body/velocity", rr.Scalars([
            float(self._body_velocity[0]),
            float(self._body_velocity[1]),
            float(self._body_velocity[2])
        ]))

        # Log Vicon status
        rr.log("status/vicon_valid", rr.Scalars(float(self._vicon_data_valid)))

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
                        position, rotation_vector = pose
                        self._position = position
                        self._rotation_vector = rotation_vector
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

                # 3. Log to Rerun
                self._log_to_rerun(channels)

                # 4. Send to ELRS
                self.elrs.set_channels(channels)

                # 5. Sleep to maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, loop_period - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            print("RealDeployment loop cancelled")
        finally:
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
