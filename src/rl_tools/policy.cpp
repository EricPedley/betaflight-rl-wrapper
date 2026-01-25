#include <stdio.h>
#include <limits>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
extern "C" {
    #include "rx/rx.h"
#ifdef USE_CLI_DEBUG_PRINT
    #include "cli/cli_debug_print.h"
#endif
    #include "flight/mixer.h"
    #include "flight/imu.h"
    #include "sensors/gyro.h"
    #include "sensors/acceleration.h"
    #include "flight/imu.h"
    #include "drivers/time.h"
#if defined(RL_TOOLS_BETAFLIGHT_TARGET_BETAFPVG473) || defined(RL_TOOLS_BETAFLIGHT_TARGET_PAVO20)
    #include "config.h"
#endif
#if defined(RL_TOOLS_BETAFLIGHT_TARGET_SAVAGEBEE_PUSHER) || defined(RL_TOOLS_BETAFLIGHT_TARGET_BETAFPVG473) || defined(RL_TOOLS_BETAFLIGHT_TARGET_PAVO20)
#define OVERWRITE_DEFAULT_LED_WITH_POSITION_FEEDBACK
#endif
#ifdef OVERWRITE_DEFAULT_LED_WITH_POSITION_FEEDBACK
    #include "drivers/light_led.h"
    #include "drivers/sound_beeper.h"
#endif
    #include "neural_network.h"
    #undef RNG
}
#pragma GCC diagnostic pop

using T = float;
using TI = uint32_t;
static constexpr int PRINTF_FACTOR = 1000;

uint64_t previous_micros = 0;
bool previous_micros_set = false;
uint32_t micros_overflow_counter = 0;

uint64_t previous_rl_tools_tick = 0;
bool previous_rl_tools_tick_set = false;
uint32_t rl_tools_tick = 0;


bool first_run = true;
bool active = true;
TI activation_tick = 0;
T acceleration_integral[3] = {0, 0, 0};
#if defined(RL_TOOLS_BETAFLIGHT_TARGET_PAVO20) || defined(RL_TOOLS_BETAFLIGHT_TARGET_SAVAGEBEE_PUSHER)
constexpr T ACCELERATION_INTEGRAL_TIMECONSTANT = 0.035;
#else
constexpr T ACCELERATION_INTEGRAL_TIMECONSTANT = 0.025;
#endif
constexpr bool USE_ACCELERATION_INTEGRAL_FEEDFORWARD_TERM = false;

#ifndef USE_CLI_DEBUG_PRINT
void cliPrintLinef(const char *format, ...){/*noop*/}
#endif

template <typename T>
T clip(T x, T min, T max){
	if(x > max){
		return max;
	}
	if(x < min){
		return min;
	}
	return x;
}
template <typename T, int ELEMENT>
T quaternion_to_rotation_matrix(T q[4]){
	// row-major
	T qw = q[0];
	T qx = q[1];
	T qy = q[2];
	T qz = q[3];

	static_assert(ELEMENT >= 0 && ELEMENT < 9);
	if constexpr(ELEMENT == 0){
		return 1 - 2*qy*qy - 2*qz*qz;
	}
	if constexpr(ELEMENT == 1){
		return     2*qx*qy - 2*qw*qz;
	}
	if constexpr(ELEMENT == 2){
		return     2*qx*qz + 2*qw*qy;
	}
	if constexpr(ELEMENT == 3){
		return     2*qx*qy + 2*qw*qz;
	}
	if constexpr(ELEMENT == 4){
		return 1 - 2*qx*qx - 2*qz*qz;
	}
	if constexpr(ELEMENT == 5){
		return     2*qy*qz - 2*qw*qx;
	}
	if constexpr(ELEMENT == 6){
		return     2*qx*qz - 2*qw*qy;
	}
	if constexpr(ELEMENT == 7){
		return     2*qy*qz + 2*qw*qx;
	}
	if constexpr(ELEMENT == 8){
		return 1 - 2*qx*qx - 2*qy*qy;
	}
	return 0;
}

template <typename T>
void quaternion_to_rotation_matrix(T q[4], T R[9]){
	R[0] = quaternion_to_rotation_matrix<T, 0>(q);
	R[1] = quaternion_to_rotation_matrix<T, 1>(q);
	R[2] = quaternion_to_rotation_matrix<T, 2>(q);
	R[3] = quaternion_to_rotation_matrix<T, 3>(q);
	R[4] = quaternion_to_rotation_matrix<T, 4>(q);
	R[5] = quaternion_to_rotation_matrix<T, 5>(q);
	R[6] = quaternion_to_rotation_matrix<T, 6>(q);
	R[7] = quaternion_to_rotation_matrix<T, 7>(q);
	R[8] = quaternion_to_rotation_matrix<T, 8>(q);
}

template <typename T>
void rotate_vector(T R[9], T v[3], T v_rotated[3]){
	v_rotated[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
	v_rotated[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
	v_rotated[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

template <typename T>
void rotation_vector_to_quaternion(T rv[3], T q[4]){
	T angle = sqrtf(rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2]);
	if(angle < 1e-6f){
		// Small angle approximation - identity quaternion
		q[0] = 1;
		q[1] = 0;
		q[2] = 0;
		q[3] = 0;
	} else {
		T half_angle = angle / 2;
		T s = sinf(half_angle) / angle;
		q[0] = cosf(half_angle);  // w
		q[1] = rv[0] * s;          // x
		q[2] = rv[1] * s;          // y
		q[3] = rv[2] * s;          // z
	}
}

T from_channel(T value){
    static_assert(PWM_RANGE_MIN == 1000, "PWM_RANGE_MIN must be 1000");
    static_assert(PWM_RANGE_MAX == 2000, "PWM_RANGE_MAX must be 2000");
    return (value - PWM_RANGE_MIN) / (T)(PWM_RANGE_MAX - PWM_RANGE_MIN) * 2 - 1;
}

// constants
static constexpr T MOTOR_FACTOR = 1.0f;
static constexpr int NN_INPUT_DIM = 12;
static constexpr int NN_OUTPUT_DIM = 4;

// setpoint (target position in world frame)
static T target_position[3] = {0, 0, 1};

// input (from simulator via RC channels)
T position[3] = {0, 0, 0};
T linear_velocity[3] = {0, 0, 0};

T rl_tools_rpms[4] = {0, 0, 0, 0};


// Rotate vector using transpose of rotation matrix (world to body)
template <typename T>
void rotate_vector_transpose(T R[9], T v[3], T v_rotated[3]){
    // R^T * v (transpose multiplication)
    v_rotated[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
    v_rotated[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
    v_rotated[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
}

void reset(){
    acceleration_integral[0] = 0;
    acceleration_integral[1] = 0;
    acceleration_integral[2] = 0;
}


extern "C" void rl_tools_status(void){
    cliPrintLinef("Neural Network Policy Active");
    cliPrintLinef("Input dim: %d, Output dim: %d", NN_INPUT_DIM, NN_OUTPUT_DIM);
    cliPrintLinef("MOTOR_FACTOR: %d / %d", (int)(MOTOR_FACTOR*PRINTF_FACTOR), PRINTF_FACTOR);
}

bool prevArmed=false;

extern "C" void rl_tools_control(bool armed){
    if(armed && !prevArmed) {
        reset();
    }
    prevArmed = armed;
    if(first_run){
        first_run = false;
        cliPrintLinef("Neural Network Policy Control started");
    }
    timeUs_t now_narrow = micros();
    previous_micros = now_narrow;
    previous_micros_set = true;
    uint64_t now = micros_overflow_counter * std::numeric_limits<timeUs_t>::max() + now_narrow;

    bool tick_now = false;
    if(previous_rl_tools_tick_set){
        if((now - previous_rl_tools_tick >= 1000)){
            rl_tools_tick++;
            previous_rl_tools_tick = now;
            tick_now = true;
        }
    }
    else{
        rl_tools_tick = 0;
        previous_rl_tools_tick_set = true;
        previous_rl_tools_tick = now;
        tick_now = true;
    }

    // Read position from RC channels (world frame)
    position[0] = from_channel(rcData[7]);
    position[1] = from_channel(rcData[8]);
    position[2] = from_channel(rcData[9]);

    // Read orientation as rotation vector from RC channels and convert to quaternion
    #ifdef RL_TOOLS_BETAFLIGHT_VERSION_4_5
    quaternion q;
    #else
    quaternion_t q;
    #endif

    T rv[3];
    rv[0] = from_channel(rcData[13]);
    rv[1] = from_channel(rcData[14]);
    rv[2] = from_channel(rcData[15]);

    T qr[4];
    rotation_vector_to_quaternion(rv, qr);
    q.w = qr[0];
    q.x = qr[1];
    q.y = qr[2];
    q.z = qr[3];
    imuSetAttitudeQuat(q.w, q.x, q.y, q.z);

    // Read linear velocity from RC channels (world frame)
    linear_velocity[0] = from_channel(rcData[10]);
    linear_velocity[1] = from_channel(rcData[11]);
    linear_velocity[2] = from_channel(rcData[12]);

    // Build rotation matrix from quaternion (body to world)
    T q_vec[4], R[9];
    q_vec[0] = q.w;
    q_vec[1] = q.x;
    q_vec[2] = q.y;
    q_vec[3] = q.z;
    quaternion_to_rotation_matrix(q_vec, R);

    // Build neural network input vector (12 elements):
    // [body_linear_velocity(3), body_angular_velocity(3), body_projected_gravity(3), body_position_setpoint(3)]
    float nn_input[NN_INPUT_DIM];
    float nn_output[NN_OUTPUT_DIM];

    // 1. Body frame linear velocity: R^T * world_velocity
    T body_linear_velocity[3];
    rotate_vector_transpose(R, linear_velocity, body_linear_velocity);
    nn_input[0] = body_linear_velocity[0];
    nn_input[1] = body_linear_velocity[1];
    nn_input[2] = body_linear_velocity[2];

    // 2. Body frame angular velocity: from gyro (already in body frame)
    constexpr float GYRO_CONVERSION_FACTOR = (T)M_PI / 180.0f;
    #if defined(RL_TOOLS_BETAFLIGHT_TARGET_BETAFPVG473) || defined(RL_TOOLS_BETAFLIGHT_TARGET_SAVAGEBEE_PUSHER)
    nn_input[3] = gyro.gyroADC[0] * GYRO_CONVERSION_FACTOR;
    nn_input[4] = gyro.gyroADC[1] * GYRO_CONVERSION_FACTOR;
    nn_input[5] = gyro.gyroADC[2] * GYRO_CONVERSION_FACTOR;
    #else
    nn_input[3] = gyro.gyroADCf[0] * GYRO_CONVERSION_FACTOR;
    nn_input[4] = gyro.gyroADCf[1] * GYRO_CONVERSION_FACTOR;
    nn_input[5] = gyro.gyroADCf[2] * GYRO_CONVERSION_FACTOR;
    #endif

    T gravity_world[3] = {0, 0, -1};
    T body_gravity[3];
    rotate_vector_transpose(R, gravity_world, body_gravity);
    nn_input[6] = body_gravity[0];
    nn_input[7] = body_gravity[1];
    nn_input[8] = body_gravity[2];

    // 4. Body frame position setpoint: R^T * (target_position - current_position)
    T position_error_world[3];
    position_error_world[0] = target_position[0] - position[0];
    position_error_world[1] = target_position[1] - position[1];
    position_error_world[2] = target_position[2] - position[2];
    T body_position_setpoint[3];
    rotate_vector_transpose(R, position_error_world, body_position_setpoint);
    nn_input[9] = body_position_setpoint[0];
    nn_input[10] = body_position_setpoint[1];
    nn_input[11] = body_position_setpoint[2];

    #ifdef OVERWRITE_DEFAULT_LED_WITH_POSITION_FEEDBACK
    #if defined(RL_TOOLS_BETAFLIGHT_TARGET_PAVO20)
    ledSet(0, fabsf(body_position_setpoint[0]) > 0.1f);
    #else
    ledSet(1, fabsf(body_position_setpoint[0]) > 0.1f);
    #endif
    #endif

    // Debug logging
    if(tick_now && rl_tools_tick % 1000 == 0){
        cliPrintLinef("NN Input: vx%d vy%d vz%d wx%d wy%d wz%d gx%d gy%d gz%d px%d py%d pz%d",
            (int)(nn_input[0]*PRINTF_FACTOR),
            (int)(nn_input[1]*PRINTF_FACTOR),
            (int)(nn_input[2]*PRINTF_FACTOR),
            (int)(nn_input[3]*PRINTF_FACTOR),
            (int)(nn_input[4]*PRINTF_FACTOR),
            (int)(nn_input[5]*PRINTF_FACTOR),
            (int)(nn_input[6]*PRINTF_FACTOR),
            (int)(nn_input[7]*PRINTF_FACTOR),
            (int)(nn_input[8]*PRINTF_FACTOR),
            (int)(nn_input[9]*PRINTF_FACTOR),
            (int)(nn_input[10]*PRINTF_FACTOR),
            (int)(nn_input[11]*PRINTF_FACTOR)
        );
    }

    // Run neural network inference
    timeUs_t pre_inference = micros();
    nn_forward(nn_input, nn_output);

    // Apply actions to motors
    // uint8_t target_indices[4] = {1, 0, 2, 3}; // remapping from Crazyflie to Betaflight motor indices
    uint8_t target_indices[4] = {1, 0, 3, 2}; // remapping that works for sim2sim transfer. Not sure why these are not the identity, must've screwed up indexing somewhere in the sysid/training pipeline
    for(TI action_i = 0; action_i < NN_OUTPUT_DIM; action_i++){
        if(active){
            T clipped_action = clip(nn_output[action_i], (T)-1, (T)1);
            clipped_action = (clipped_action * 0.5f + 0.5f); // [0, 1]
            motor[target_indices[action_i]] = 1000 + clipped_action * 1000;
            // motor[target_indices[action_i]] = (int)((0.5+action_i/8.0f) * 1000);
        }
        else{
            #if defined(RL_TOOLS_BETAFLIGHT_TARGET_BETAFPVG473) || defined(RL_TOOLS_BETAFLIGHT_TARGET_PAVO20)
            motor[target_indices[action_i]] = 0; // stop the motors
            #endif
        }
    }
}