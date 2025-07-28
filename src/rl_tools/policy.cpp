#include <stdio.h>



#include <rl_tools/operations/arm.h>

#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/inference/executor/executor.h>

#include "blob/policy.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wregister"
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
	#include "config.h"
#if defined(RL_TOOLS_BETAFLIGHT_TARGET_SAVAGEBEE_PUSHER) || defined(RL_TOOLS_BETAFLIGHT_TARGET_BETAFPVG473) || defined(RL_TOOLS_BETAFLIGHT_TARGET_PAVO20)
#define OVERWRITE_DEFAULT_LED_WITH_POSITION_FEEDBACK
#endif
#ifdef OVERWRITE_DEFAULT_LED_WITH_POSITION_FEEDBACK
	#include "drivers/light_led.h"
	#include "drivers/sound_beeper.h"
#endif
	#undef RNG
}
#pragma GCC diagnostic pop

namespace rlt = rl_tools;

namespace other{
    using DEV_SPEC = rlt::devices::DefaultARMSpecification;
    using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
}

struct RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG{
    using DEVICE = other::DEVICE;
    using TI = typename other::DEVICE::index_t;
    using RNG = other::DEVICE::SPEC::RANDOM::ENGINE<>;
    static constexpr TI TEST_SEQUENCE_LENGTH_ACTUAL = 5;
    static constexpr TI TEST_BATCH_SIZE_ACTUAL = 2;
    using ACTOR_TYPE_ORIGINAL = rlt::checkpoint::actor::TYPE;
    using POLICY = rlt::checkpoint::actor::TYPE::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_SEQUENCE_LENGTH<TI, 1>;
	using POLICY_TEST = POLICY;
    using T = typename POLICY::SPEC::T;
    static auto& policy() {
        return rlt::checkpoint::actor::module;
    }
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
	#ifdef RL_TOOLS_BETAFLIGHT_TARGET_BETAFPVG473
    static constexpr TI CONTROL_INTERVAL_INTERMEDIATE_NS = 0.4 * 1000 * 1000; // Adjust based on the Control invocation dt statistics
	#else
    static constexpr TI CONTROL_INTERVAL_INTERMEDIATE_NS = 0.5 * 1000 * 1000; // Adjust based on the Control invocation dt statistics
	#endif
    static constexpr TI CONTROL_INTERVAL_NATIVE_NS = 10 * 1000 * 1000; // Training is 100hz
    static constexpr TI TIMING_STATS_NUM_STEPS = 100;
    static constexpr bool FORCE_SYNC_INTERMEDIATE = true;
    static constexpr TI FORCE_SYNC_NATIVE = 0;
    static constexpr bool DYNAMIC_ALLOCATION = false;
    using WARNING_LEVELS = rlt::inference::executor::WarningLevelsDefault<T>;
};

// #define RL_TOOLS_DISABLE_TEST
#include <rl_tools/inference/applications/l2f/c_backend.h>

static_assert(sizeof(rl_tools::inference::applications::l2f::executor.executor.policy_buffer) < 10000);


using T = RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG::T;
using TI = RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG::TI;
static constexpr int PRINTF_FACTOR = 1000;

uint64_t previous_micros = 0;
bool previous_micros_set = false;
uint32_t micros_overflow_counter = 0;

uint64_t previous_rl_tools_tick = 0;
bool previous_rl_tools_tick_set = false;
uint32_t rl_tools_tick = 0;
timeUs_t previous_timing = 0;
bool previous_timing_set = false;


bool first_run = true;
bool active = false;
TI activation_tick = 0;
T acceleration_integral[3] = {0, 0, 0};
constexpr T ACCELERATION_INTEGRAL_TIMECONSTANT = 0.03;
constexpr bool USE_ACCELERATION_INTEGRAL_FEEDFORWARD_TERM = true;
#ifdef RL_TOOLS_BETAFLIGHT_TARGET_SAVAGEBEE_PUSHER
static constexpr T MOTOR_FACTOR = 0.45f;
#elif defined(RL_TOOLS_BETAFLIGHT_TARGET_BETAFPVG473)
static constexpr T MOTOR_FACTOR = 0.4f;
#elif defined(RL_TOOLS_BETAFLIGHT_TARGET_PAVO20)
static constexpr T MOTOR_FACTOR = 0.7f;
#else
// HUMMINGBIRD
static constexpr T MOTOR_FACTOR = 0.5f;
#endif

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
template <typename T>
void quaternion_multiplication(T q1[4], T q2[4], T q_res[4]){
	q_res[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
	q_res[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
	q_res[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
	q_res[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
}
template <typename T>
void quaternion_conjugate(T q[4], T q_res[4]){
	q_res[0] = +q[0];
	q_res[1] = -q[1];
	q_res[2] = -q[2];
	q_res[3] = -q[3];
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
enum class TestObservationMode: TI{
    ANGULAR_VELOCITY = 0,
    ORIENTATION = 1,
    LINEAR_VELOCITY = 2,
    POSITION = 3,
    ACTION_HISTORY = 4,
};

// constants
static constexpr T MAX_POSITION_ERROR = 0.6;
static constexpr T MAX_LINEAR_VELOCITY_ERROR = 2.0;

// setpoint
static T target_position[3] = {0, 0, 0};
static T target_orientation[4] = {1, 0, 0, 0};
static T target_linear_velocity[3] = {0, 0, 0};

// input
T position[3] = {0, 0, 0};
T linear_velocity[3] = {0, 0, 0};

T rl_tools_rpms[4] = {0, 0, 0, 0};

static constexpr TI NUM_EXECUTOR_STATII = 100;
RLtoolsInferenceExecutorStatus intermediate_statii[NUM_EXECUTOR_STATII];
bool intermediate_statii_full = false;
TI intermediate_statii_index = 0;
RLtoolsInferenceExecutorStatus native_statii[NUM_EXECUTOR_STATII];
bool native_statii_full = false;
TI native_statii_index = 0;

T prev_rc_0 = 0;
bool prev_rc_0_set = false;


static constexpr TI NUM_RL_TOOLS_CONTROL_INVOCATION_DTS = 100;
timeUs_t rl_tools_control_invocation_dts[NUM_RL_TOOLS_CONTROL_INVOCATION_DTS];
TI rl_tools_control_invocation_dts_index = 0;
bool rl_tools_control_invocation_dts_full = false;


// state
static T previous_action[4] = {-1, -1, -1, -1};


void observe(RLtoolsInferenceApplicationsL2FObservation& observation, TestObservationMode mode){
	// converting from FRD to FLU
	T qd[4] = {1, 0, 0, 0}, Rt_inv[9];
	if(mode >= TestObservationMode::ORIENTATION){
		// FRD to FLU
		// Validating Julia code:
		// using Rotations
		// FRD2FLU = [1 0 0; 0 -1 0; 0 0 -1]
		// q = rand(UnitQuaternion)
		// q2 = UnitQuaternion(q.q.s, q.q.v1, -q.q.v2, -q.q.v3)
		// diff = q2 - FRD2FLU * q * transpose(FRD2FLU)
		// @assert sum(abs.(diff)) < 1e-10

		T qt[4], qtc[4], qr[4];
		qt[0] = target_orientation[0]; // conjugate to build the difference between setpoint and current
		qt[1] = target_orientation[1];
		qt[2] = target_orientation[2];
		qt[3] = target_orientation[3];
		quaternion_conjugate(qt, qtc);
		quaternion_to_rotation_matrix(qtc, Rt_inv);

		#ifdef RL_TOOLS_BETAFLIGHT_VERSION_4_5
        quaternion q;
		#else
        quaternion_t q;
		#endif
        getQuaternion(&q);

		qr[0] = q.w;
		qr[1] = q.x;
		qr[2] = q.y;
		qr[3] = q.z;
		// qr = qt * qd
		// qd = qt' * qr
		quaternion_multiplication(qtc, qr, qd);

		observation.orientation[0] = qd[0];
		observation.orientation[1] = qd[1];
		observation.orientation[2] = qd[2];
		observation.orientation[3] = qd[3];
	}
	else{
		observation.orientation[0] = 1;
		observation.orientation[1] = 0;
		observation.orientation[2] = 0;
		observation.orientation[3] = 0;
	}
	if(mode >= TestObservationMode::POSITION){
		T p[3], pt[3]; // FLU
		p[0] = (position[0] - target_position[0]);
		p[1] = (position[1] - target_position[1]);
		p[2] = (position[2] - target_position[2]);
		rotate_vector(Rt_inv, p, pt);
		observation.position[0] = clip(pt[0], -MAX_POSITION_ERROR, MAX_POSITION_ERROR);
		observation.position[1] = clip(pt[1], -MAX_POSITION_ERROR, MAX_POSITION_ERROR);
		observation.position[2] = clip(pt[2], -MAX_POSITION_ERROR, MAX_POSITION_ERROR);
	}
	else{
		observation.position[0] = 0;
		observation.position[1] = 0;
		observation.position[2] = 0;
	}
	if(mode >= TestObservationMode::LINEAR_VELOCITY){
		T v[3], vt[3];
		v[0] = (linear_velocity[0] - target_linear_velocity[0]);
		v[1] = (linear_velocity[1] - target_linear_velocity[1]);
		v[2] = (linear_velocity[2] - target_linear_velocity[2]);
		rotate_vector(Rt_inv, v, vt);
		observation.linear_velocity[0] = clip(vt[0], -MAX_LINEAR_VELOCITY_ERROR, MAX_LINEAR_VELOCITY_ERROR);
		observation.linear_velocity[1] = clip(vt[1], -MAX_LINEAR_VELOCITY_ERROR, MAX_LINEAR_VELOCITY_ERROR);
		observation.linear_velocity[2] = clip(vt[2], -MAX_LINEAR_VELOCITY_ERROR, MAX_LINEAR_VELOCITY_ERROR);
	}
	else{
		observation.linear_velocity[0] = 0;
		observation.linear_velocity[1] = 0;
		observation.linear_velocity[2] = 0;
	}
	if(mode >= TestObservationMode::ANGULAR_VELOCITY){
        
        constexpr float GYRO_CONVERSION_FACTOR = (T)M_PI / 180.0f;
		observation.angular_velocity[0] = gyro.gyroADCf[0] * GYRO_CONVERSION_FACTOR;
		observation.angular_velocity[1] = gyro.gyroADCf[1] * GYRO_CONVERSION_FACTOR;
		observation.angular_velocity[2] = gyro.gyroADCf[2] * GYRO_CONVERSION_FACTOR;
		// printf("Gyro: [%.2f %.2f %.2f] vs [%.2f %.2f %.2f]\n", (double)rl_tools_angular_velocity[0], (double)rl_tools_angular_velocity[1], (double)rl_tools_angular_velocity[2], (double)g[0], (double)g[1], (double)g[2]);
	}
	else{
		observation.angular_velocity[0] = 0;
		observation.angular_velocity[1] = 0;
		observation.angular_velocity[2] = 0;
	}
	for(int action_i=0; action_i < RL_TOOLS_INTERFACE_APPLICATIONS_L2F_ACTION_DIM; action_i++){
		observation.previous_action[action_i] = previous_action[action_i];
	}
}
void reset(){
	acceleration_integral[0] = 0;
	acceleration_integral[1] = 0;
	acceleration_integral[2] = 0;
	for(int action_i=0; action_i < RL_TOOLS_INTERFACE_APPLICATIONS_L2F_ACTION_DIM; action_i++){
		previous_action[action_i] = -1;
	}
	rl_tools_inference_applications_l2f_reset();
}


extern "C" void rl_tools_status(void){
	RLtoolsInferenceApplicationsL2FAction action;
	cliPrintLinef("RLtools: checkpoint: %s", rl_tools_inference_applications_l2f_checkpoint_name());
	float abs_diff = rl_tools_inference_applications_l2f_test(&action);
	for(TI output_i = 0; output_i < RL_TOOLS_INTERFACE_APPLICATIONS_L2F_ACTION_DIM; output_i++){
		cliPrintLinef("RLtools: output[%d]: %d / %d", output_i, (int)(action.action[output_i]*PRINTF_FACTOR), PRINTF_FACTOR);
	}
	cliPrintLinef("RLtools: checkpoint test diff: %d / %d", (int)(abs_diff*PRINTF_FACTOR), PRINTF_FACTOR);
	cliPrintLinef("RLtools: MOTOR_FACTOR: %d / %d", (int)(MOTOR_FACTOR*PRINTF_FACTOR), PRINTF_FACTOR);
}

T from_channel(T value){
	static_assert(PWM_RANGE_MIN == 1000, "PWM_RANGE_MIN must be 1000");
	static_assert(PWM_RANGE_MAX == 2000, "PWM_RANGE_MAX must be 2000");
	return (value - PWM_RANGE_MIN) / (T)(PWM_RANGE_MAX - PWM_RANGE_MIN) * 2 - 1;
}

extern "C" void rl_tools_control(bool armed){
    if(first_run){
        first_run = false;
        cliPrintLinef("RLtools Inference Applications L2F Control started");
        rl_tools_inference_applications_l2f_init();
    }
    timeUs_t now_narrow = micros();
	timeUs_t diff = 0;
	bool diff_set = false;
    if(previous_micros_set){
		if(now_narrow < previous_micros){
			micros_overflow_counter++;
		}
		diff = now_narrow - previous_micros;
		diff_set = true;
		rl_tools_control_invocation_dts[rl_tools_control_invocation_dts_index] = diff;
		rl_tools_control_invocation_dts_index++;
		if(rl_tools_control_invocation_dts_index >= NUM_RL_TOOLS_CONTROL_INVOCATION_DTS){
			rl_tools_control_invocation_dts_index = 0;
			rl_tools_control_invocation_dts_full = true;
		}
	} 
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

	position[0] = from_channel(rcData[0]);
	position[1] = from_channel(rcData[1]);
	position[2] = from_channel(rcData[3]); // the AETR channel map seems to be already applied
	T yaw = from_channel(rcData[2]);
	#ifdef RL_TOOLS_BETAFLIGHT_VERSION_4_5
	quaternion q;
	#else
	quaternion_t q;
	#endif
	getQuaternion(&q);
	T d_e = sqrtf(q.w*q.w + q.z*q.z);
	if(d_e >= 1e-6){
		// Only do yaw correction if we are not in the yaw singularity

		T q_estimator_conj[4];
		q_estimator_conj[0] = q.w/d_e;
		q_estimator_conj[1] = 0;
		q_estimator_conj[2] = 0;
		q_estimator_conj[3] = -q.z/d_e;

		T q_external[4];
		T pre = yaw/2.0 * M_PI;
		q_external[0] = cosf(pre);
		q_external[1] = 0;
		q_external[2] = 0;
		q_external[3] = sinf(pre);

		T q_yaw_correction[4];
		quaternion_multiplication(q_external, q_estimator_conj, q_yaw_correction);
		T q_delta[4];
		T alpha = 0.02;
		q_delta[0] = (1.0f - alpha) * 1.0f + alpha * q_yaw_correction[0];
		q_delta[1] = 0;
		q_delta[2] = 0;
		q_delta[3] = (1.0f - alpha) * 0.0f + alpha * q_yaw_correction[3];

		T q_estimator[4];
		q_estimator[0] = q.w;
		q_estimator[1] = q.x;
		q_estimator[2] = q.y;
		q_estimator[3] = q.z;

		T temp[4];
		quaternion_multiplication(q_delta, q_estimator, temp);
		T normalization_factor = sqrtf(temp[0]*temp[0] + temp[1]*temp[1] + temp[2]*temp[2] + temp[3]*temp[3]);
		q.w = temp[0] / normalization_factor;
		q.x = temp[1] / normalization_factor;
		q.y = temp[2] / normalization_factor;
		q.z = temp[3] / normalization_factor;
		imuSetAttitudeQuat(q.w, q.x, q.y, q.z);
		// imuSetAttitudeQuat(q_external[0], q_external[1], q_external[2], q_external[3]);
		// if(tick_now && rl_tools_tick % 1000 == 0){
		// 	cliPrintLinef("q_external %d %d %d %d / %d", (int)(q_external[0]*PRINTF_FACTOR), (int)(q_external[1]*PRINTF_FACTOR), (int)(q_external[2]*PRINTF_FACTOR), (int)(q_external[3]*PRINTF_FACTOR), PRINTF_FACTOR);
		// }
	}

	linear_velocity[0] = from_channel(rcData[5]);
	linear_velocity[1] = from_channel(rcData[6]);
	linear_velocity[2] = from_channel(rcData[7]);

	T acceleration_body[3];
	static constexpr T GRAVITY = 9.81;
	acceleration_body[0] = (acc.accADC[0] / acc.dev.acc_1G) * GRAVITY;
	acceleration_body[1] = (acc.accADC[1] / acc.dev.acc_1G) * GRAVITY;
	acceleration_body[2] = (acc.accADC[2] / acc.dev.acc_1G) * GRAVITY;

	T q_vec[4], R[9];
	q_vec[0] = q.w;
	q_vec[1] = q.x;
	q_vec[2] = q.y;
	q_vec[3] = q.z;
	quaternion_to_rotation_matrix(q_vec, R);

	T acceleration[3];
	rotate_vector(R, acceleration_body, acceleration);

	acceleration[2] -= GRAVITY;

	if(diff_set){
		T dt = diff * 1e-6f;
		T ACCELERATION_INTEGRAL_DECAY = expf(-dt / ACCELERATION_INTEGRAL_TIMECONSTANT);
		acceleration_integral[0] = acceleration_integral[0] * ACCELERATION_INTEGRAL_DECAY + acceleration[0] * dt;
		acceleration_integral[1] = acceleration_integral[1] * ACCELERATION_INTEGRAL_DECAY + acceleration[1] * dt;
		acceleration_integral[2] = acceleration_integral[2] * ACCELERATION_INTEGRAL_DECAY + acceleration[2] * dt;
	}


	if(tick_now && rl_tools_tick % 100 == 0){
		// cliPrintLinef("RAW: x %d y %d z %d w %d x %d y %d z %d vx %d vy %d vz %d",
		// 	(int)(position[0]*PRINTF_FACTOR),
		// 	(int)(position[1]*PRINTF_FACTOR),
		// 	(int)(position[2]*PRINTF_FACTOR),
		// 	(int)(q.w*PRINTF_FACTOR),
		// 	(int)(q.x*PRINTF_FACTOR),
		// 	(int)(q.y*PRINTF_FACTOR),
		// 	(int)(q.z*PRINTF_FACTOR),
		// 	(int)(linear_velocity[0]*PRINTF_FACTOR),
		// 	(int)(linear_velocity[1]*PRINTF_FACTOR),
		// 	(int)(linear_velocity[2]*PRINTF_FACTOR)
		// );
		// cliPrintLinef("ACC body: x %d y %d z %d ACC: x %d y %d z %d INTEGRAL: x %d y %d z %d",
		// 	(int)(acceleration_body[0]*PRINTF_FACTOR),
		// 	(int)(acceleration_body[1]*PRINTF_FACTOR),
		// 	(int)(acceleration_body[2]*PRINTF_FACTOR),
		// 	(int)(acceleration[0]*PRINTF_FACTOR),
		// 	(int)(acceleration[1]*PRINTF_FACTOR),
		// 	(int)(acceleration[2]*PRINTF_FACTOR),
		// 	(int)(acceleration_integral[0]*PRINTF_FACTOR),
		// 	(int)(acceleration_integral[1]*PRINTF_FACTOR),
		// 	(int)(acceleration_integral[2]*PRINTF_FACTOR)
		// );
	}

	if(USE_ACCELERATION_INTEGRAL_FEEDFORWARD_TERM){
		// since the mocap position/velocity feedback has a significant delay, we add an accelerometer-based feedforward term
		linear_velocity[0] += acceleration_integral[0];
		linear_velocity[1] += acceleration_integral[1];
		linear_velocity[2] += acceleration_integral[2];
	}



    T aux1 = rcData[4];
    bool next_active = aux1 > 1750;
    if(!active && next_active){
        reset();
        cliPrintLinef("Resetting Inference Executor (Recurrent State)");
		activation_tick += 1;
    }
    active = next_active;

	T rc_0 = rcData[4];
	if(prev_rc_0_set){
		if(prev_rc_0 != rc_0){
			cliPrintLinef("RC 0 changed from %d/%d to %d/%d", (int)(prev_rc_0*PRINTF_FACTOR), PRINTF_FACTOR, (int)(rc_0*PRINTF_FACTOR), PRINTF_FACTOR);
		}
	}
	prev_rc_0 = rc_0;
	prev_rc_0_set = true;


    // float roll = rcData[0];
    // float pitch = rcData[1];
    // float yaw = rcData[2];
    // float throttle = rcData[3];
    // static constexpr T MANUAL_POSITION_GAIN = 0.5;
    // target_position[0] = MANUAL_POSITION_GAIN * (pitch - 1500) / 500; // pitch
    // target_position[1] = -MANUAL_POSITION_GAIN * (roll - 1500) / 500; // roll
    // target_position[2] = MANUAL_POSITION_GAIN * (throttle - 1500) / 500 + 0.2f; // throttle

	RLtoolsInferenceApplicationsL2FObservation observation;
	RLtoolsInferenceApplicationsL2FAction action;
	observe(observation, TestObservationMode::ACTION_HISTORY);
	#ifdef OVERWRITE_DEFAULT_LED_WITH_POSITION_FEEDBACK
	#if defined(RL_TOOLS_BETAFLIGHT_TARGET_PAVO20)
	ledSet(0, fabsf(observation.position[0]) > 0.1f);
	#else
	ledSet(1, fabsf(observation.position[0]) > 0.1f);
	#endif
	#endif
	if(tick_now && rl_tools_tick % 1000 == 0){
		cliPrintLinef("OBS: x %d y %d z %d w %d x %d y %d z %d vx %d vy %d vz %d avx %d avy %d avz %d acc-x %d acc-y %d acc-z %d",
			(int)(observation.position[0]*PRINTF_FACTOR),
			(int)(observation.position[1]*PRINTF_FACTOR),
			(int)(observation.position[2]*PRINTF_FACTOR),
			(int)(observation.orientation[0]*PRINTF_FACTOR),
			(int)(observation.orientation[1]*PRINTF_FACTOR),
			(int)(observation.orientation[2]*PRINTF_FACTOR),
			(int)(observation.orientation[3]*PRINTF_FACTOR),
			(int)(observation.linear_velocity[0]*PRINTF_FACTOR),
			(int)(observation.linear_velocity[1]*PRINTF_FACTOR),
			(int)(observation.linear_velocity[2]*PRINTF_FACTOR),
			(int)(observation.angular_velocity[0]*PRINTF_FACTOR),
			(int)(observation.angular_velocity[1]*PRINTF_FACTOR),
			(int)(observation.angular_velocity[2]*PRINTF_FACTOR),
			(int)(acceleration_body[0]*PRINTF_FACTOR),
			(int)(acceleration_body[1]*PRINTF_FACTOR),
			(int)(acceleration_body[2]*PRINTF_FACTOR)
		);
	}
	timeUs_t pre_inference = micros();
	RLtoolsInferenceExecutorStatus executor_status = rl_tools_inference_applications_l2f_control(now*1000, &observation, &action);
	if(executor_status.source == RL_TOOLS_INFERENCE_EXECUTOR_STATUS_SOURCE_CONTROL){
		if(executor_status.step_type == RL_TOOLS_INFERENCE_EXECUTOR_STATUS_STEP_TYPE_NATIVE){
			native_statii[native_statii_index] = executor_status;
			native_statii_index++;
			if(native_statii_index >= NUM_EXECUTOR_STATII){
				native_statii_index = 0;
				native_statii_full = true;
			}
		}
		else{
			intermediate_statii[intermediate_statii_index] = executor_status;
			intermediate_statii_index++;
			if(intermediate_statii_index >= NUM_EXECUTOR_STATII){
				intermediate_statii_index = 0;
				intermediate_statii_full = true;
			}
		}
	}
	auto inference_time = micros() - pre_inference;
    // printf("observation: [%.2f %.2f %.2f] [%.2f %.2f %.2f %.2f] [%.2f %.2f %.2f] [%.2f %.2f %.2f] [%.2f %.2f %.2f %.2f] => [%.2f %.2f %.2f %.2f]\n",
	// 	   (double)observation.position[0], (double)observation.position[1], (double)observation.position[2],
	// 	   (double)observation.orientation[0], (double)observation.orientation[1], (double)observation.orientation[2], (double)observation.orientation[3],
	// 	   (double)observation.linear_velocity[0], (double)observation.linear_velocity[1], (double)observation.linear_velocity[2],
	// 	   (double)observation.angular_velocity[0], (double)observation.angular_velocity[1], (double)observation.angular_velocity[2],
	// 	   (double)observation.previous_action[0], (double)observation.previous_action[1], (double)observation.previous_action[2], (double)observation.previous_action[3],
	// 	   (double)action.action[0], (double)action.action[1], (double)action.action[2], (double)action.action[3]
	// 	);

    uint8_t target_indices[4] = {1, 0, 2, 3}; // remapping from Crazyflie to Betaflight motor indices
    // uint8_t target_indices[4] = {0, 1, 2, 3};
    for(TI action_i = 0; action_i < RL_TOOLS_INTERFACE_APPLICATIONS_L2F_ACTION_DIM; action_i++){
        if(active){

			T clipped_action = clip(action.action[action_i], (T)-1, (T)1);
			previous_action[action_i] = clipped_action;
			clipped_action = (clipped_action * 0.5f + 0.5f); // [0, 1]
			T nerfed_action = clipped_action * MOTOR_FACTOR;
			rl_tools_rpms[action_i] = nerfed_action;
			#ifdef RL_TOOLS_BETAFLIGHT_TARGET_PAVO20
			static constexpr T MIN_THROTTLE = 0.2f; // motors turn off otherwise
			if(nerfed_action < MIN_THROTTLE){
				nerfed_action = MIN_THROTTLE;
			}
			#endif
            motor[target_indices[action_i]] = nerfed_action * 2000;
			// motor[target_indices[action_i]] =  (activation_tick % 10) * 200;
			// if(activation_tick % 5 == action_i){
			// 	motor[target_indices[action_i]] =  (activation_tick % 10) * 200;
			// }
			// else{
			// 	motor[target_indices[action_i]] = 0;
			// }
			// motor[target_indices[action_i]] = 1000 + 1000 * (rl_tools_tick % 10000) / 10000.0;
		}
		else{
			#if defined(RL_TOOLS_BETAFLIGHT_TARGET_BETAFPVG473) || defined(RL_TOOLS_BETAFLIGHT_TARGET_PAVO20)
			motor[target_indices[action_i]] = 0; // stop the motors
			#endif
		}
    }
	if(tick_now && rl_tools_tick % 1000 == 0){
		cliPrintLinef("Action %d %d %d %d", 
			(int)(motor[0]),
			(int)(motor[1]),
			(int)(motor[2]),
			(int)(motor[3])
		);
	}
	if(tick_now && rl_tools_tick % 1000 == 0){
		cliPrintLinef("Inference time: %lu us", inference_time);
		if(intermediate_statii_full){
			TI num_healthy_statii = 0;
			bool latest_non_healthy_set = false;
			RLtoolsInferenceExecutorStatus latest_non_healthy;
			for(TI i = 0; i < NUM_EXECUTOR_STATII; i++){
				TI index = (intermediate_statii_index + i) % NUM_EXECUTOR_STATII;
				auto status = intermediate_statii[i];
				num_healthy_statii += status.OK;
				if(!status.OK){
					latest_non_healthy_set = true;
					latest_non_healthy = status;
				}
			}
			cliPrintLinef("RLtoolsPolicy: INTERMEDIATE: %d/%d healthy status", num_healthy_statii, NUM_EXECUTOR_STATII);
			if(latest_non_healthy_set && (!latest_non_healthy.timing_bias.OK || !latest_non_healthy.timing_jitter.OK)){
				cliPrintLinef("RLtoolsPolicy: INTERMEDIATE: BIAS %d/%dx JITTER %d/%dx", (int)(latest_non_healthy.timing_bias.MAGNITUDE * PRINTF_FACTOR), PRINTF_FACTOR, (int)(latest_non_healthy.timing_jitter.MAGNITUDE * PRINTF_FACTOR), PRINTF_FACTOR);
			}
		}
		if(native_statii_full){
			TI num_healthy_statii = 0;
			bool latest_non_healthy_set = false;
			RLtoolsInferenceExecutorStatus latest_non_healthy;
			for(TI i = 0; i < NUM_EXECUTOR_STATII; i++){
				TI index = (native_statii_index + i) % NUM_EXECUTOR_STATII;
				auto status = native_statii[i];
				num_healthy_statii += status.OK;
				if(!status.OK){
					latest_non_healthy_set = true;
					latest_non_healthy = status;
				}
			}
			cliPrintLinef("RLtoolsPolicy: NATIVE: %d/%d healthy status", num_healthy_statii, NUM_EXECUTOR_STATII);
			if(latest_non_healthy_set && (!latest_non_healthy.timing_bias.OK || !latest_non_healthy.timing_jitter.OK)){
				cliPrintLinef("RLtoolsPolicy: NATIVE: BIAS %d/%dx JITTER %d/%dx", (int)(latest_non_healthy.timing_bias.MAGNITUDE * PRINTF_FACTOR), PRINTF_FACTOR, (int)(latest_non_healthy.timing_jitter.MAGNITUDE * PRINTF_FACTOR), PRINTF_FACTOR);
			}
		}
		if(rl_tools_control_invocation_dts_full){
			T rl_tools_control_invocation_dt_mean = 0;
			T rl_tools_control_invocation_dt_std = 0;
			for(TI i = 0; i < NUM_RL_TOOLS_CONTROL_INVOCATION_DTS; i++){
				auto dt = rl_tools_control_invocation_dts[i];
				rl_tools_control_invocation_dt_mean += dt;
				rl_tools_control_invocation_dt_std += dt * dt;
			}
			rl_tools_control_invocation_dt_mean /= NUM_RL_TOOLS_CONTROL_INVOCATION_DTS;
			rl_tools_control_invocation_dt_std /= NUM_RL_TOOLS_CONTROL_INVOCATION_DTS;
			rl_tools_control_invocation_dt_std -= rl_tools_control_invocation_dt_mean * rl_tools_control_invocation_dt_mean;
			rl_tools_control_invocation_dt_std = sqrt(rl_tools_control_invocation_dt_std);
			cliPrintLinef("RLtoolsPolicy: Control invocation dt mean: %lu us, std: %lu us", (unsigned long)rl_tools_control_invocation_dt_mean, (unsigned long)rl_tools_control_invocation_dt_std);
		}
	}
}