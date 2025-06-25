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
#ifdef RL_TOOLS_SERIAL
#include "drivers/serial.h"
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wregister"
#pragma GCC diagnostic ignored "-Wpedantic"
extern "C" {
    #include "rx/rx.h"
    #include "flight/mixer.h"
    #include "sensors/gyro.h"
    #include "flight/imu.h"
    #include "drivers/time.h"
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
    static constexpr TI CONTROL_INTERVAL_INTERMEDIATE_NS = 1 * 1000 * 1000; // Inference is at 500hz
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

uint64_t previous_micros = 0;
bool previous_micros_set = false;
uint32_t micros_overflow_counter = 0;

bool first_run = true;
bool active = false;

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
static constexpr T MAX_POSITION_ERROR = 0.3;
static constexpr T MAX_LINEAR_VELOCITY_ERROR = 0.3;

// setpoint
static T target_position[3] = {0, 0, 0.2};
static T target_orientation[4] = {1, 0, 0, 0};
static T target_linear_velocity[3] = {0, 0, 0};

// input
T rl_tools_position[3] = {0, 0, 0};
T rl_tools_linear_velocity[3] = {0, 0, 0};

T rl_tools_rpms[4] = {0, 0, 0, 0};

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
		p[0] = (rl_tools_position[0] - target_position[0]);
		p[1] = (rl_tools_position[1] - target_position[1]);
		p[2] = (rl_tools_position[2] - target_position[2]);
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
		v[0] = (rl_tools_linear_velocity[0] - target_linear_velocity[0]);
		v[1] = (rl_tools_linear_velocity[1] - target_linear_velocity[1]);
		v[2] = (rl_tools_linear_velocity[2] - target_linear_velocity[2]);
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
	for(int action_i=0; action_i < RL_TOOLS_INTERFACE_APPLICATIONS_L2F_ACTION_DIM; action_i++){
		previous_action[action_i] = -1;
	}
	rl_tools_inference_applications_l2f_reset();
}


extern "C" float rl_tools_test(void){
	// RLtoolsInferenceApplicationsL2FAction action;
	// float abs_diff = rl_tools_inference_applications_l2f_test(&action);
	// printf("Checkpoint test diff: %f\n", abs_diff);
	// for(TI output_i = 0; output_i < RL_TOOLS_INTERFACE_APPLICATIONS_L2F_ACTION_DIM; output_i++){
	// 	printf("output[%d]: %f\n", output_i, action.action[output_i]);
	// }
    // return abs_diff;
    float aux1 = rcData[4];
    // printf("AUX1 %f\n", aux1);
    float motor1 = motor[0];
    printf("Motor1 %f\n", (double)motor1);
    return 0;
}

extern "C" void rl_tools_control(void){
    if(first_run){
        first_run = false;
        printf("RLtools Inference Applications L2F Control started\n");
        rl_tools_inference_applications_l2f_init();
    }
    timeUs_t now_narrow = micros();
    if(previous_micros_set && (now_narrow < previous_micros)){
        micros_overflow_counter++;
		printf("Micros overflow\n");
		exit(1);
    }
    previous_micros = now_narrow;
    uint64_t now = micros_overflow_counter * std::numeric_limits<timeUs_t>::max() + now_narrow;

    float aux1 = rcData[4];
    bool next_active = aux1 > 1750;
    if(!active && next_active){
        reset();
        printf("Resetting Inference Executor (Recurrent State)\n");
    }
    active = next_active;


    float roll = rcData[0];
    float pitch = rcData[1];
    float yaw = rcData[2];
    float throttle = rcData[3];
    static constexpr T MANUAL_POSITION_GAIN = 0.5;
    target_position[0] = MANUAL_POSITION_GAIN * (pitch - 1500) / 500; // pitch
    target_position[1] = -MANUAL_POSITION_GAIN * (roll - 1500) / 500; // roll
    target_position[2] = MANUAL_POSITION_GAIN * (throttle - 1500) / 500 + 0.2f; // throttle

	RLtoolsInferenceApplicationsL2FObservation observation;
	RLtoolsInferenceApplicationsL2FAction action;
	observe(observation, TestObservationMode::ACTION_HISTORY);
	auto executor_status = rl_tools_inference_applications_l2f_control(now*1000, &observation, &action);
	if(executor_status.step_type == RL_TOOLS_INFERENCE_EXECUTOR_STATUS_STEP_TYPE_NATIVE){
		if(!executor_status.timing_bias.OK || !executor_status.timing_jitter.OK){
			printf("RLtoolsPolicy: NATIVE: BIAS %fx JITTER %fx\n", (double)executor_status.timing_bias.MAGNITUDE, (double)executor_status.timing_jitter.MAGNITUDE);
		}
	}
    // printf("observation: [%.2f %.2f %.2f] [%.2f %.2f %.2f %.2f] [%.2f %.2f %.2f] [%.2f %.2f %.2f] [%.2f %.2f %.2f %.2f] => [%.2f %.2f %.2f %.2f]\n",
	// 	   (double)observation.position[0], (double)observation.position[1], (double)observation.position[2],
	// 	   (double)observation.orientation[0], (double)observation.orientation[1], (double)observation.orientation[2], (double)observation.orientation[3],
	// 	   (double)observation.linear_velocity[0], (double)observation.linear_velocity[1], (double)observation.linear_velocity[2],
	// 	   (double)observation.angular_velocity[0], (double)observation.angular_velocity[1], (double)observation.angular_velocity[2],
	// 	   (double)observation.previous_action[0], (double)observation.previous_action[1], (double)observation.previous_action[2], (double)observation.previous_action[3],
	// 	   (double)action.action[0], (double)action.action[1], (double)action.action[2], (double)action.action[3]
	// 	);

    // uint8_t target_indices[4] = {1, 0, 2, 3}; // remapping from Crazyflie to Betaflight motor indices
    uint8_t target_indices[4] = {0, 1, 2, 3};
    for(TI action_i = 0; action_i < RL_TOOLS_INTERFACE_APPLICATIONS_L2F_ACTION_DIM; action_i++){
        if(active){
			T clipped_action = clip(action.action[action_i], (T)-1, (T)1);
			previous_action[action_i] = clipped_action;
			rl_tools_rpms[action_i] = clipped_action;
            motor[target_indices[action_i]] = clipped_action * 500 + 1500;
        }
    }
	#ifdef RL_TOOLS_SERIAL
		serialPort_t *uart1 = serialFindPort(SERIAL_PORT_UART1);
		if (uart1) {
			const char *txt = "Debugging on TX1\n";
			for (const char *c = txt; *c; c++) {
				serialWrite(uart1, *c);
			}
		}
	#endif
}