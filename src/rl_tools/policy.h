
#ifdef __cplusplus
extern "C" {
#endif
    float rl_tools_test(void);
    void rl_tools_control(void);
    extern float rl_tools_position[3];
    extern float rl_tools_orientation[4];
    extern float rl_tools_linear_velocity[3];
    extern float rl_tools_angular_velocity[3];
    extern float rl_tools_rpms[4];

#ifdef __cplusplus
}
#endif