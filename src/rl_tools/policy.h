
#ifdef __cplusplus
extern "C" {
#endif
    void rl_tools_status(void);
    void rl_tools_control(bool);
    extern float rl_tools_position[3];
    extern float rl_tools_linear_velocity[3];
    extern float nn_input_rpms[4];

#ifdef __cplusplus
}
#endif