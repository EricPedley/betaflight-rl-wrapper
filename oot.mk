EXTERNAL_SRC_DIR := ../../src

CXXFLAGS      = $(filter-out -std=gnu17,$(CFLAGS)) -fno-rtti -fno-exceptions -std=c++17 -I $(RL_TOOLS_ROOT)/include -I $(EXTERNAL_SRC_DIR) -Wno-unused-parameter -Wno-unused-variable -Wno-unused-local-typedefs -Wno-double-promotion -DUSE_CLI_DEBUG_PRINT

CXX_SRC = rl_tools/policy.cpp
C_SRC_EXT = rl_tools/neural_network.c rl_tools/nn_helpers.c
TARGET_OBJS += $(addsuffix .o,$(addprefix $(TARGET_OBJ_DIR)/,$(basename $(CXX_SRC))))
TARGET_OBJS += $(addsuffix .o,$(addprefix $(TARGET_OBJ_DIR)/,$(basename $(C_SRC_EXT))))

$(TARGET_OBJ_DIR)/%.o: $(EXTERNAL_SRC_DIR)/%.cpp
	$(V1) mkdir -p $(dir $@)
	$(V1) $(CROSS_CXX) -c -o $@ $(CXXFLAGS) $(CC_DEFAULT_OPTIMISATION) $<

$(TARGET_OBJ_DIR)/rl_tools/%.o: $(EXTERNAL_SRC_DIR)/rl_tools/%.c
	$(V1) mkdir -p $(dir $@)
	$(V1) $(CROSS_CC) -c -o $@ $(CFLAGS) $(CC_DEFAULT_OPTIMISATION) $<
