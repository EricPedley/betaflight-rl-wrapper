EXTERNAL_SRC_DIR := $(ROOT_DIR)/../src
CXXFLAGS      = $(filter-out -std=gnu17,$(CFLAGS)) -fno-rtti -fno-exceptions -std=c++17 -I $(RL_TOOLS_ROOT)/include -I $(EXTERNAL_SRC_DIR) -Wno-unused-parameter -Wno-unused-variable -Wno-unused-local-typedefs

CXX_SRC = rl_tools/policy.cpp
TARGET_OBJS += $(addsuffix .o,$(addprefix $(TARGET_OBJ_DIR)/,$(basename $(CXX_SRC))))

# SRC += rl_tools/policy.cpp

$(TARGET_OBJ_DIR)/%.o: $(EXTERNAL_SRC_DIR)/%.cpp
	$(V1) mkdir -p $(dir $@)
	$(V1) $(CROSS_CXX) -c -o $@ $(CXXFLAGS) $(CC_DEFAULT_OPTIMISATION) $<

# Same as in Makefile but using CROSS_CXX instead
$(TARGET_ELF): $(TARGET_OBJS) $(LD_SCRIPT) $(LD_SCRIPTS)
	@echo "Linking $(TARGET_NAME)" "$(STDOUT)"
	$(V1) $(CROSS_CC) -o $@ $(filter-out %.ld,$^) $(LD_FLAGS)
	$(V1) $(SIZE) $(TARGET_ELF)
