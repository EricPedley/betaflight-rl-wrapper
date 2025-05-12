```
./build.sh
```

```
make arm_sdk_install
make configs
make TARGET=HUMMINGBIRD_F4_V4 RL_TOOLS_PATH=$(pwd)/../../../
```

Note: On macOS you might need to increase the memory available to the Docker VM



### Debug

```
 arm-none-eabi-nm --print-size --plugin ./tools/gcc-arm-none-eabi-10.3-2021.10/lib/gcc/arm-none-eabi/10.3.1/liblto_plugin.so obj/main/HUMMINGBIRD_F4_V4/src/main/flight/rl_tools_adapter.o
```