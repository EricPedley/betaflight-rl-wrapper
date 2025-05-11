docker build -t rltools/betaflight . && docker run -it -v $(pwd):/betaflight --mount type=bind,source=/Users/jonas/rl_tools,target=/rl-tools,readonly --rm rltools/betaflight
