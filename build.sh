docker build -t rltools/betaflight . && docker run -it -v $(pwd):/betaflight --mount type=bind,source=$(cd ../../ && pwd),target=/rl-tools,readonly --rm rltools/betaflight
