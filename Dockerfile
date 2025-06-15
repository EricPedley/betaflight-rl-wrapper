FROM --platform=linux/amd64 ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt upgrade
RUN apt -y install build-essential git curl clang-18 python3 python-is-python3
# RUN git clone https://github.com/betaflight/betaflight.git
WORKDIR /rl-tools/embedded_platforms/betaflight/betaflight
# RUN make arm_sdk_install
# RUN make configs
