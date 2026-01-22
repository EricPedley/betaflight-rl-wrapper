Plan to replicate foundation policy paper and set up good dev environment for drone research:

Minimal path where everything works:
1. figure out how to send position info to SITL over RC channels
2. foundation policy just works, try hardcoding open-loop trajectory following

SITL runs now and is sending motor outputs but its just staying on the ground.

3. figure out how to send position info to real drone with rc channels (possibly lua script). Original author used a script sending to an external elrs module
4. deploy IRL and it just works
5. brainstorm extensions or new research directions

Possibly needed debugging steps:
1. log neural net debug info to blackbox
2. get blackbox logs to work in SITL
3. sync up blackbox and external position data
4. rewrite external position sending using MSP messages over the VTX port (similar to https://medium.com/illumination/fpv-autonomous-operation-with-betaflight-and-raspberry-pi-0caeb4b3ca69)
5. get bidirectional dshot to work in SITL

Things slowing me down:
1. cpp breakpoints are not working

Extensions brainstorming