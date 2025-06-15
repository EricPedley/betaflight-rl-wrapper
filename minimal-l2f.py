from copy import copy
import numpy as np
import asyncio, websockets, json
import l2f
from l2f import vector2 as vector

device = l2f.Device()
rng = vector.VectorRng()
env = vector.VectorEnvironment()
ui = l2f.UI()
params = vector.VectorParameters()
state = vector.VectorState()

vector.initialize_rng(device, rng, 0)
vector.initialize_environment(device, env)
vector.sample_initial_parameters(device, env, params, rng)
vector.sample_initial_state(device, env, params, state, rng)

async def main():
    uri = "ws://localhost:13337/backend"
    async with websockets.connect(uri) as websocket:
        handshake = json.loads(await websocket.recv(uri))
        assert(handshake["channel"] == "handshake")
        namespace = handshake["data"]["namespace"]
        ui.ns = namespace
        ui_message = vector.set_ui_message(device, env, ui)
        parameters_message = vector.set_parameters_message(device, env, params, ui)
        await websocket.send(ui_message)
        await websocket.send(parameters_message)
        for _ in range(500):
            action = np.ones((env.N_ENVIRONMENTS, env.ACTION_DIM), dtype=np.float32) * -1
            ui_state = copy(state)
            for i, s in enumerate(ui_state.states):
                s.position[0] += i * 0.1
            state_action_message_string = vector.set_state_action_message(device, env, params, ui, ui_state, action)
            state_action_message = json.loads(state_action_message_string)
            state_action_message["data"]["state"][0]["position"] = [0.0, 0.0, 0.0]
            state_action_message["data"]["state"][0]["orientation"] = [1.0, 0.0, 0.0, 0.0]
            await websocket.send(json.dumps(state_action_message))
            await asyncio.sleep(0.01)

if __name__ == "__main__":
    asyncio.run(main())