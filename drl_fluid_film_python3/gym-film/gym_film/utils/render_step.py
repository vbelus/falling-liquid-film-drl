# Unused but stupid - obs and env not updated


def render_step(env, obs,  model=None, mode="no control"):
    if mode == "control":
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    elif mode == "no control":
        obs, rewards, done, info = env._no_action_step()
    elif mode == "full power":
        obs, rewards, done, info = env._full_power_action_step()
    env.render()


def render_step_vec(env, obs, model=None, mode="no control"):
    if mode == "control":
        action, _states = model.predict(obs)
        action = action[0]
        obs, rewards, done, info = env.env_method("step", action, indices=0)[0]
    elif mode == "no control":
        obs, rewards, done, info = env.env_method(
            "_no_action_step", None, indices=0)[0]
    elif mode == "full power":
        obs, rewards, done, info = env.env_method(
            "_full_power_action_step", None, indices=0)[0]
    env.env_method("render", None, indices=0)