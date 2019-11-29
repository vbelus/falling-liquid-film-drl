from gym.envs.registration import register

register(
    id='film-v0',
    entry_point='gym_film.envs:FilmEnv',
    reward_threshold=1.0,
)

register(
    id='film-parallel-v0',
    entry_point='gym_film.envs:FilmEnvClient',
    reward_threshold=1.0,
)
