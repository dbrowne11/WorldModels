from gymnasium.envs.registration import register

register(
    id='MultiCarRacing-v1',
    entry_point='Envs:MultiCarRacingV1',
    max_episode_steps=None,
)

register(
    id='CarRacingViz-v1',
    entry_point='Envs:CarRacingVisualizer',
    max_episode_steps= 1000
)


from Envs.multiagent_CarRacing import MultiCarRacingV1
from Envs.CarRacingVisualizer import CarRacingViz