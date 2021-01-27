from gym.envs.registration import register

register(
    id='parametric-continuous-walker-v0',
    entry_point='ACL_bench.gym_flowers_v2.envs.parametric_continuous_walker:ParametricContinuousWalker'
)

register(
    id='parametric-continuous-drawing-path-v0',
    entry_point='ACL_bench.gym_flowers_v2.envs.parametric_continuous_drawing_path:ParametricContinuousDrawingPath'
)

register(
    id='parametric-continuous-climber-v0',
    entry_point='ACL_bench.gym_flowers_v2.envs.parametric_continuous_climber:ParametricContinuousClimber'
)

register(
    id='parametric-continuous-parkour-v0',
    entry_point='ACL_bench.gym_flowers_v2.envs.parametric_continuous_parkour:ParametricContinuousParkour'
)

register(
    id='parametric-continuous-flat-parkour-v0',
    entry_point='ACL_bench.gym_flowers_v2.envs.parametric_continuous_flat_parkour:ParametricContinuousWalker'
)