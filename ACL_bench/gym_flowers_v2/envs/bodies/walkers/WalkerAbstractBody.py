from ACL_bench.gym_flowers_v2.envs.bodies.AbstractBody import AbstractBody
from ACL_bench.gym_flowers_v2.envs.bodies.BodyTypesEnum import BodyTypesEnum

class WalkerAbstractBody(AbstractBody):
    def __init__(self, scale, motors_torque, nb_steps_under_water):
        super(WalkerAbstractBody, self).__init__(scale, motors_torque)

        self.body_type = BodyTypesEnum.WALKER
        self.nb_steps_can_survive_under_water = nb_steps_under_water