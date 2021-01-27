from ACL_bench.gym_flowers_v2.envs.bodies.AbstractBody import AbstractBody
from ACL_bench.gym_flowers_v2.envs.bodies.BodyTypesEnum import BodyTypesEnum

class AmphibianAbstractBody(AbstractBody):
    def __init__(self, scale, motors_torque, density):
        super(AmphibianAbstractBody, self).__init__(scale, motors_torque)

        self.body_type = BodyTypesEnum.AMPHIBIAN
        self.DENSITY = density