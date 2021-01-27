from enum import Enum
from ACL_bench.gym_flowers_v2.envs.bodies.BodyTypesEnum import BodyTypesEnum

from ACL_bench.gym_flowers_v2.envs.bodies.back_bodies.BackChimpanzee import BackChimpanzee

from ACL_bench.gym_flowers_v2.envs.bodies.climbers.ClimbingChestProfileChimpanzee import ClimbingChestProfileChimpanzee
from ACL_bench.gym_flowers_v2.envs.bodies.climbers.ClimbingProfileChimpanzee import ClimbingProfileChimpanzee

from ACL_bench.gym_flowers_v2.envs.bodies.swimmers.FishBody import FishBody

from ACL_bench.gym_flowers_v2.envs.bodies.amphibians.AmphibiousBipedalBody import AmphibiousBipedalBody

from ACL_bench.gym_flowers_v2.envs.bodies.walkers.old.OldBigQuadruBody import OldBigQuadruBody
from ACL_bench.gym_flowers_v2.envs.bodies.walkers.old.OldClassicBipedalBody import OldClassicBipedalBody
from ACL_bench.gym_flowers_v2.envs.bodies.walkers.SmallBipedalBody import SmallBipedalBody
from ACL_bench.gym_flowers_v2.envs.bodies.walkers.BigQuadruBody import BigQuadruBody
from ACL_bench.gym_flowers_v2.envs.bodies.walkers.ClassicBipedalBody import ClassicBipedalBody
from ACL_bench.gym_flowers_v2.envs.bodies.walkers.HumanBody import HumanBody
from ACL_bench.gym_flowers_v2.envs.bodies.walkers.MillipedeBody import MillipedeBody
from ACL_bench.gym_flowers_v2.envs.bodies.walkers.ProfileChimpanzee import ProfileChimpanzee
from ACL_bench.gym_flowers_v2.envs.bodies.walkers.SpiderBody import SpiderBody
from ACL_bench.gym_flowers_v2.envs.bodies.walkers.WheelBody import WheelBody


class BodiesEnum(Enum):
    small_bipedal = SmallBipedalBody
    classic_bipedal = ClassicBipedalBody
    big_quadru = BigQuadruBody
    spider = SpiderBody
    millipede = MillipedeBody
    wheel = WheelBody
    human = HumanBody
    old_classic_bipedal = OldClassicBipedalBody
    profile_chimpanzee = ProfileChimpanzee
    back_chimpanzee = BackChimpanzee
    old_big_quadru = OldBigQuadruBody
    fish = FishBody
    climbing_profile_chimpanzee = ClimbingProfileChimpanzee
    climbing_chest_profile_chimpanzee = ClimbingChestProfileChimpanzee
    amphibious_bipedal = AmphibiousBipedalBody

    @classmethod
    def get_body_type(self, body_name):
        if body_name in ['climbing_chest_profile_chimpanzee', 'climbing_profile_chimpanzee']:
            return BodyTypesEnum.CLIMBER
        elif body_name == 'fish':
            return BodyTypesEnum.SWIMMER
        elif body_name == 'amphibious_bipedal':
            return BodyTypesEnum.AMPHIBIAN
        else:
            return BodyTypesEnum.WALKER