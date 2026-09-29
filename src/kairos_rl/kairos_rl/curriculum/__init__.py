# Curriculum Learning environments for KAIROS Pick & Place
from kairos_rl.curriculum.level0_reach import ReachEnv
from kairos_rl.curriculum.level1_pick import PickEnv
from kairos_rl.curriculum.level2_place import PlaceEnv
from kairos_rl.curriculum.level3_full import FullPickPlaceEnv

ENVS = {
    0: ReachEnv,
    1: PickEnv,
    2: PlaceEnv,
    3: FullPickPlaceEnv,
}
