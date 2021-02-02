import numpy as np
import scipy.stats as sp
import math
import seaborn as sns; sns.set()

# thank you https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

class ToyEnvV2(object):  # n-dimensional grid
    def __init__(self, nb_cubes=10, idx_first_cube=0, nb_dims=2, noise=0.0):
        self.nb_cubes = nb_cubes  # Number of hypercubes per dimensions
        self.nb_dims = nb_dims  # Number of dimensions

        self.nb_total_cubes = nb_cubes ** nb_dims
        self.step_size = 1 / nb_cubes
        self.bnds = [np.arange(0, 1 + self.step_size, self.step_size) for i in range(nb_dims)]
        self.params = []
        self.cube_competence = np.zeros((nb_cubes,) * nb_dims)
        self.noise = noise
        self.max_per_cube = 100
        self.epsilon = 1e-6

        # get first cube index
        x = np.arange(0,nb_cubes,1)
        y = np.arange(0,nb_cubes,1)
        all_cube_idxs = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        self.idx_first_cube = np.array(all_cube_idxs[idx_first_cube])
        print(self.idx_first_cube)


    def reset(self):
        self.cube_competence = np.zeros((self.nb_cubes,) * self.nb_dims)
        self.params = []

    def get_score(self):  # Returns the percentage of "mastered" hypercubes (A cube is "mastered" if its competence >75)
        score = np.where(self.cube_competence > (3 * (self.max_per_cube / 4)))  #
        return (len(score[0]) / self.nb_total_cubes) * 100

    def get_cube_competence(self):
        return self.cube_competence

    def get_cube_idx_from_param(self, param):
        arr_p = np.array([param])
        cubes = sp.binned_statistic_dd(arr_p, np.ones(arr_p.shape), 'count',
                                       bins=self.bnds).statistic
        cube_idx = tuple([v[0] for v in cubes[0].nonzero()])
        return cube_idx

    def episode(self, param):
        # Ensure param values fall in bounds
        for v in param:
            if (v < 0.0) or (v > 1.0):
                print('param is out of bounds')
                exit(1)

        p = np.array(param[0:self.nb_dims]).astype(np.float32)  # discard potential useless dimensions
        #print(p)
        # add epsilons if at task space boundary to prevent binning issues
        for i,v in enumerate(p):
            if v == 0:
                p = p.copy()
                p[i] += self.epsilon
            if v == 1:
                p = p.copy()
                p[i] -= self.epsilon

        self.params.append(p)

        # 1 - Find in which hypercube the parameter vector falls
        cube_idx = self.get_cube_idx_from_param(param)

        # 2 - Check if hypercube is "unlocked" by checking if a previous adjacent neighbor is unlocked
        #if all(v == 0 for v in cube_idx):  # If initial cube, no need to have unlocked neighbors to learn
        if (self.idx_first_cube == np.array(cube_idx)).all():  # If initial cube, no need to have unlocked neighbors to learn
            self.cube_competence[cube_idx] = min(self.cube_competence[cube_idx] + 1, self.max_per_cube)
        else:  # Find index of previous adjacent neighboring hypercubes
            prev_cube_idx = [[idx, max(0, idx - 1), min(self.nb_cubes-1, idx + 1)] for idx in cube_idx]
            previous_neighbors_idx = np.array(np.meshgrid(*prev_cube_idx)).T.reshape(-1, len(prev_cube_idx))
            for pn_idx in previous_neighbors_idx:
                prev_idx = tuple(pn_idx)
                if all(v == cube_idx[i] for i, v in enumerate(prev_idx)):  # Original hypercube, not previous neighbor
                    continue
                if not any(v == cube_idx[i] for i, v in enumerate(prev_idx)):  # enforce 4-neighbors, not 8
                    continue
                else:
                    if self.cube_competence[prev_idx] >= (
                            3 * (self.max_per_cube / 4)):  # Previous neighbor with high comp
                        self.cube_competence[cube_idx] = min(self.cube_competence[cube_idx] + 1, self.max_per_cube)
                        break

        normalized_competence = np.interp(self.cube_competence[cube_idx], (0, self.max_per_cube), (0, 1))
        return normalized_competence