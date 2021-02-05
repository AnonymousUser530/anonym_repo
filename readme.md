TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL
==================================
TeachMyAgent is a testbed platform for **Automatic Curriculum Learning** methods. We leverage Box2D procedurally generated environments to assess the performance of teacher algorithms in continuous task spaces.
Our repository provides:
- **Two parametric Box2D environments**: Stumps Tracks and Parkour
- **Multiple embodiments** with different locomotion skills (e.g. bipedal walker, spider, climbing chimpanzee, fish)
- **Two DeepRL students**: SAC and PPO
- **Several ACL algorithms**: ADR, ALP-GMM, Covar-GMM, SPDL, GoalGAN, Setter-Solver, RIAC
- **Two benchmark experiments** using elements above: Skill-specific comparison and global performance assessment
- **A notebook for systematic analysis** of results using statistical tests along with visualisation tools (plots, videos...)
 
 ![global_schema](./TeachMyAgent/graphics/readme_graphics/global_schema.png)
 
 ## Table of Contents  
**[Installation](#installation)**<br>
**[Visualising results](#visualising-results)**<br>
**[Code structure](#code-structure)**<br>

## Installation

1- Get the repository
```
git clone https://github.com/AnonymousUser530/anonym_repo
cd anonym_repo/
```
2- Install it, using Conda for example (use Python >= 3.6)
```
conda create --name teachMyAgent python=3.6
conda activate teachMyAgent
pip install -e .
```

**Note: For Windows users, add `-f https://download.pytorch.org/whl/torch_stable.html` to the `pip install -e .` command.**

## Launching an experiment
You can launch an experiment using `run.py`:
```
python run.py --exp_name <name> --env <environment_name> <optional environment parameters> --student <student_name> <optional student parameters> --teacher <teacher_name> <optional teacher parameters>
```

Here is an example of a 10 millions steps training of PPO with the fish embodiment in Parkour using GoalGAN as teacher:

```
python run.py --exp_name TestExperiment --env parametric-continuous-parkour-v0 --embodiment fish --student ppo --nb_env_steps 10 --teacher GoalGAN --use_pretrained_samples
```

## Visualising results
1- Launch a jupyter server:
```
jupyter notebook
```

2- Open our Results_Analysis.ipynb notebook

3- Import your data

4- Run plot definitions

5- Run plots with appropriate parameters

## Code structure
Our code is shared between 4 main folders in the *TeachMyAgent* package:
- *environments*: definition of our two procedurally generated environments along with embodiments
- *students*: SAC and PPO's implementations
- *teachers*: all the ACL algorithms
- *run_utils*: utils for running experiments and generating benchmark scripts

### Environments
All our environments respect the OpenAI Gym's interface. We use the environment's constructor to provide parameters such as embodiment.
Our environments must additionally provide a `set_environment()` method used by teachers to set tasks (warning: this method must be called before the `reset()` function).
Here is an example showing how to use the Parkour environment:
```python
import numpy as np
import time
import gym
import TeachMyAgent.environments

env = gym.make('parametric-continuous-parkour-v0', agent_body_type='fish', movable_creepers=True)
env.set_environment(input_vector=np.zeros(3), water_level = 0.1)
env.reset()

while True:
    _, _, d, _ = env.step(env.action_space.sample())
    env.render()
    time.sleep(0.1)
```
Hence, one can easily add a new environment as long as it implements the methods presented above. The new environment must then be added to the registration in `TeachMyAgent/environments/__init__.py`.
 
#### Embodiments
We put our embodiments in `TeachMyAgent/environments/envs/bodies`. We classify them in three main categories (walkers, climbers, swimmers).
Each embodiment extends the `AbstractBody` class specifying basic methods such as sending actions to motors or creating the observation vector. Additionally, each embodiment extends an abstract class of its type (e.g. walker or swimmer) defining methods related to type-specific behaviour.
Finally, `BodiesEnum` enum is used to list all embodiments and provide access to their class using a string name. One must therefore add its new embodiment in the appropriate folder, extend and implement the methods in the appropriate abstract class and finally add its new class to the `BodiesEnum`.
  
### Students
We modified SpinningUp's implementation of SAC and OpenAI Baselines' implementation of PPO in order to make them use a teacher algorithm. For this, a DeepRL student part of our TeachMyAgent must take a teacher as parameter and call its `record_train_step` method at each step as well as its `record_train_episode` and `set_env_params` methods before every reset of the environment. Additionally, it must also take a test environment and use it to test its policy on it.
Here is an example of the way this must be implemented:
```python
# Train policy
o, r, d, ep_ret = env.reset(), 0, False, 0
Teacher.record_train_task_initial_state(o)
for t in range(total_steps):
    a = get_action(o)
    o2, r, d, infos = env.step(a)
    ep_ret += r
    Teacher.record_train_step(o, a, r, o2, d)
    o = o2

    if d:
        success = False if 'success' not in infos else infos["success"]
        Teacher.record_train_episode(ep_ret, ep_len, success)
        params = Teacher.set_env_params(env)
        o, r, d, ep_ret = env.reset(), 0, False, 0
        Teacher.record_train_task_initial_state(o)

# Test policy
for j in range(n):
    Teacher.set_test_env_params(test_env)
    o, r, d, ep_ret = test_env.reset(), 0, False, 0
    while not d:
        o, r, d, _ = test_env.step(get_action(o))
        ep_ret += r
    Teacher.record_test_episode(ep_ret, ep_len)
```
### Teachers
All our teachers extend the same `AbstractTeacher` class which defines their required methods (e.g. `episodic_update` or `sample_task`). Teachers are then called through the `TeacherController` class, being the one passed to DeepRL students.
This class handles the storage of sampled tasks, possible reward interpretation as well as test tasks used when the `set_test_env_params` method is called.
In order to add a new teacher, one must extend the `AbstractTeacher` class and add its class among the possible ones in the following lines of the `TeacherController`:
```python
# setup tasks generator
if teacher == 'Random':
    self.task_generator = RandomTeacher(mins, maxs, seed=seed, **teacher_params)
elif teacher == 'RIAC':
    self.task_generator = RIAC(mins, maxs, seed=seed, **teacher_params)
elif teacher == 'ALP-GMM':
    self.task_generator = ALPGMM(mins, maxs, seed=seed, **teacher_params)
elif teacher == 'Covar-GMM':
    self.task_generator = CovarGMM(mins, maxs, seed=seed, **teacher_params)
elif teacher == 'ADR':
    self.task_generator = ADR(mins, maxs, seed=seed, scale_reward=scale_reward, **teacher_params)
elif teacher == 'Self-Paced':
    self.task_generator = SelfPacedTeacher(mins, maxs, seed=seed, **teacher_params)
elif teacher == 'GoalGAN':
    self.task_generator = GoalGAN(mins, maxs, seed=seed, **teacher_params)
elif teacher == 'Setter-Solver':
    self.task_generator = SetterSolver(mins, maxs, seed=seed, **teacher_params)
else:
    print('Unknown teacher')
    raise NotImplementedError
```