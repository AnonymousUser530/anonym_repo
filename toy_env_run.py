import argparse
from TeachMyAgent.run_utils.teacher_args_handler import TeacherArgsHandler
from TeachMyAgent.toy_env.toy_env import ToyEnvV2
import numpy as np
import json
import time
import copy
from TeachMyAgent.students.spinup.utils.run_utils import setup_logger_kwargs
from TeachMyAgent.students.spinup.utils.logx import EpochLogger
import pickle

if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='toy_env_test')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--nb_episodes', type=float, default=2)  # Training time, expressed in 100K episodes
    parser.add_argument('--epoch_step', type=float, default=0.1)  # Test interval, expressed in 100K episodes
    parser.add_argument('--nb_cubes', type=int, default=None)
    parser.add_argument('--nb_rot', type=int, default=None)
    parser.add_argument('--rnd_start_cube', '-rsc', action='store_true')
    parser.add_argument('--light', action='store_true')  # lightweight saves
    parser.add_argument('--nb_test_episodes', type=int, default=1) # USEFUL TO THE TEACHERCONTROLLER BUT HAS NO IMPACT

    # Teacher-specific arguments:
    TeacherArgsHandler.set_parser_arguments(parser)

    #################################################
    ############## Initialize teacher ###############
    #################################################
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
    save_dir = logger.output_dir

    # SEED EXPERIMENT
    np.random.seed(args.seed)

    initial_dist = {
        "mean": np.array([0, 0]),
        "variance": np.diag([0.01, 0.01])
    }
    target_dist = {
        "mean": np.array([0.5, 0.5]),
        "variance": np.diag([0.25, 0.25])
    }

    env_bounds = {
        "param": [[0, 1],
                   [0, 1]]
    }

    args.env_reward_lb = 0
    args.env_reward_ub = 1

    Teacher = TeacherArgsHandler.get_object_from_arguments(args, env_bounds, initial_dist, target_dist)
    teacher_name = args.teacher

    # Initialize toy env
    nb_dims = 2

    if args.nb_cubes is None:
        nb_cubes = np.random.randint(10,31)
    else:
        nb_cubes = args.nb_cubes
    if args.rnd_start_cube:
        start_cube_idx = np.random.randint(0,400)
    else:
        start_cube_idx = args.seed % (nb_cubes**nb_dims)
    print("start cube idx: {}".format(start_cube_idx))
    if args.nb_rot is None:
        nb_task_space_rot = np.random.randint(0,4)
    else:
        nb_task_space_rot = args.nb_rot
    # if args.use_ground_truth:
    #     if args.toy_env_2:
    #         teacher_params['student_params'] = [start_cube_idx]
    #     else:
    #         teacher_params['student_params'] = [nb_task_space_rot]

    print("init book keeping")
    # dump config
    config_dict = vars(args)
    config_dict['nb_cubes'] = nb_cubes
    config_dict['nb_rot'] = nb_task_space_rot
    config_dict['start_cube_idx'] = start_cube_idx
    with open(save_dir + '/config.json', "w", encoding="utf8") as handle:
         json.dump(config_dict, handle)

    nb_episodes = int(args.nb_episodes * 100000)
    epoch_step = args.epoch_step * 100000
    env = ToyEnvV2(nb_dims=nb_dims, nb_cubes=nb_cubes, idx_first_cube=start_cube_idx)
    env.reset()

    epochs_score = []
    epochs_time = []
    epochs_episode_nb = []
    epochs_comp_grid = []
    comp_grid_at_teacher_updates = [env.get_cube_competence().astype(np.int8)]
    train_tasks = []
    train_rewards = []
    episode_all_mastered = -1


    print('launching {} for {} on toy env with {} cubes and {} 90rot'.format(teacher_name, nb_episodes, nb_cubes, nb_task_space_rot))
    # Main loop: collect experience in env and update/log each epoch
    verbose = True
    start_time = time.time()

    def value_estimator(states):
        competences = env.get_cube_competence()
        return np.array([competences[tuple(state)] for state in states])

    Teacher.set_value_estimator(value_estimator)
    for i in range(nb_episodes + 1):
        if (i % epoch_step) == 0:  # training epoch completed, record score
            epochs_time.append(time.time() - start_time)
            epochs_score.append(env.get_score())
            epochs_comp_grid.append(env.get_cube_competence().astype(np.int8))
            epochs_episode_nb.append(i)
            if nb_dims == 2:
                if verbose:
                    print("it:{}, score:{}".format(i, epochs_score[-1]))
                    print(epochs_comp_grid[-1])
            else:
                if verbose:
                    print("it:{}, score:{}".format(i, epochs_score[-1]))

            if teacher_name == 'CEGT' and i > 0:
                Teacher.send_test_info(epochs_comp_grid[-1].flatten())

        # sample task params
        task_params = copy.copy(Teacher.task_generator.sample_task())
        assert type(task_params[0]) == np.float32
        train_tasks.append(task_params)
        Teacher.task_generator.record_initial_state(task_params,
                                                    env.get_cube_idx_from_param(task_params))

        if (i % 50) == 0:
            if env.get_score() == 100.0 and episode_all_mastered == -1:
                print('task space mastered at ep {} !!'.format(i))
                episode_all_mastered = i
        #print('ep:{},p={}'.format(i, task_params))

        reward = env.episode(task_params)
        for i in range(int(reward)):
            Teacher.task_generator.step_update(None, None, i, None, None)

        is_teacher_updated = Teacher.task_generator.episodic_update(np.array(task_params), reward, 0 < reward < 1)
        if is_teacher_updated:
            comp_grid_at_teacher_updates.append(env.get_cube_competence().astype(np.int8))

        train_rewards.append(reward)

    # Pickle data
    with open(save_dir + '/env_params_save.pkl', 'wb') as handle:
        dump_dict = {'env_params_train': np.array(train_tasks).astype(np.float16),
                     'env_train_rewards': np.array(train_rewards).astype(np.float16),
                     'epochs_score': epochs_score,
                     'epochs_comp_grid': epochs_comp_grid,
                     'teacher_update_comp_grid': comp_grid_at_teacher_updates,
                     'epochs_episode_nb': epochs_episode_nb,
                     'time': epochs_time,
                     'ep_all_mastered': episode_all_mastered}
        dump_dict = Teacher.task_generator.dump(dump_dict)
        if teacher_name == 'ALP-GMM' or (teacher_name == 'CEGT' and args.use_alpgmm):
            dump_dict['tasks_alps'] = np.array(dump_dict['tasks_alps']).astype(np.float16)
            dump_dict['tasks_origin'] = np.array(dump_dict['tasks_origin']).astype(np.int8)

        if args.light is True:  # discard non essential saves
            light_dump_dict = {'epochs_score': epochs_score,
                               'epochs_comp_grid': epochs_comp_grid,
                               'epochs_episode_nb': epochs_episode_nb,
                               'time': epochs_time,
                               'ep_all_mastered': episode_all_mastered}
            dump_dict = light_dump_dict


        pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)