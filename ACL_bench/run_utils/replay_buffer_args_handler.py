from ACL_bench.run_utils.abstract_args_handler import AbstractArgsHandler
from ACL_bench.teachers.algos.replay_buffers.LPReplayBuffer import LPReplayBuffer
from ACL_bench.teachers.algos.replay_buffers.PER import PrioritizedReplayBuffer
from ACL_bench.teachers.algos.replay_buffers.random_replay_buffer import ReplayBuffer


class ReplayBufferArgsHandler(AbstractArgsHandler):
    @staticmethod
    def set_parser_arguments(parser):
        parser.add_argument('--replay', type=str,
                            default="default")  # replay buffer type ("default", "lp","alp","rew","per")
        parser.add_argument('--replay_alpha', type=float, default=1.0)

    @staticmethod
    def get_object_from_arguments(args, teacher_task_dim, obs_dim, act_dim):
        if args.replay == 'default':
            replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=args.buf_size)
        elif args.replay == 'alp':
            replay_buffer = LPReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                           size=args.buf_size, task_dim=teacher_task_dim, alpha=args.replay_alpha,
                                           use_alp=True)
        elif args.replay == 'lp':
            replay_buffer = LPReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                           size=args.buf_size, task_dim=teacher_task_dim, alpha=args.replay_alpha,
                                           use_alp=False)
        elif args.replay == 'rew':
            replay_buffer = LPReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                           size=args.buf_size, task_dim=teacher_task_dim, alpha=args.replay_alpha,
                                           use_alp=False, reward_based=True)
        elif args.replay == 'per':
            replay_buffer = PrioritizedReplayBuffer(args.buf_size, 0.0)  # TODO alpha as argument, not always 0.6
        else:
            raise NotImplementedError('Unknown replay buffer type: {}'.format(args.replay))


        return replay_buffer
