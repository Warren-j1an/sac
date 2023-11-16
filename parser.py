import argparse


def get_parser(configs):
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="quadruped_walk")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_train_frames', default=3e6, type=float)
    args = parser.parse_args()

    configs.task = args.task
    configs.seed = args.seed
    configs.num_train_frames = args.num_train_frames
    configs.logdir = configs.logdir + args.task + '/' + str(args.seed)
    return configs
