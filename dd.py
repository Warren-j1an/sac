import collections
import os.path

from dataset_distillation import DatasetDistillation
import dmc
from logger import Logger, TerminalOutput, JSONLOutput, TensorBoardOutput
import numpy as np
from parser import get_parser
import pathlib
from replay_buffer import Replay
import ruamel.yaml as yaml
from SACAgent import SACAgent
import sys
import torch
import utils


class Workspace:
    def __init__(self):
        yaml_load = yaml.YAML(typ='safe', pure=True)
        self.configs = yaml_load.load((pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
        self.configs = utils.ObjDict(self.configs)
        self.configs = get_parser(self.configs)
        load_dir = pathlib.Path(self.configs.logdir).expanduser()
        if not load_dir.is_dir():
            raise ValueError("Load dir not exists.")
        log_dir = pathlib.Path(str(load_dir).replace('results', 'distillation_result')).expanduser()
        self.configs.logdir = str(log_dir)
        print('load dir:', str(load_dir))
        print('log dir', str(log_dir))
        log_dir.mkdir(parents=True, exist_ok=True)

        self.train_env = dmc.make(self.configs.task, self.configs.action_repeat, self.configs.seed, self.configs.vision,
                                  self.configs.height, self.configs.weight)
        self.eval_env = dmc.make(self.configs.task, self.configs.action_repeat, self.configs.seed, self.configs.vision,
                                 self.configs.height, self.configs.weight)

        self.agent = None
        if os.path.exists(str(load_dir) + '/snapshot.pt'):
            self.load(str(load_dir) + '/snapshot.pt')
        else:
            raise ValueError('Load agent not exists.')

        outputs = [
            TerminalOutput(),
            JSONLOutput(log_dir),
            TensorBoardOutput(log_dir) if self.configs.use_tb else None,
        ]
        self.logger = Logger(outputs, self.configs.action_repeat)

        self.replay_buffer = Replay(str(load_dir) + '/episode', self.configs.capacity,
                                    self.configs.min_len, self.configs.max_len, self.configs.discount,
                                    self.configs.seed, self.configs.ongoing, self.configs.prioritize_ends)

        self.eval(self.replay_buffer.total_steps)

    def train(self):
        step, episode_reward, episode_step, episode = 0, 0, 0, 0
        metrics = collections.defaultdict(list)
        time_step = self.train_env.reset()
        self.replay_buffer.add_step(time_step)
        while step * self.configs.action_repeat < self.configs.num_train_frames:
            if step % self.configs.eval_every == 0:
                self.eval(step)
                print('start train:')

            action = self.agent.act(time_step.observation, False)
            time_step = self.train_env.step(action)
            self.replay_buffer.add_step(time_step)
            episode_reward += time_step.reward
            episode_step += 1
            step += 1

            if time_step.last():
                episode += 1
                self.logger.scalar('train_reward', episode_reward, step)
                self.logger.scalar('episode_length', episode_step, step)
                self.logger.scalar('episode', episode, step)

                episode_step, episode_reward = 0, 0
                time_step = self.train_env.reset()
                self.replay_buffer.add_step(time_step)

            if step % self.configs.log_every == 0:
                for name, values in metrics.items():
                    self.logger.scalar(name, np.array(values, np.float64).mean(), step)
                    metrics[name].clear()
                self.logger.add(self.replay_buffer.stats, step)
                self.logger.write(step, True)

            if step > self.configs.update_start:
                met = {}
                for _ in range(self.configs.utd):
                    batch = next(self.replay_buffer.dataset(self.configs.batch, self.configs.length))
                    met.update(self.agent.update(batch, step))
                [metrics[key].append(value) for key, value in met.items()]

            if step % self.configs.save_every == 0:
                self.save()

        self.eval(step)
        self.save()

    def eval(self, step):
        print('start evaluate:')
        reward = 0
        for episode in range(self.configs.num_eval_episodes):
            episode_reward = 0
            time_step = self.eval_env.reset()
            while not time_step.last():
                action = self.agent.act(time_step.observation, True)
                time_step = self.eval_env.step(action)
                episode_reward += time_step.reward
            reward += episode_reward
        reward /= self.configs.num_eval_episodes
        self.logger.scalar('eval_reward', reward, step)
        self.logger.scalar('eval_episode', self.configs.num_eval_episodes, step)
        self.logger.write(step, True)

    def save(self):
        # TODO
        pass

    def load(self, path):
        filename = pathlib.Path(path).expanduser()
        with filename.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


if __name__ == '__main__':
    work = Workspace()
    # work.train()
