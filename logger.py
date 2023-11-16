import json
import numpy as np
import os
import pathlib
import time
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, outputs, multiplier=1):
        self._outputs = outputs
        self._multiplier = multiplier
        self._last_step = None
        self._last_time = None
        self._metrics = []

    def add(self, mapping, step, prefix=None):
        step = step * self._multiplier
        for name, value in dict(mapping).items():
            name = f'{prefix}_{name}' if prefix else name
            value = np.array(value)
            if len(value.shape) not in (0, 2, 3, 4):
                raise ValueError(
                    f"Shape {value.shape} for name '{name}' cannot be "
                    "interpreted as scalar, image, or video.")
            self._metrics.append((step, name, value))

    def scalar(self, name, value, step):
        self.add({name: value}, step)

    def image(self, name, value, step):
        self.add({name: value}, step)

    def video(self, name, value, step):
        self.add({name: value}, step)

    def write(self, step, fps=False):
        fps and self.scalar('fps', self._compute_fps(step), step)
        if not self._metrics:
            return
        for output in self._outputs:
            output(self._metrics)
        self._metrics.clear()

    def _compute_fps(self, step):
        step = step * self._multiplier
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration


class TerminalOutput:
    def __init__(self):
        self.keys = ['eval_reward', 'eval_episode', 'train_reward', 'episode_length', 'episode', 'total_steps', 'total_episodes', 'loaded_steps',
                     'loaded_episodes', 'fps']

    def __call__(self, summaries):
        step = max(s for s, _, _, in summaries)
        scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0 and k in self.keys}
        formatted = {k: self._format_value(v) for k, v in scalars.items()}
        print(f'[{step}]', ' / '.join(f'{k} {v}' for k, v in formatted.items()))

    @staticmethod
    def _format_value(value):
        if value == 0:
            return '0'
        elif 0.01 < abs(value) < 10000:
            value = f'{value:.2f}'
            value = value.rstrip('0')
            value = value.rstrip('0')
            value = value.rstrip('.')
            return value
        else:
            value = f'{value:.1e}'
            value = value.replace('.0e', 'e')
            value = value.replace('+0', '')
            value = value.replace('+', '')
            value = value.replace('-0', '-')
            return value


class JSONLOutput:
    def __init__(self, logdir):
        self._logdir = pathlib.Path(logdir).expanduser()

    def __call__(self, summaries):
        scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
        step = max(s for s, _, _, in summaries)
        with (self._logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **scalars}) + '\n')


class TensorBoardOutput:

    def __init__(self, logdir):
        self._logdir = os.path.expanduser(logdir)
        self._sw = SummaryWriter(str(self._logdir + '/tb'))

    def __call__(self, summaries):
        for step, name, value in summaries:
            self._sw.add_scalar('scalars/' + name, value, step)
