# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class RenderWrapper(dm_env.Environment):
    def __init__(self, env, vision, camera=2, size=(84, 84)):
        self.env = env
        self.vision = vision
        self.camera = camera
        self.size = size

        if self.vision:
            self.shape = np.array([3, *self.size])
        else:
            shape = 0
            for k, v in env.observation_spec().items():
                shape = shape + v.shape[0] if len(v.shape) else shape + 1
            self.shape = np.array([shape])

        self.obs_spec = specs.BoundedArray(shape=self.shape,
                                           dtype=np.uint8,
                                           minimum=0,
                                           maximum=255,
                                           name='observation')

    def transform_observation(self, time_step):
        if self.vision:
            obs = self.env.physics.render(*self.size, camera_id=self.camera)
            obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = np.array([], dtype=np.float32)
            for k, v in time_step.observation.items():
                obs = np.concatenate((obs, v), axis=0, dtype=np.float32) if len(v.shape) else np.append(obs, v)

        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self.env.reset()
        return self.transform_observation(time_step)

    def step(self, action):
        time_step = self.env.step(action)
        return self.transform_observation(time_step)

    def observation_spec(self):
        return self.obs_spec

    def action_spec(self):
        return self.env.action_spec()

    def __getattr__(self, name):
        return getattr(self.env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, action_repeat, seed, vision, height=84, width=84):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    # add renderings for classical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        env = RenderWrapper(env, vision, camera_id, size=(height, width))
    env = ExtendedTimeStepWrapper(env)
    return env
