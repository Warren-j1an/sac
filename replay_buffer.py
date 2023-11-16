import collections
import datetime
import io
import numpy as np
import pathlib
import uuid


def episode_len(episode):
    return len(episode['action']) - 1


def count_episodes(directory):
    filenames = list(directory.glob('*.npz'))
    num_episodes = len(filenames)
    num_steps = sum(int(str(n).split('-')[-1][:-4]) - 1 for n in filenames)
    return num_episodes, num_steps


def load_episodes(directory, capacity=None, min_len=1):
    filenames = sorted(directory.glob('*.npz'))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split('-')[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue
        episodes[str(filename)] = episode
    return episodes


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = episode_len(episode)
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename


class Replay:
    def __init__(self, directory, capacity, min_len, max_len, discount, seed, ongoing=False, prioritize_ends=False):
        self.directory = pathlib.Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.min_len = min_len
        self.max_len = max_len
        self.discount = discount
        self.random = np.random.RandomState(seed=seed)
        self.ongoing = ongoing
        self.prioritize_ends = prioritize_ends
        self.data_spaces = ['observation', 'action', 'reward', 'discount']

        self.complete_eps = load_episodes(self.directory, capacity, min_len)
        self.ongoing_eps = collections.defaultdict(lambda: collections.defaultdict(list))
        self.total_episodes, self.total_steps = count_episodes(self.directory)
        self.loaded_episodes = len(self.complete_eps)
        self.loaded_steps = sum(episode_len(x) for x in self.complete_eps.values())

    @property
    def stats(self):
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'loaded_steps': self.loaded_steps,
            'loaded_episodes': self.loaded_episodes,
        }

    def add_step(self, transition, worker=0):
        episode = self.ongoing_eps[worker]
        for key in self.data_spaces:
            episode[key].append(transition[key])
        if transition.last():
            self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode):
        length = episode_len(episode)
        if length < self.min_len:
            print(f'Skipping short episode of length {length}.')
            return
        self.total_steps += length
        self.loaded_steps += length
        self.total_episodes += 1
        self.loaded_episodes += 1
        episode = {key: convert(value) for key, value in episode.items()}
        filename = save_episode(self.directory, episode)
        self.complete_eps[str(filename)] = episode
        self.enforce_limit()

    def dataset(self, batch, length):
        while True:
            obs, action, reward, discount, next_obs = [], [], [], [], []
            for _ in range(batch):
                episode = self.generate_chunks(length)
                obs.append(episode['observation'][:-1])
                action.append(episode['action'][1:])
                reward.append(episode['reward'][1:])
                next_obs.append(episode['observation'][1:])
            obs = np.concatenate(obs, axis=0)
            action = np.concatenate(action, axis=0)
            reward = np.expand_dims(np.concatenate(reward, axis=0), axis=1)
            next_obs = np.concatenate(next_obs, axis=0)
            discount = np.ones_like(reward) * self.discount
            yield (obs, action, reward, discount, next_obs)

    def generate_chunks(self, length):
        sequence = self.sample_sequence()
        chunk = collections.defaultdict(list)
        added = 0
        while added < length:
            needed = length - added
            adding = {k: v[:needed] for k, v in sequence.items()}
            sequence = {k: v[needed:] for k, v in sequence.items()}
            for key, value in adding.items():
                chunk[key].append(value)
            added += len(adding['action'])
            if len(sequence['action']) < 1:
                sequence = self.sample_sequence()
        chunk = {k: np.concatenate(v) for k, v in chunk.items()}
        return chunk

    def sample_sequence(self):
        episodes = list(self.complete_eps.values())
        if self.ongoing:
            episodes += [
                x for x in self.ongoing_eps.values()
                if episode_len(x) >= self.min_len]
        episode = self.random.choice(episodes)
        total = len(episode['action'])
        length = total
        if self.max_len:
            length = min(length, self.max_len)
        # Randomize length to avoid all chunks ending at the same time in case the
        # episodes are all the same length.
        length -= np.random.randint(self.min_len)
        length = max(self.min_len, length)
        upper = total - length + 1
        if self.prioritize_ends:
            upper += self.min_len
        index = min(self.random.randint(upper), total - length)
        sequence = {
            k: convert(v[index: index + length])
            for k, v in episode.items() if not k.startswith('log_')}
        sequence['is_first'] = np.zeros(len(sequence['action']), np.bool_)
        sequence['is_first'][0] = True
        if self.max_len:
            assert self.min_len <= len(sequence['action']) <= self.max_len
        return sequence

    def enforce_limit(self):
        if not self.capacity:
            return
        while self.loaded_episodes > 1 and self.loaded_steps > self.capacity:
            # Relying on Python preserving the insertion order of dicts.
            oldest, episode = next(iter(self.complete_eps.items()))
            self.loaded_steps -= episode_len(episode)
            self.loaded_episodes -= 1
            del self.complete_eps[oldest]


def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value
