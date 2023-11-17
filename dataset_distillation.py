import numpy as np
import torch
import torch.nn.functional as F
import utils


class DatasetDistillation:
    def __init__(self, num, obs_shape, act_shape, action_repeat, discount, lr, device):
        self.num = num
        self.step_reward = action_repeat * 1
        self.device = device

        self.obs = torch.tensor(np.random.rand(num, *obs_shape),
                                dtype=torch.float, requires_grad=True, device=device)
        self.action = torch.tensor(np.random.rand(num, *act_shape),
                                   dtype=torch.float, requires_grad=True, device=device)
        self.reward = torch.tensor(np.linspace(0, 6, num + 1)[:-1].reshape(-1, 1),
                                   dtype=torch.float, requires_grad=False, device=device)
        self.discount = torch.tensor(np.full(shape=(num, 1), fill_value=discount),
                                     dtype=torch.float, requires_grad=False, device=device)
        self.next_obs = torch.tensor(np.random.rand(num, *obs_shape),
                                     dtype=torch.float, requires_grad=True, device=device)

        self.opt = torch.optim.Adam([self.obs, self.action, self.next_obs], lr=lr)

    def get_data(self, batch):
        reward = batch[2].copy()
        index = reward.reshape(-1)
        for i in index:
            i = np.floor(i / self.step_reward * self.num) if i != self.step_reward else self.num - 1
        obs_ = self.obs[index]
        action_ = self.action[index]
        reward_ = self.reward[index]
        discount_ = self.discount[index]
        next_obs_ = self.next_obs[index]

        return obs_, action_, reward_, discount_, next_obs_

    def update(self, agent, batch):
        metrics = dict()

        obs_sync, action_sync, reward_sync, discount_sync, next_obs_sync = self.get_data(batch)
        if agent.vision:
            obs_sync = agent.encoder(obs_sync)
            next_obs_sync = agent.encoder(next_obs_sync)
        critic_grad_sync = self.critic_grad(agent, obs_sync, action_sync, reward_sync, discount_sync, next_obs_sync)
        actor_grad_sync = self.actor_grad(agent, obs_sync)

        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
        if agent.vision:
            obs = agent.encoder(obs)
            next_obs = agent.encoder(next_obs)
        critic_grad_real = self.critic_grad(agent, obs, action, reward, discount, next_obs)
        actor_grad_real = self.actor_grad(agent, obs)

        for i in critic_grad_real:
            i.detach()
        for i in actor_grad_real:
            i.detach()

        critic_loss = utils.match_loss(critic_grad_sync, critic_grad_real, self.device, "mse")
        actor_loss = utils.match_loss(actor_grad_sync, actor_grad_real, self.device, "mse")
        loss = critic_loss + actor_loss  # param to actor

        if agent.use_tb:
            metrics['DD_loss'] = loss.item()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return metrics

    @staticmethod
    def critic_grad(agent, obs, action, reward, discount, next_obs):
        dist = agent.actor(next_obs)
        next_action = dist.sample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = agent.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - agent.alpha * log_prob
        target_Q = reward + (discount * target_V)

        Q1, Q2 = agent.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if agent.vision:
            agent.encoder_opt.zero_grad(set_to_none=True)
        agent.critic_opt.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(critic_loss, list(agent.critic.parameters()), create_graph=True)
        if agent.vision:
            grad += torch.autograd.grad(critic_loss, list(agent.encoder.parameters()), create_graph=True)

        return grad

    @staticmethod
    def actor_grad(agent, obs):
        dist = agent.actor(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = agent.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = (agent.alpha.detach() * log_prob - Q).mean()

        agent.actor_opt.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(actor_loss, list(agent.actor.parameters()), create_graph=True)
        return grad
