from network import Encoder, Actor, Critic
import numpy as np
import torch
import torch.nn.functional as F
import utils


class SACAgent:
    def __init__(self, obs_shape, action_shape, device, feature_dim, hidden_dim,
                 ensemble, init_temperature, alpha_lr, actor_lr, encoder_lr, critic_lr,
                 critic_tau, critic_target_update_frequency, learnable_temperature, use_tb, vision):
        self.device = device
        self.ensemble = ensemble
        self.critic_tau = critic_tau
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature
        self.use_tb = use_tb
        self.vision = vision

        self.encoder = Encoder(obs_shape).to(device) if self.vision else None
        repr_dim = int(self.encoder.repr_dim if self.vision else obs_shape[0])

        self.actor = Actor(repr_dim, action_shape, feature_dim, hidden_dim).to(device)

        self.critic = [Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device) for _ in range(ensemble)]
        self.critic_target = [Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device) for _ in range(ensemble)]
        for i in range(ensemble):
            self.critic_target[i].load_state_dict(self.critic[i].state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature), requires_grad=True, device=self.device)
        self.target_entropy = -action_shape[0]

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr) if self.vision else None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = [torch.optim.Adam(self.critic[i].parameters(), lr=critic_lr) for i in range(ensemble)]
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.train()
        for i in range(ensemble):
            self.critic_target[i].train()

        self.max_Q = [0, 0]  # for categorical dataset

    def train(self, training=True):
        self.training = training
        if self.vision:
            self.encoder.train(training)
        self.actor.train(training)
        for i in range(self.ensemble):
            self.critic[i].train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0)) if self.vision else obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.mean if eval_mode else dist.sample()
        return action.cpu().detach().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs):
        metrics = dict()

        index = np.random.choice(self.ensemble)

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.sample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target[index](next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
            target_Q = reward + (discount * target_V)

        Q1, Q2 = [None] * self.ensemble, [None] * self.ensemble
        critic_loss = 0
        for i in range(self.ensemble):
            Q1[i], Q2[i] = self.critic[i](obs, action)
            critic_loss += F.mse_loss(Q1[i], target_Q) + F.mse_loss(Q2[i], target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            for i in range(self.ensemble):
                metrics[f'critic_q{i}_1'] = Q1[i].mean().item()
                metrics[f'critic_q{i}_2'] = Q2[i].mean().item()
                self.max_Q[0] = max(self.max_Q[0], torch.max(Q1[i]).item())
                self.max_Q[1] = max(self.max_Q[1], torch.max(Q2[i]).item())
            metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        if self.vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        for i in range(self.ensemble):
            self.critic_opt[i].zero_grad(set_to_none=True)
        critic_loss.backward()
        for i in range(self.ensemble):
            self.critic_opt[i].step()
        if self.vision:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q = 0
        for i in range(self.ensemble):
            Q1, Q2 = self.critic[i](obs, action)
            Q += torch.min(Q1, Q2)
        Q /= self.ensemble

        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_entropy'] = -log_prob.mean().item()

        # optimize the actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.learnable_temperature:
            alpha_loss = self.alpha * (-log_prob.mean() - self.target_entropy).detach()

            if self.use_tb:
                metrics['target_entropy'] = self.target_entropy
                metrics['alpha_loss'] = alpha_loss.item()
                metrics['alpha_value'] = self.alpha.item()

            self.log_alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.log_alpha_opt.step()

        return metrics

    def update(self, batch, step):
        metrics = dict()

        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        if self.vision:
            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        metrics.update(self.update_critic(obs, action, reward, discount, next_obs))
        metrics.update(self.update_actor(obs.detach()))

        if step % self.critic_target_update_frequency == 0:
            for i in range(self.ensemble):
                utils.soft_update_params(self.critic[i], self.critic_target[i], self.critic_tau)

        return metrics
