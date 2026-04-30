import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import os


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def train(env_name="CartPole-v1", hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000):

    env = gym.make(env_name)

    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    logits_net = mlp([obs_dim] + hidden_sizes + [n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    
    def reward_to_go(rews):
        n = len(rews)
        rtgs = np.zeros_like(rews, dtype=np.float32)
        running_sum = 0
        for i in reversed(range(n)):
            running_sum += rews[i]
            rtgs[i] = running_sum
        return rtgs

   
    all_returns = []

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []

        obs, _ = env.reset()
        ep_rews = []

        while True:
            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret = sum(ep_rews)
                batch_rets.append(ep_ret)

              
                batch_weights += list(reward_to_go(ep_rews))

                obs, _ = env.reset()
                ep_rews = []

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        loss = compute_loss(
            torch.as_tensor(batch_obs, dtype=torch.float32),
            torch.as_tensor(batch_acts, dtype=torch.int32),
            torch.as_tensor(batch_weights, dtype=torch.float32)
        )
        loss.backward()
        optimizer.step()

        return loss, batch_rets

    for i in range(epochs):
        loss, rets = train_one_epoch()

        avg_return = np.mean(rets)
        all_returns.append(avg_return)  

        print(f"epoch: {i:3d} \t loss: {loss:.3f} \t return: {avg_return:.3f}")

 
    os.makedirs("results", exist_ok=True)
    np.save("results/rtg.npy", all_returns)


# run
print("\nRunning Reward-to-Go Policy Gradient...\n")
train()