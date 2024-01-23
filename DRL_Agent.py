import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import random
from torch import nn,tensor
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
from pathlib import Path


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
#MDP model (s, a, r, new_s)
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#DRL_Agent
def create_model():
    model = nn.Sequential(
      nn.Conv2d(2,4,(3,3), stride=1),
      nn.Conv2d(4,8,(3,3), stride=1),
      nn.Conv2d(8,16,(3,3), stride=1),
      nn.Flatten(),
      nn.Linear(64, 3)
    )
    return model

def select_action(state, env, Q, eps_threshold, device):
    sample = np.random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return Q(state).argmax(keepdim=True)
    else:
        return torch.tensor([[np.random.randint(env.mobility)]], device=device, dtype=torch.long)

def optimize_model(policy_Q, target_Q, optimizer, memory, BATCH_SIZE, GAMMA, device):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, \
                      batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_Q(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_Q(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_Q.parameters(), 100)
    optimizer.step()
    
def plot_reward(rewards, resp_marks=[], resp_values=[], result=False):
    durations_t = torch.tensor(rewards, dtype=torch.float)
    plt.clf()
    plt.plot(durations_t.numpy(), '.', alpha=0.3, label='Episode reward')
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99).fill_(durations_t.mean()), means))
        plt.plot(means.numpy(), label='Mean')
    if result:
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Train History')
        plt.legend()
        plt.savefig('temp.png')
    else:
        plt.title('Training...')
    if len(resp_marks) > 0:
        for i, r in enumerate(resp_marks):
            plt.axvline(r, ymax=0.48)
            plt.axvline(r, ymin=0.52)
            _, _, ymin, ymax = plt.axis()
            plt.text(r, ymin + (ymax - ymin) * 0.5, str(resp_values[i]), rotation='vertical')
    plt.draw()
    plt.gcf().canvas.flush_events()
    plt.pause(0.01)
    
def explore_rate(x, e0, e1, e_decay):
    return e1 + (e0-e1) * (1 - x / e_decay)
    
def response_episode(r, e0, e1, e_decay):
    return np.round(e_decay * np.log((e0-e1) / (1 - r - e1))) 
    
def train(env, policy_Q, target_Q, criterion, optimizer, memory, device, params):
    plt.ion()
    plt.figure(figsize=(12,6))
    if torch.cuda.is_available():
        num_episodes = params['N_EPS']
    else:
        num_episodes = params['N_EPS']
        
    
    rewards = []
    steps_done = 0
    stage = 0
    responsibility = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1]
    eps_marks = []
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get its state
        state, _ = env.reset()
        ep_reward = 0
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        eps_threshold = explore_rate(i_episode, params['EPS_START'], params['EPS_END'], params['EPS_DECAY'])
        if (1 - eps_threshold > responsibility[stage]):
            stage += 1
            eps_marks.append(i_episode)
            
        for t in count():
            action = select_action(state, env, policy_Q, eps_threshold, device)
            steps_done += 1
            observation, reward, terminated, _ = env.step(action.item())
            ep_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated
            
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_Q, target_Q, optimizer, memory, params['BATCH_SIZE'], params['GAMMA'], device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_Q.state_dict()
            policy_net_state_dict = policy_Q.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*params['TAU'] + target_net_state_dict[key]*(1-params['TAU'])
            target_Q.load_state_dict(target_net_state_dict)

            if done:
                rewards.append(ep_reward)
                if (i_episode) % params['REPORT'] == 0:
                    plot_reward(rewards, resp_marks=eps_marks, resp_values=responsibility[:stage])
                break
    print('Complete')
    plot_reward(rewards, result=True)
    plt.ioff()
    plt.show()
    torch.save(target_Q.state_dict(), params['VERSION']+'.pt')
    
    
def test(env, target_Q, device):
    plt.imshow(env.surface)
    for g in range(len(env.y_in)):
        track = []
        prec0 = 0
        state, pos = env.reset(g)
        track.append(pos)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = target_Q(state).argmax(keepdim=True)
            observation, reward, terminated, new_pos = env.step(action.item())
            track.append(new_pos)
            done = terminated
            if done:
                break
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        track = np.array(track)
        plt.plot(track[:, 1], track[:,0], '.-', label=f'gate {g}', alpha=0.5)
    plt.legend()
    plt.show()
    