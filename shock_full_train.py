from modules.Airport_env import *
from modules.DRL_Agent import *
from modules.tests import *

params = {
    'VERSION': 'Pilot_v1.5_test',
    'BATCH_SIZE': 200,
    'GAMMA': 0.99,
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'EPS_DECAY': 5000,
    'N_EPS': 5000,
    'REPORT': 500,
    'LR': 1e-4,
    'TAU': 0.2
    }


fig, ax = plt.subplots(1,1, figsize=(12,6));


#map picture
Main_surface = np.array(
   [[0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,0,1,1,0,0],
    [0,0,0,0,1,1,1,1,1,1],
    [1,1,1,1,1,0,1,1,0,0],
    [0,0,0,0,1,1,1,1,0,0],
    [1,1,1,1,1,1,0,1,1,1],
    [0,0,0,0,1,1,1,1,0,0],
    [1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]]
)
ax.imshow(Main_surface)
plt.show()

#base train
env = Airport(Main_surface)
env.add(1)
policy_Q=DispatcherRL(env.fleet[0].mobility)
target_Q=DispatcherRL(env.fleet[0].mobility)
target_Q.load_state_dict(policy_Q.state_dict())
criterion = nn.CrossEntropyLoss()
optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
device = "cpu"
memory = ReplayMemory(10000)

train_rewards = train(env, policy_Q, target_Q, 
                        criterion, optimizer, memory, 
                        device, params)
                        
#base test
traj1, rew1 = test(env, target_Q, device)
show_trajectory(env, traj1, rew1)
plt.title('Base experiment')
plt.legend()
plt.tight_layout()
plt.show()

#broken map test
Main_surface[2, -2:] = 0
env = Airport(Main_surface)
env.add(1)
shock_step = 6
new_start = traj1[shock_step][0]
env.fleet[0].set_route(new_start.astype(int), env.closest_exit(new_start[0]))

traj2, rew2 = test(env, target_Q, device)
show_trajectory(env, np.append(traj1[:shock_step+1], traj2[1:], axis=0), np.append(rew1[:shock_step], rew2, axis=0))
plt.plot(new_start[1], new_start[0], 'x', c='r', markersize=20)
plt.title('Before retraining')
plt.legend()
plt.tight_layout()
plt.show()


params = {
    'VERSION': 'Pilot_1_2_shock',
    'BATCH_SIZE': 200,
    'GAMMA': 0.99,
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'EPS_DECAY': 400,
    'N_EPS': 2000,
    'REPORT': 200,
    'LR': 1e-4,
    'TAU': 0.2
    }
    
    
policy_Q=DispatcherRL(env.fleet[0].mobility)
policy_Q.load_state_dict(target_Q.state_dict())
criterion = nn.CrossEntropyLoss()
optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
memory = ReplayMemory(1000)

retrain_rewards = train(env, policy_Q, target_Q, 
                        criterion, optimizer, memory, 
                           device, params, train_rewards)
                           
durations_t = tensor(retrain_rewards, dtype=float)
plt.plot(durations_t.numpy(), '.', alpha=0.3, label='Episode reward')
means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
means = cat((zeros(99).fill_(durations_t.mean()), means))
plt.plot(means.numpy(), label='Mean')
plt.axvline(5000)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Train History')
plt.legend()
plt.savefig('full_history.png')

with open('history.npy', 'wb') as f:
    np.save(f, np.array(retrain_rewards))






