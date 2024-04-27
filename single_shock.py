from modules import *
from torch import load

params = {
    'VERSION_from': 'Dispatcher_3.5of4',
    'VERSION': 'Dispatcher_3.5of4s1',
    'BATCH_SIZE': 500,
    'GAMMA': 0.99,
    'EPS_START': 0.9,
    'EPS_END': 0.01,
    'N_EPS': 2000,
    'EPS_DECAY': 1500,
    'REPORT': 500,
    'LR': 1e-4,
    'TAU': 0.2
    }

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
#plt.imshow(Main_surface)
#plt.show()

env = Airport(Main_surface)
k_planes = 4
env.add(k_planes)
target_Q=DispatcherRL(env.fleet[0].mobility, 1, k_planes)
target_Q.load_state_dict(load('models/'+params['VERSION_from']+'.pt')['MODEL'])
device='cpu'

traj1, rew1 = test(env, target_Q, device)
#animate_trajectory(env, traj1, rew1)


Main_surface[2, -2:] = 0
env2 = Airport(Main_surface)
env2.add(4)
shock_step = 6
for i, p in enumerate(env2.fleet):
    new_start = traj1[shock_step][i]
    p.set_route(new_start.astype(int), env2.closest_exit(new_start[0]))
traj2, rew2 = test(env2, target_Q, device)
#animate_trajectory(env2, np.append(traj1[:shock_step+1], traj2[1:], axis=0), np.append(rew1[:shock_step], rew2, axis=0))

# After-shock retrain    
policy_Q=DispatcherRL(env2.fleet[0].mobility, 1, k_planes)
policy_Q.load_state_dict(target_Q.state_dict())
criterion = nn.CrossEntropyLoss()
optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
memory = ReplayMemory(1000)

retrain_rewards, eps2, resp2 = train(env2, policy_Q, target_Q, 
                        criterion, optimizer, memory, 
                           device, params)
traj3, rew3 = test(env2, target_Q, device)
print(traj3)
#animate_trajectory(env2, np.append(traj1[:shock_step+1], traj3[1:], axis=0), np.append(rew1[:shock_step], rew3, axis=0))

