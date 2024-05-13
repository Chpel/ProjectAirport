from modules import *
from torch import load

params = {
    'VERSION_from': 'Dispatcher_4of4',
    'VERSION': 'Dispatcher_4of4s',
    'BATCH_SIZE': 500,
    'GAMMA': 0.99,
    'EPS_START': 0.3,
    'EPS_END': 0.05,
    'N_EPS': 5000,
    'EPS_DECAY': 4500,
    'REPORT': 500,
    'LR': 1e-4,
    'TAU': 0.1
    }
    
manual_seed(1234)
random.seed(1234)

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
env.add(k_planes,True)
target_Q=DispatcherRL_M(env.fleet[0].mobility, 1, k_planes)
target_Q.load_state_dict(load('models/'+params['VERSION_from']+'.pt')['MODEL'])
device='cpu'

traj1, rew1 = test(env, target_Q, device)
#animate_trajectory(env, traj1, rew1)


Main_surface[2, -2:] = 0
shock_step = 5
env2 = Airport(Main_surface, shock_step)
env2.add(4, True)
for i, p in enumerate(env2.fleet):
    new_start = traj1[shock_step][i]
    p.set_route(new_start.astype(int), env2.closest_exit(new_start[0]))
    print(p.start, p.dest)
traj2, rew2 = test(env2, target_Q, device)
#animate_trajectory(env2, np.append(traj1[:shock_step+1], traj2[1:], axis=0), np.append(rew1[:shock_step], rew2, axis=0))


# After-shock retrain   
policy_Q=DispatcherRL_M(env2.fleet[0].mobility, 1, k_planes)
policy_Q.load_state_dict(target_Q.state_dict())
criterion = nn.SmoothL1Loss()
for p in policy_Q.sm.parameters():
    p.requires_grad = False
optimizer = Adam(filter(lambda p: p.requires_grad, policy_Q.parameters()), lr=params['LR'])
memory = ReplayMemory(5000)



retrain_rewards, eps2, resp2 = train(env2, policy_Q, target_Q, 
                        criterion, optimizer, memory, 
                           device, params)
                           

traj3, rew3 = test(env2, target_Q, device)
print(traj3)
animate_trajectory(env2, np.append(traj1[:shock_step+1], traj3[1:], axis=0), np.append(rew1[:shock_step], rew3, axis=0))

