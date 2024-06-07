from modules import *

params = {
    'VERSION': 'Dispatcher_test',
    'BATCH_SIZE': 500,
    'GAMMA': 0.99,
    'EPS_START': 0.5,
    'EPS_END': 0.05,
    'N_EPS': 17000,
    'EPS_DECAY': 13000,
    'REPORT': 500,
    'MEMORY': 10000,
    'LR': 1e-4,
    'TAU': 0.1
    }

manual_seed(1234)
random.seed(1234)


fig, ax = plt.subplots(1,2, figsize=(12,6));

#randomization dynamics
x = np.linspace(0, params['N_EPS'], 100)
ax[0].plot(x, explore_rate_linear(x, params['EPS_START'], params['EPS_END'], params['EPS_DECAY']))
ax[0].axhline(params['EPS_END'], color='r')

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
ax[1].imshow(Main_surface)
plt.show()

env = Airport(Main_surface)
k_planes = 4
env.add(k_planes)
policy_Q=DispatcherRL_M(env.fleet[0].mobility, k_outputs=k_planes)
target_Q=DispatcherRL_M(env.fleet[0].mobility, k_outputs=k_planes)
target_Q.load_state_dict(policy_Q.state_dict())
# Compute Huber loss
criterion = nn.SmoothL1Loss()
#optimizer = Adagrad(policy_Q.parameters(), lr=params['LR'], lr_decay=1e-6)
optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
device = "cpu"
memory = ReplayMemory(params['MEMORY'])

train(env, policy_Q, target_Q, criterion, optimizer, memory, device, params)
with no_grad():
    traj1, rew1 = test(env, target_Q, device)
animate_trajectory(env, traj1, rew1)