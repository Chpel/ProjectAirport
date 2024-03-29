from modules import *


params = {
    'VERSION': 'Dispatcher_v1',
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
env.add(4)
policy_Q=DispatcherRL(env.fleet[0].mobility, k_outputs=4)
target_Q=DispatcherRL(env.fleet[0].mobility, k_outputs=4)
target_Q.load_state_dict(policy_Q.state_dict())
criterion = nn.CrossEntropyLoss()
optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
device = "cpu"
memory = ReplayMemory(10000)

train(env, policy_Q, target_Q, criterion, optimizer, memory, device, params)
test(env, target_Q, device)