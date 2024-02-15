from modules.Airport_env import *
from modules.DRL_Agent import *

params = {
    'VERSION': 'Navigator_v2_4000',
    }

fig, ax = plt.subplots(1,1, figsize=(7,6));

#map picture
Main_surface = np.array(
    [[1,1,1,1,1,0,1,1,0,0],
    [0,0,0,0,1,1,1,1,1,1],
    [1,1,1,1,1,0,1,1,0,0],
    [0,0,0,0,1,1,1,1,0,0],
    [1,1,1,1,1,1,0,1,1,1],
    [0,0,0,0,1,1,1,1,0,0],
    [1,1,1,1,1,0,0,0,0,0]]
)
ax.imshow(Main_surface)
plt.show()

#old map test
env = Airport(Main_surface)
target_Q=create_model()
target_Q.load_state_dict(torch.load(params['VERSION']+'.pt'))
device='cpu'
test(env, target_Q, device)

#broken map test
Main_surface[4, -2:] = 0
env = Airport(Main_surface)
target_Q=create_model()
target_Q.load_state_dict(torch.load(params['VERSION']+'.pt'))
test(env, target_Q, device)

#broken map re-train
params = {
    'VERSION': 'Navigator_v2Su_2000',
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
    
fig, ax = plt.subplots(1,2, figsize=(12,6));

#randomization dynamics
x = np.linspace(0, params['N_EPS'], 100)
ax[0].plot(x, params['EPS_END'] + (params['EPS_START'] - params['EPS_END']) * \
            np.exp(-1. * x / params['EPS_DECAY']))
ax[0].axhline(params['EPS_END'], color='r')

#map picture
ax[1].imshow(Main_surface)
plt.show()
    
policy_Q=create_model()
policy_Q.load_state_dict(target_Q.state_dict())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(policy_Q.parameters(), lr=params['LR'])
memory = ReplayMemory(10000)

train(env, policy_Q, target_Q, criterion, optimizer, memory, device, params)

#broken map test
target_Q=create_model()
target_Q.load_state_dict(torch.load(params['VERSION']+'.pt'))
test(env, target_Q, device)