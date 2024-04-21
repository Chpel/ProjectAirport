from modules import *

params = {
    'VERSION': 'Pilot_42',
    'BATCH_SIZE': 500,
    'GAMMA': 0.99,
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'N_EPS': 5000,
    'EPS_DECAY': 4500,
    'REPORT': 500,
    'LR': 1e-4,
    'TAU': 0.1
    }
    
params2 = {
    'VERSION': 'Pilot_41s',
    'BATCH_SIZE': 500,
    'GAMMA': 0.99,
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'N_EPS': 1500,
    'EPS_DECAY': 1000,
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

#base train
env1 = Airport(Main_surface)
env1.add(1)
new_gate = np.array([env1.y_in[3], 0])
env1.fleet[0].set_route(new_gate.astype(int), env1.closest_exit(new_gate[0]))
policy_Q=DispatcherRL(env1.fleet[0].mobility)
target_Q=DispatcherRL(env1.fleet[0].mobility)
target_Q.load_state_dict(policy_Q.state_dict())
criterion = nn.CrossEntropyLoss()
optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
device = "cpu"
memory = ReplayMemory(10000)

train_rewards, eps1, resp1 = train(env1, policy_Q, target_Q, 
                        criterion, optimizer, memory, 
                        device, params)
                        
#base test
traj1, rew1 = test(env1, target_Q, device)

#broken map test
Main_surface[5, -2:] = 0
env2 = Airport(Main_surface)
env2.add(1)
shock_step = 6
new_start = traj1[shock_step][0]
env2.fleet[0].set_route(new_start.astype(int), env2.closest_exit(new_start[0]))
traj2, rew2 = test(env2, target_Q, device)


    
# After-shock retrain    
policy_Q=DispatcherRL(env2.fleet[0].mobility)
policy_Q.load_state_dict(target_Q.state_dict())
criterion = nn.CrossEntropyLoss()
optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
memory = ReplayMemory(1000)

retrain_rewards, eps2, resp2 = train(env2, policy_Q, target_Q, 
                        criterion, optimizer, memory, 
                           device, params2, train_rewards)
traj3, rew3 = test(env2, target_Q, device)


with open('history4.npy', 'wb') as f:
    np.save(f, np.array(retrain_rewards))






