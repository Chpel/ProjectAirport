from modules import *
from torch import load


cases = [
    (0, 2, 5, '11', '12'),
    (1, 2, 5, '21', '22'),
    (2, 5, 2, '32', '31'),
    (3, 5, 2, '42', '41'),
]

for i, exit0, exit1, base, shock in cases:

    params = {
        'VERSION': 'Pilot_'+base,
        'BATCH_SIZE': 500,
        'GAMMA': 0.99,
        'EPS_START': 0.9,
        'EPS_END': 0.05,
        'N_EPS': 5000,
        'EPS_DECAY': 4500,
        'REPORT': 500,
        'MEMORY': 10000,
        'LR': 1e-4,
        'TAU': 0.1
        }
        
    params2 = {
        'VERSION': 'Pilot_'+shock+'s',
        'BATCH_SIZE': 500,
        'GAMMA': 0.99,
        'EPS_START': 0.9,
        'EPS_END': 0.05,
        'N_EPS': 1500,
        'EPS_DECAY': 1000,
        'REPORT': 500,
        'MEMORY': 1000,
        'LR': 1e-4,
        'TAU': 0.2
    }
    
    params3 = {
        'VERSION': 'Pilot_'+base+'s',
        'BATCH_SIZE': 500,
        'GAMMA': 0.99,
        'EPS_START': 0.9,
        'EPS_END': 0.05,
        'N_EPS': 1500,
        'EPS_DECAY': 1000,
        'REPORT': 500,
        'MEMORY': 1000,
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
    
    
    
    manual_seed(1234)
    random.seed(1234)

    #base train
    env1 = Airport(Main_surface)
    env1.add(1)
    new_gate = np.array([env1.y_in[i], 0])
    env1.fleet[0].set_route(new_gate.astype(int), env1.closest_exit(new_gate[0]))
    policy_Q=DispatcherRL(env1.fleet[0].mobility)
    target_Q=DispatcherRL(env1.fleet[0].mobility)
    target_Q.load_state_dict(policy_Q.state_dict())
    criterion = nn.SmoothL1Loss()
    optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
    device = "cpu"
    memory = ReplayMemory(10000)
    train_rewards = []
    train_rewards, eps1, resp1 = train(env1, policy_Q, target_Q, 
                            criterion, optimizer, memory, 
                            device, params,train_rewards)
    train_rewards2 = train_rewards.copy()

                            
    #base test
    with no_grad():
        traj1, rew1 = test(env1, target_Q, device)
    show_trajectory(env1, traj1, rew1)
    plt.show()

    #broken1 map test
    Main_surface[exit0, -2:] = 0
    env2 = Airport(Main_surface)
    env2.add(1)
    shock_step = 6
    new_start = traj1[shock_step][0]
    env2.fleet[0].set_route(new_start.astype(int), env2.closest_exit(new_start[0]))
    with no_grad():
        traj2, rew2 = test(env2, target_Q, device)
    show_trajectory(env2, traj2, rew2)
    plt.show()
        
    # After-shock retrain    
    policy_Q=DispatcherRL(env2.fleet[0].mobility)
    target_Q1=DispatcherRL(env1.fleet[0].mobility)
    policy_Q.load_state_dict(target_Q.state_dict())
    target_Q1.load_state_dict(target_Q.state_dict())
    criterion = nn.SmoothL1Loss()
    optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
    memory = ReplayMemory(1000)

    retrain_rewards, eps2, resp2 = train(env2, policy_Q, target_Q1, 
                            criterion, optimizer, memory, 
                               device, params2, train_rewards)
    with no_grad():
        traj3, rew3 = test(env2, target_Q1, device)
    show_trajectory(env2, traj3, rew3)
    plt.show()

    with open('history'+base+shock+'.npy', 'wb') as f:
        np.save(f, np.array(retrain_rewards))
        
    #broken2 map test
    Main_surface[exit0, -2:] = 1
    Main_surface[exit1, -2:] = 0
    env2 = Airport(Main_surface)
    env2.add(1)
    shock_step = 6
    new_start = traj1[shock_step][0]
    env2.fleet[0].set_route(new_start.astype(int), env2.closest_exit(new_start[0]))
    with no_grad():
        traj2, rew2 = test(env2, target_Q, device)
    show_trajectory(env2, traj2, rew2)
    plt.show()
        
    # After-shock retrain    
    policy_Q=DispatcherRL(env2.fleet[0].mobility)
    target_Q2=DispatcherRL(env1.fleet[0].mobility)
    policy_Q.load_state_dict(target_Q.state_dict())
    target_Q2.load_state_dict(target_Q.state_dict())
    criterion = nn.SmoothL1Loss()
    optimizer = Adam(policy_Q.parameters(), lr=params['LR'])
    memory = ReplayMemory(1000)

    retrain_rewards, eps2, resp2 = train(env2, policy_Q, target_Q2, 
                            criterion, optimizer, memory, 
                               device, params3, train_rewards2)
    with no_grad():
        traj3, rew3 = test(env2, target_Q2, device)
    show_trajectory(env2, traj3, rew3)
    plt.show()

    with open('history'+base+base+'.npy', 'wb') as f:
        np.save(f, np.array(retrain_rewards))






