from modules import *
from torch import load

cases = [
    (0, 2, '11', '12'),
    (1, 2, '21', '22'),
    (2, 5, '32', '31'),
    (3, 5, '42', '41')
]

for i, exit0, base, shock in cases:

    params = {
        'base': 'Pilot_'+base,
        'shock': 'Pilot_'+shock+'s',
        'trainhis': 'history'+str(i+1),
        'res_scheme': 'shock_scheme'+str(i+1),
        'res_his': 'full_history'+str(i+1),
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

    env1 = Airport(Main_surface)
    k_planes = 1
    env1.add(k_planes)
    new_gate = np.array([env1.y_in[i], 0])
    env1.fleet[0].set_route(new_gate.astype(int), env1.closest_exit(new_gate[0]))
    target_Q=DispatcherRL(env1.fleet[0].mobility, 1, k_planes)
    target_Q.load_state_dict(load('models/'+params['base']+'.pt')['MODEL'])
    device='cpu'
    traj1, rew1 = test(env1, target_Q, device)



    Main_surface[exit0, -2:] = 0
    env2 = Airport(Main_surface)
    env2.add(1)
    shock_step = 6
    new_start = traj1[shock_step][0]
    env2.fleet[0].set_route(new_start.astype(int), env2.closest_exit(new_start[0]))
    traj2, rew2 = test(env2, target_Q, device)


    target_Q.load_state_dict(load('models/'+params['shock']+'.pt')['MODEL'])
    traj3, rew3 = test(env2, target_Q, device)


    fig, ax = plt.subplots(1,3,figsize=(15,6))

    cases = [
            (0,'Base experiment', env1, traj1, rew1),
            (1,'Before retraining', env2, np.append(traj1[:shock_step+1], traj2[1:], axis=0), np.append(rew1[:shock_step], rew2, axis=0)),
            (2,'After retraining', env2, np.append(traj1[:shock_step+1], traj3[1:], axis=0), np.append(rew1[:shock_step], rew3, axis=0))
            ]

    for i, title, env, traj, rew in cases:
        show_trajectory(env, traj, rew, ax=ax[i])
        ax[i].set_title(title)
    ax[1].plot(new_start[1], new_start[0], 'x', c='r', markersize=20)
    ax[2].plot(new_start[1], new_start[0], 'x', c='r', markersize=20)
    plt.tight_layout()
    plt.savefig(params['res_scheme']+'.png')
    plt.show()


"""
plt.figure(figsize=(15,6))
retrain_rewards = np.load(params['trainhis']+'.npy')
plot_reward(retrain_rewards, result=True, interactive=False)
plt.axvline(load(params['base']+'.pt')['N_EPS'])
plt.savefig(params['res_his']+'.png')
plt.show()
"""