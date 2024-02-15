from modules.Airport_env import *
from modules.DRL_Agent import *


params = {
    'VERSION': 'Pilot_v1.5',
    }


fig, ax = plt.subplots(1,1, figsize=(7,6));

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

env = Airport(Main_surface)
env.add(1)
target_Q=create_model(env.fleet[0].mobility)
target_Q.load_state_dict(torch.load(params['VERSION']+'.pt')['MODEL'])
device='cpu'


for g in range(len(env.y_in)):
    track_x = []
    track_y = []
    prec0 = 0
    gate = env.y_in[g]
    env.fleet[0].set_route(np.array([gate, 0]), env.closest_exit(gate))
    state, pos = env.reset()
    plt.ion()
    track_x.append(pos[0][1])
    track_y.append(pos[0][0])
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = target_Q(state).argmax(keepdim=True)
        observation, reward, terminated, new_pos = env.step(action.item())
        track_x.append(new_pos[0][1])
        track_y.append(new_pos[0][0])
        done = terminated
        
        plt.clf()
        plt.imshow(observation[0])
        for o in env.y_out:
            plt.text(env.max_x - 0.25, o, str(observation[0][o, -1]))
        plt.plot(track_x, track_y, '.-', label=f'gate {g}', color='orange', alpha=0.5)
        plt.arrow(track_x[-1], track_y[-1], *(env.fleet[0].vectors[env.fleet[0].v] / 2), width=0.05)
        plt.xlim(-0.5, 9.5)
        plt.draw()
        plt.gcf().canvas.flush_events()
        plt.pause(0.5)
        
        if done:
            break
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    plt.ioff()
    plt.show()