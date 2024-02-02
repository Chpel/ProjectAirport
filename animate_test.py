from Airport_env import *
from DRL_Agent import *


params = {
    'VERSION': 'Free_Pilot_v1',
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

env = Airplane_v2(Main_surface)
target_Q=create_model(env.mobility)
target_Q.load_state_dict(torch.load(params['VERSION']+'.pt')['MODEL'])
device='cpu'


for g in range(len(env.y_in)):
    track_x = []
    track_y = []
    prec0 = 0
    state, pos = env.reset(g)
    plt.ion()
    track_x.append(pos[1])
    track_y.append(pos[0])
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = target_Q(state).argmax(keepdim=True)
        observation, reward, terminated, new_pos = env.step(action.item())
        track_x.append(new_pos[1])
        track_y.append(new_pos[0])
        done = terminated
        
        plt.clf()
        plt.imshow(env.cur_map)
        for o in env.y_out:
            plt.text(env.max_x - 0.25, o, str(env.cur_map[o, -1]))
        plt.plot(track_x, track_y, '.-', label=f'gate {g}', color='orange', alpha=0.5)
        plt.draw()
        plt.gcf().canvas.flush_events()
        plt.pause(0.5)
        
        if done:
            break
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    plt.ioff()
    plt.show()