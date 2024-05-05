from modules import *
from torch import load

params = {
    'VERSION': 'Dispatcher_3.5of4s1',
    }


fig, ax = plt.subplots(1,1, figsize=(7,6));

#map picture
Main_surface = np.array(
   [[0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,0,1,1,0,0],
    [0,0,0,0,1,1,1,1,0,0],
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
k_planes = 4
env.add(k_planes)
target_Q=DispatcherRL(env.fleet[0].mobility, 1, k_planes)
target_Q.load_state_dict(load(params['VERSION']+'.pt')['MODEL'])
device='cpu'

traj1, rew1 = test(env, target_Q, device)
gif_trajectory(env, traj1, rew1)

