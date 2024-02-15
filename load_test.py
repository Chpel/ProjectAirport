from modules.Airport_env import *
from modules.DRL_Agent import *


params = {
    #'VERSION': 'Free_Pilot_v1',
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
test(env, target_Q, device)
