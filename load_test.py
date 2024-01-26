from Airport_env import *
from DRL_Agent import *


params = {
    'VERSION': 'Navigator_v2',
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

env = Airplane_v1(Main_surface)
target_Q=create_model()
target_Q.load_state_dict(torch.load(params['VERSION']+'.pt')['MODEL'])
device='cpu'
test(env, target_Q, device)
