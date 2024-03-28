from modules.DRL_Agent import *

def test(env, target_Q, device):
    state, pos = env.reset()
    track = []
    rews = []
    track.append(pos)
    state = tensor(state, dtype=float32, device=device).unsqueeze(0)
    for t in count():
        action = target_Q(state).argmax(-1)
        #print(track[-1], env.fleet[0].v, end=' ')
        observation, reward, terminated, new_pos = env.step(action.tolist()[0])
        rews.append(reward)
        track.append(new_pos)
        #print(action.item(), ' -> ', track[-1], env.fleet[0].v)
        done = terminated
        if done:
            break
        state = tensor(observation, dtype=float32, device=device).unsqueeze(0)
    track = np.array(track)
    return track, rews
    
    
def show_trajectory(env, path, rs):
    plt.imshow(env.surface)
    for g in range(len(env.fleet)):
        print(path[:, g])
        plt.plot(path[:,g, 1], path[:,g,0], '.-', label=f'gate {g+1}', alpha=0.5)
    print(rs)
    
    
def animate_trajectory(env, path, rs):
    plt.ion()
    prev_p = None
    for points in path:
        plt.clf()
        plt.imshow(env.surface)
        if isinstance(prev_p, np.ndarray):
            plt.scatter(prev_p[:, 1], prev_p[:,0],c=['blue', 'orange', 'green', 'red'], alpha=0.3)   
        plt.scatter(points[:, 1], points[:,0],c=['blue', 'orange', 'green', 'red'])   
        prev_p = points.copy()
        plt.draw()
        plt.gcf().canvas.flush_events()
        plt.pause(1)
    plt.ioff()
    plt.show()