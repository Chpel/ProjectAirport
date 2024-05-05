from modules.TrainDQN import *

def test(env, target_Q, device):
    state, pos = env.reset()
    track = []
    rews = []
    track.append(pos)
    state = tensor(state, dtype=float32, device=device).unsqueeze(0)
    for t in count():
        action = target_Q(state).argmax(-1)
        observation, reward, terminated, new_pos = env.step(action.tolist()[0])
        rews.append(reward)
        track.append(new_pos)
        done = terminated
        if done:
            break
        state = tensor(observation, dtype=float32, device=device).unsqueeze(0)
    track = np.array(track)
    return track, rews
    
    
def show_trajectory(env, path, rs, ax=plt):
    ax.imshow(env.surface)
    for g in range(len(env.fleet)):
        print(path[:, g])
        ax.plot(path[:,g, 1], path[:,g,0], '.-', label=f'gate {g+1}', alpha=0.5)
    print(rs)
    
    
def animate_trajectory(env, path, rs):
    plt.ion()
    prev_p = None
    k = len(env.fleet)
    col_list = ['blue', 'orange', 'green', 'red'][:k]
    for points in path:
        plt.clf()
        plt.imshow(env.surface)
        if isinstance(prev_p, np.ndarray):
            plt.scatter(prev_p[:, 1], prev_p[:,0],c=col_list, alpha=0.3)   
        plt.scatter(points[:, 1], points[:,0],c=col_list)   
        prev_p = points.copy()
        plt.draw()
        plt.gcf().canvas.flush_events()
        plt.pause(1)
    plt.ioff()
    plt.show()
    
    
def gif_trajectory(env, path, rs):
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    ax.set_xlim([-0.5, env.surface.shape[1]-0.5])
    ax.imshow(env.surface)
    k = len(env.fleet)
    
    points=path[0]
    col_list = ['blue', 'orange', 'green', 'red'][:k]
    pscatt = ax.scatter(points[:, 1], points[:,0],c=col_list)  
    
    def animate(i):
        points = path[i]
        pscatt.set_offsets(points[:,::-1])
        return pscatt,

    ani = animation.FuncAnimation(fig, animate, repeat=True,
                                        frames=len(path), interval=500)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=2,
                                     metadata=dict(artist='Me'),
                                     bitrate=1800)
    ani.save('scatter.gif', writer=writer)

    plt.show()