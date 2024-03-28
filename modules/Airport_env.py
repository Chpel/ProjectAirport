import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Airport:
    class Plane:
        def __init__(self, start: np.ndarray, end: np.ndarray):
            # position, direction
            self.pos = start.copy()
            self.start = start.copy()
            self.dest = end.copy()
            self.v = 0
            # activity inside the enviroment
            # active plane is able to move and execute commands from agent
            self.active = True
            # visible plane is interactive with other (can be an obstacle)
            self.visible = True
            # movement properties dx, dy
            self.vectors = np.array([[1,0], [1,1], [0,1],
                                   [-1,1], [-1,0], [-1,-1],
                                   [0,-1], [1,-1]])
            self.movements = np.array([[0, 2, 1], [1,1,1], [0,0,1], [-1,-1, 1], [0, -2, 1], [0,0,0]])
            self.len_v = len(self.vectors)
            self.mobility = len(self.movements)
            #rewards_dict
            self.rewards = {'non-taxi': -2, 'wrong-gate': -1, 
                            'corr-gate': 5, 'inactive': 0, 
                            'ok': 0, 'reverse': -0.1}

        def reset(self, force_act=True, force_vis=True):
            self.pos = self.start.copy()
            self.v = 0
            self.active = force_act
            self.visible = force_vis

        def step(self, choice):
            if not self.active:
                return
            move = self.vectors[(self.v + self.movements[choice, 0]) % self.len_v]
            self.v = (self.v + self.movements[choice, 1]) % self.len_v
            self.pos += move[::-1] * self.movements[choice, 2]

        def save(self):
            self.start = self.pos

        def set_route(self, start, end):
            self.start = start.copy()
            self.dest = end.copy()

        def state(self, shape):
            image = np.zeros(shape)
            if self.visible:
                image[self.pos[0], self.pos[1]] = 1
                direct = self.pos + self.vectors[self.v][::-1]
                if np.all(0 <= direct) and np.all(direct < shape):
                    image[direct[0], direct[1]] = -1
            image[self.dest[0], self.dest[1]] = 10 #destination
            return image, self.pos

        def reward(self, permit):
            if not self.active:
                return self.rewards['inactive'], False
            if permit == 0: #non taxi-way
                self.active = False
                return self.rewards['non-taxi'], True
            if self.pos[1] == self.dest[1]: #task done
                self.active = False
                self.visible = False
                return (self.rewards['corr-gate'] if self.pos[0] == self.dest[0] else self.rewards['wrong-gate']), True
            return self.rewards['ok'] + (self.rewards['reverse'] if self.vectors[self.v][0] < 0 else 0), False

        def __eq__(self, other):
            return np.all(self.pos == other.pos) and self.visible and other.visible


    def __init__(self, surface: np.ndarray, k_fleet=1):
        # taxi-way base form and its properties
        self.surface = surface.copy()
        self.max_x = self.surface.shape[1] - 1
        self.y_in = np.argwhere(self.surface[:,0] == 1)[:,0]
        self.y_out = np.argwhere(self.surface[:,-1] == 1)[:,0]
        self.cur_map = surface.copy()
        # fleet properties
        self.fleet = []
        # time
        self.t = 0
        self.max_t = 15
        #rewards
        self.rewards = {'crash': -2, 'ok': 0.02, 'stop': -0.1}

    def closest_exit(self, y):
        return np.array([self.y_out[np.argmin(np.abs(self.y_out - y))], self.max_x])

    def add(self, k):
        assert k <= len(self.y_in), 'Переполнение входов'
        for i in range(k):
            self.fleet.append(self.Plane(np.array([self.y_in[i], 0]), self.closest_exit(self.y_in[i])))

    def reset(self, rand_start=False):
        self.t = 0
        if rand_start and len(self.fleet) < len(self.y_in):
            y0s = np.random.choice(self.y_in, len(self.fleet), replace=False)
            for i, y0 in enumerate(y0s):
                self.fleet[i].set_route(np.array([y0, 0]), self.closest_exit(y0))
        for p in self.fleet:
            p.reset()
        return self.c_state()

    def is_active(self):
        for i, p in enumerate(self.fleet):
            if p.active:
                return True
        return False

    def c_state(self):
        # complete image of all agents
        # (len([map + 4*(pos & dest)]), map.shape[0], map.shape[1])
        res = np.zeros((len(self.fleet)+1, *self.surface.shape))
        all_pos = np.zeros((len(self.fleet), 2))
        res[0] = self.surface.copy() #map
        for i, p in enumerate(self.fleet):
            res[i+1], all_pos[i] = p.state(self.surface.shape) #position
        return res, all_pos

    def dc_state(self):
        # list of images of all autonomous agents
        # (len.fleet, len([map, pos, other_pos]), map.shape[0], map.shape[1])
        res = np.zeros((len(self.fleet), 3, *self.surface.shape))
        all_pos = np.zeros((len(self.fleet), 2))
        for i, p in enumerate(self.fleet):
            all_pos[i] = p.pos.copy()
        for i, p in enumerate(self.fleet):
            res[i,0] = self.surface.copy()
            res[i,1], _ = p.state(self.surface.shape)
            res[i,1, p.dest, -1] = 2
            for j, o_p in enumerate(all_pos):
                res[i, 2, o_p[1], o_p[0]] = 1 if i != j and o_p.visible else 0
        return res, all_pos


    def c_reward(self, choice):
        # All or Nothing responsibilty
        # Ep_stop if every plane are inactive (by failure either success)
        r = 0
        for i, p in enumerate(self.fleet):
            if not p.active:
                continue
            if choice[i] == 5:
                r += self.rewards['stop']
                continue
            for j, p1 in enumerate(self.fleet):
                if i == j:
                    continue
                if p == p1: # crash between planes
                    p.active = False
                    r += self.rewards['crash']
            r0, err = p.reward(self.surface[p.pos[0], p.pos[1]])
            r += r0
        return r, not self.is_active()

    def dc_reward(self):
        # Minimal responsibilty
        # The episode when all planes become inactive
        rs = np.zeros(len(self.fleet))
        for i, p in enumerate(self.fleet):
            for j, p1 in enumerate(self.fleet):
                if i == j:
                    continue
                if p == p1 and p.active: # crash between planes
                    p.active = False
                    rs[i] += self.rewards['crash']
            rs[i] += p.reward(self.surface[p.pos[0], p.pos[1]])
        return rs, not self.is_active()

    def step(self, choice):
        self.t += 1
        # s_t+1
        for i, p in enumerate(self.fleet):
            p.step(choice[i])
        # reward
        r, ep_end = self.c_reward(choice)
        # new state after reward calc
        new_state, pos = self.c_state()
        #end of episode
        return new_state, r, ep_end or (self.t > self.max_t), pos