import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Plane:
    def __init__(self, start: np.ndarray, end: np.ndarray):
        # position (current, previous, starting and planned), direction
        self.pos = start.copy()
        self.prev = start.copy()
        self.start = start.copy()
        self.dest = end.copy()
        self.v = 0
        # plane activity inside the enviroment
        # 0: plane is able to move and execute commands from agent
        # 1: plane finished the task (non-active and non-interacting with active agents)
        #-1: plane failed (non-active, but still interacts with agents (being an obstacle))
        self.status = 0
        # movement properties dx, dy
        self.vectors = np.array([[1,0], [1,1], [0,1],
                               [-1,1], [-1,0], [-1,-1],
                               [0,-1], [1,-1]])
        self.movements = np.array([[0, 2, 1], [1,1,1], [0,0,1], [-1,-1, 1], [0, -2, 1], [0,0,0]])
        self.len_v = len(self.vectors)
        self.mobility = len(self.movements)
        #rewards_dict
        self.rewards = {'non-taxi': -1, 
                        'wrong-gate': 0.25, 
                        'corr-gate': 1,
                        'inactive': 0, 
                        'ok': -0.01}

    def reset(self):
        self.pos = self.start.copy()
        self.prev = self.start.copy()
        self.v = 0
        self.status = 0

    def step(self, choice):
        if self.status != 0:
            return
        move = self.vectors[(self.v + self.movements[choice, 0]) % self.len_v]
        self.v = (self.v + self.movements[choice, 1]) % self.len_v
        self.prev[:] = self.pos
        self.pos += move[::-1] * self.movements[choice, 2]

    def save(self):
        self.start = self.pos

    def set_route(self, start, end):
        self.start = start.copy()
        self.dest = end.copy()

    def state(self, shape):
        image = np.zeros(shape)
        if self.status <= 0:
            image[self.pos[0], self.pos[1]] = 1
            direct = self.pos + self.vectors[self.v][::-1]
            if np.all(0 <= direct) and np.all(direct < shape):
                image[direct[0], direct[1]] = -1
        return image, self.pos

    def reward(self, permit):
        if self.status != 0:
            return self.rewards['inactive']
        if permit == 0: #non taxi-way
            self.status = -1
            return self.rewards['non-taxi']
        if self.pos[1] == self.dest[1]: #task done
            self.status = 1
            return (self.rewards['corr-gate'] if self.pos[0] == self.dest[0] else self.rewards['wrong-gate'])
        return self.rewards['ok']

    def __eq__(self, other):
        # objects stopped at the same point
        eq_full = np.all(self.pos == other.pos)
        # vertical intersection
        ver_inter = (self.pos[0] == other.pos[0]) \
                    and (self.prev[0] == other.prev[0])\
                    and (self.pos[1] == other.prev[1]) \
                    and (self.prev[1] == other.pos[1])
        # horizontal intersection
        ver_inter = (self.pos[1] == other.pos[1]) \
                    and (self.prev[1] == other.prev[1]) \
                    and (self.pos[0] == other.prev[0]) \
                    and (self.prev[0] == other.pos[0])
        # oncoming traffic
        aga_inter = np.all(self.pos == other.prev) and np.all(self.prev == other.pos)
        return (eq_full or ver_inter or ver_inter or aga_inter) \
               and self.status <= 0 and other.status <= 0
               

class Airport:
    

    def __init__(self, surface: np.ndarray, t0 = 0):
    
        # taxi-way base form and its properties
        self.surface = surface.copy()
        self.max_x = self.surface.shape[1] - 1
        self.y_in = np.argwhere(self.surface[:,0] == 1)[:,0]
        self.y_out = np.argwhere(self.surface[:,-1] == 1)[:,0]
        self.cur_map = surface.copy()
        # fleet properties
        self.fleet = []
        self.statuses = np.array([0,0,0])
        # time
        self.t0 = t0
        self.t = self.t0
        self.max_t = 12
        #rewards
        self.rewards = {'crash': -1,
                        'stop': -0.2}

    def closest_exit(self, y):
        return np.array([self.y_out[np.argmin(np.abs(self.y_out - y))], self.max_x])

    def add(self, k):
        assert k <= len(self.y_in), 'Переполнение входов'
        for i in range(k):
            self.fleet.append(Plane(np.array([self.y_in[i], 0]), self.closest_exit(self.y_in[i])))

    def reset(self, shock=False):
        self.t = self.t0
        for p in self.fleet:
            p.reset()
        if shock:
            for p in self.fleet:
                p.set_route(p.start, self.closest_exit(p.start))
        self.statuses[:] = 0
        self.statuses[1] = len(self.fleet)
        return self.state()
       
    def update_stat(self):
        self.statuses[:] = 0 
        for p in self.fleet:
            self.statuses[p.status + 1] += 1
          
    def result_code(self):
        if (self.statuses[1] == 0):
            if (self.statuses[0] == len(self.fleet)):
                return -2 #complete failure
            if (self.statuses[2] == len(self.fleet)):
                return 2 #complete success
            return 1 #semi-success
        if (self.t > self.max_t):
            return -1 #episode truncation
        return 0 #episode continues

    def state(self):
        # complete image of all agents
        # (len([4, map + (pos & dest)]), map.shape[0], map.shape[1])
        res = np.zeros((len(self.fleet), 2, *self.surface.shape))
        all_pos = np.zeros((len(self.fleet), 2))
        for i, p in enumerate(self.fleet):
            res[i,0] = self.surface.copy() #map
            res[i, 0, p.dest[0], p.dest[1]] = 10 #destination
            res[i,1], all_pos[i] = p.state(self.surface.shape) #position
        return res, all_pos

    def reward(self, choice):
        r = 0
        for i, p in enumerate(self.fleet):
            if p.status != 0:
                continue
            if choice[i] == 5:
                r += self.rewards['stop']
                continue
            for j, p1 in enumerate(self.fleet):
                if i == j:
                    continue
                if p == p1: # crash between planes
                    p.status = -1
                    r += self.rewards['crash']
            r0 = p.reward(self.surface[p.pos[0], p.pos[1]])
            r += r0
        return r

    def step(self, choice):
        self.t += 1
        # s_t+1
        for i, p in enumerate(self.fleet):
            p.step(choice[i])
        # reward
        r = self.reward(choice)
        # new state after reward calc
        new_state, pos = self.state()
        # Episode result codes
        self.update_stat()
        res = self.result_code()
        return new_state, r, res, pos