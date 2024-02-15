import numpy as np
import matplotlib
import matplotlib.pyplot as plt
        
class Airport:
    class Plane:
        def __init__(self, start: np.ndarray, end: int):
            # position, direction
            self.pos = start.copy()
            self.start = start.copy()
            self.dest = end
            self.v = 0
            # movement properties dx, dy
            self.vectors = np.array([[1,0], [1,1], [0,1], 
                                   [-1,1], [-1,0], [-1,-1],
                                   [0,-1], [1,-1]])
            self.movements = np.array([[0, 2], [1,1], [0,0], [-1,-1], [0, -2]]) 
            self.len_v = len(self.vectors)
            self.mobility = len(self.movements)
        
        def reset(self):
            self.pos = self.start
            self.v = 0
            
        def step(self, choice):
            move = self.vectors[(self.v + self.movements[choice][0]) % self.len_v]
            self.v = (self.v + self.movements[choice][1]) % self.len_v
            self.pos += move[::-1]
            
        def save(self):
            self.start = self.pos
            
        def set_route(self, start, end):
            self.start = start.copy()
            self.dest = end
        
    
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
        
    def closest_exit(self, y):
        return self.y_out[np.argmin(np.abs(self.y_out - y))]
        
    def add(self, k):
        assert k <= len(self.y_in), 'Переполнение входов'
        for i in range(k):
            self.fleet.append(self.Plane(np.array([self.y_in[k], 0]), self.closest_exit(self.y_in[k])))
        
    def reset(self, rand_start=False):
        self.t = 0
        if rand_start and len(self.fleet) < len(self.y_in):
            y0s = np.random.choice(self.y_in, len(self.fleet), replace=False)
            for i, y0 in enumerate(y0s):
                self.fleet[i].set_route(np.array([y0, 0]), self.closest_exit(y0))
        for p in self.fleet:
            p.reset()
        return self.state()
            
    def state(self):
        image = np.append([self.surface], [np.zeros(self.surface.shape)], axis=0)
        all_pos = np.zeros((len(self.fleet), 2))
        for i, p in enumerate(self.fleet):
            image[1, p.pos[0], p.pos[1]] = 1
            #image[0, self.y_out, -1] = 1
            image[0, p.dest, -1] = 100
            all_pos[i] = p.pos
            direct = p.pos + p.vectors[p.v][::-1]
            if np.all(0 <= direct) and np.all(direct < self.surface.shape):
                image[1, direct[0], direct[1]] = -1
        return image, all_pos
        
    def reward(self):
        p = self.fleet[0] #SINGLE MODE
        if self.surface[p.pos[0], p.pos[1]] != 1: #non-taxi way cell
            return -5, True
        if p.pos[1] == self.max_x: #gate cell
            return (5 if p.pos[0] == p.dest else -1), True
        return 0, False
            
    def step(self, choice):
        self.t += 1
        #s_t+1
        p = self.fleet[0] #SINGLE MODE
        p.step(choice)
        new_state, pos = self.state()
        #reward
        r, ep_end = self.reward()
        #end of episode
        return new_state, r, ep_end or (self.t > 15), pos