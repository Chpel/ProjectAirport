import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#enviroment
class Airplane_v1:
    def __init__(self, surface):
        self.x = 0
        self.y = 0
        self.s = 0
        self.movements = np.array([[1,0], [1,1], [1,-1]])
        self.mobility = len(self.movements)
        self.set_map(surface)

    def reset(self, gate=None):
        self.x = 0
        self.s = 0
        if gate == None:
            self.y = np.random.choice(self.y_in)
        else:
            self.y = self.y_in[gate]
        return self.state()

    def set_map(self, surface):
        self.surface=surface.copy()
        self.max_x = self.surface.shape[1] - 1
        self.max_y = self.surface.shape[0] - 1
        self.y_in = np.argwhere(self.surface[:,0] == 1)[:,0]
        self.y_out = np.argwhere(self.surface[:,-1] == 1)[:,0]


    def state(self):
        flight_map = np.zeros(self.surface.shape)
        flight_map[self.y,self.x] = 1
        image = np.append([self.surface], [flight_map], axis=0)
        return image, (self.y, self.x)

    def reward(self, choice):
        r = 0
        #non-taxi way cell
        if not self.Q_permit():
            r -= 0.5 * self.max_x
        elif self.x == self.max_x:
            r += self.max_x
        elif choice != 0:
            r -= 1
        return r

    def Q_permit(self):
        return self.surface[self.y, self.x] == 1

    def step(self, choice=None):
        self.x += self.movements[choice][0]
        self.y += self.movements[choice][1]
        self.s += np.linalg.norm(self.movements[choice])
        #s_t+1
        new_state, pos = self.state()
        #reward
        r = self.reward(choice)
        #end of episode
        ep_end = self.x == self.max_x or not self.Q_permit()
        return new_state, r, ep_end, pos
        
class Airplane_v2:
    def __init__(self, surface):
        self.x = 0
        self.y = 0 #spacial state
        self.v = 0 #direction
        self.t = 0 #moment of time
        self.y_dest = None #y of the planned exit
        self.vectors = np.array([[1,0], [1,1], [0,1], 
                                   [-1,1], [-1,0], [-1,-1],
                                   [0,-1], [1,-1]])
        self.movements = np.array([[0, 2], [1,1], [0,0], [-1,-1], [0, -2]]) 
        self.len_v = len(self.vectors)
        self.mobility = len(self.movements) # = exit node of the NN
        
        self.surface = surface.copy() # base form of the taxi-way
        self.max_x = self.surface.shape[1] - 1
        self.max_y = self.surface.shape[0] - 1
        self.y_in = np.argwhere(self.surface[:,0] == 1)[:,0]
        self.y_out = np.argwhere(self.surface[:,-1] == 1)[:,0]
        self.cur_map = surface.copy()

    def reset(self, gate=None):
        self.x = 0
        self.v = 0
        self.t = 0
        if gate == None:
            self.y = np.random.choice(self.y_in)
        else:
            self.y = self.y_in[gate]
        self.y_dest = self.y_out[np.argmin(np.abs(self.y_out - self.y))]
        self.cur_map = self.surface.copy()
        self.cur_map[self.cur_map == 0] = -10 #non-taxi_way cell
        self.cur_map[self.y_dest, -1] = 10 #targeted exit
        self.cur_map[self.y_out[self.y_out != self.y_dest], -1] = 3 #non-targeted exit
        return self.state()

    def state(self):
        flight_map = np.zeros(self.surface.shape)
        flight_map[self.y,self.x] = 1
        direction = self.vectors[self.v]
        y_d = self.y+direction[1]
        x_d = self.x+direction[0]
        if 0 <= y_d < self.max_y and 0 <= x_d < self.max_x:
            flight_map[y_d, x_d] = -1
        image = np.append([self.cur_map], [flight_map], axis=0)
        return image, (self.y, self.x)

    def reward(self, choice):
        r = 0
        if not self.Q_permit(): #non-taxi way cell
            r -= self.max_x
        elif self.x == self.max_x:
            r += self.max_x
        else: 
            r += self.vectors[self.v,0] * self.vectors[0,0]
        return r

    def Q_permit(self):
        return self.surface[self.y, self.x] >= 0
        
    def Q_goal(self):
        return self.y == self.y_dest and self.x == self.max_x

    def step(self, choice):
        #self.cur_map[self.cur_map[:, self.x] > 0, self.x] = -1
        self.t += 1
        move = self.vectors[(self.v + self.movements[choice][0]) % self.len_v]
        self.v = (self.v + self.movements[choice][1]) % self.len_v
        self.x += move[0]
        self.y += move[1]
        #s_t+1
        new_state, pos = self.state()
        #reward
        r = self.cur_map[self.y, self.x]
        #end of episode
        ep_end = (self.x == self.max_x) or (r <= -10) or (self.t > 2 * self.max_x)
        return new_state, r, ep_end, pos