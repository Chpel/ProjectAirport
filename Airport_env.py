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
    def __init__(self, surface, discount=0.95):
        self.x = 0
        self.y = 0 #spacial state
        self.v = 0 #direction
        self.t = 0 #moment of time
        self.d = discount
        self.vectors = np.array([[1,0], [1,1], [0,1], 
                                   [-1,1], [-1,0], [-1,-1],
                                   [-1,0], [1,-1]])
        self.movements = np.array([0, 2], [1,1], [0,0], [-1,-1], [0, -2]) 
        self.mobility = len(self.movements)
        self.set_map(surface)

    def reset(self, gate=None):
        self.x = 0
        self.v = 0
        self.t = 0
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
            r -= 2*self.max_x
        elif self.x == self.max_x:
            r += self.max_x
        return r

    def Q_permit(self):
        return self.surface[self.y, self.x] == 1

    def step(self, choice):
        self.t += 1
        move = self.vectors[(self.v + self.movements[choice][0]) % self.mobility]
        self.v = (self.v + self.movements[choice][1]) % self.mobility
        self.x += move[0]
        self.y += move[1]
        #s_t+1
        new_state, pos = self.state()
        #reward
        r = self.reward(choice)
        #end of episode
        ep_end = (self.x == self.max_x) or (not self.Q_permit()) or (self.t > 2 * self.max_x)
        return new_state, r ** max(t - self.max_x, 0), ep_end, pos