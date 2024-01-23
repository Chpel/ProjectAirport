import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#enviroment
class Airport:
    def __init__(self, surface):
        self.x = 0
        self.y = 0
        self.movements = np.array([[1,0], [1,1], [1,-1]])
        self.mobility = len(self.movements)
        self.set_map(surface)

    def reset(self, gate=None):
        self.x = 0
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

    def step(self, choice=None):
        self.x += self.movements[choice][0]
        self.y += self.movements[choice][1]
        #clamp
        self.y = np.clip(self.y, 0, self.max_y)
        #s_t+1
        new_state, pos = self.state()
        #reward
        r = self.reward(choice)
        #end of episode
        ep_end = self.x == self.max_x or not self.Q_permit()
        return new_state, r, ep_end, pos