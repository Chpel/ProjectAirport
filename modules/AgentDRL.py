#DRL_Agent
from torch import nn,stack,cat

class DispatcherRL(nn.Module): #Single-trainer
    def __init__(self, k_actions, k_agents=1, k_outputs=1, device='cpu'):
        super(DispatcherRL, self).__init__()
        self.k_nns = k_agents
        self.k_actors = k_outputs
        self.k_actions = k_actions
        if k_agents == 1: #Dispatcher
            self.model = nn.Sequential( # 1xfleetx2x9x10
            nn.Conv3d(k_outputs,8,(2,3,3), stride=1, device=device), #1x8x1x7x8
            nn.ReLU(),
            nn.Flatten(2,3), #1x8x7x8
            nn.Conv2d(8,16,(3,3), stride=1, device=device), #1x16x5x6
            nn.ReLU(),
            nn.Conv2d(16,32,(3,3), stride=1, device=device), #1x32x3x5
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3), stride=1, device=device), #1x64x1x2
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(128, k_actions * k_outputs, device=device)) #1x128

        else:
            pass #Decentralized

    def forward(self, x): #Centralised only (yet)
        x = self.model(x)
        return x.unfold(1, self.k_actions, self.k_actions) #1xfleetxactions
        
class DispatcherRL_M(nn.Module): #MARL
    def __init__(self, k_actions, k_agents=1, k_outputs=1, device='cpu'):
        super(DispatcherRL_M, self).__init__()
        self.k_nns = k_agents
        self.k_actors = k_outputs
        self.k_actions = k_actions
        if k_agents == 1: #Dispatcher
            single_model = nn.Sequential( # 1xfleetx(2x9x10)
                nn.Conv2d(2,4,(3,3), stride=1, device=device), #1x8x1x7x8
                nn.ReLU(),
                nn.Conv2d(4,8,(3,3), stride=1, device=device), #1x16x5x6
                nn.ReLU(),
                nn.Conv2d(8,16,(3,3), stride=1, device=device), #1x32x3x5
                nn.ReLU(),
                nn.Conv2d(16,32,(3,3), stride=1, device=device), #1x64x1x2
                nn.ReLU(),
                nn.Flatten(0))

            dispatcher = nn.Sequential(
                nn.Linear(64 * k_outputs, k_actions * k_outputs, device=device)) #1x128

        else:
            pass #Decentralized

    def forward(self, x): #Centralised only (yet)
        x = cat([self.single_model(x0) for x0 in x])
        x = self.dispatcher(x)
        return x.unfold(1, self.k_actions, self.k_actions) #1xfleetxactions