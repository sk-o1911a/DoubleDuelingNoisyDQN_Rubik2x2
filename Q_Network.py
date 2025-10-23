import torch
import torch.nn as nn
import torch.nn.functional as F
from Noisy_Linear import NoisyLinear

class QNetwork(nn.Module):
    def __init__(
            self,
            input_dim=144,
            hidden1=256,
            hidden2=1024,
            hidden3=512,
            hidden4=64,
            #hidden5=64,
            num_actions=12,
            dueling=True,
            noisy=True,
            sigma0=0.5,
    ):
        super().__init__()
        self.dueling = dueling
        self.noisy = noisy

        Linear = (lambda in_f, out_f: NoisyLinear(in_f, out_f, sigma0)) if noisy else nn.Linear

        # create 2 fully connected layers
        self.fc1 = Linear(input_dim, hidden1)
        self.fc2 = Linear(hidden1, hidden2)
        self.fc3 = Linear(hidden2, hidden3)
        self.fc4 = Linear(hidden3, hidden4)
        #self.fc5 = Linear(hidden4, hidden5)
        if dueling:
            self.V = Linear(hidden4, 1)  # Value stream V(s)
            self.A = Linear(hidden4, num_actions)  # Advantage stream A(s,a)

        self._init_weights_if_needed()

    # create initial weights when not using NoisyNet
    def _init_weights_if_needed(self):
        if not self.noisy:
            for m in self.modules():
                if isinstance(m, nn.Linear): #just for linear layers (fc layers)
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)


    # relu activation function and dueling + NoisyNet algorithm
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        if self.dueling:
            V = self.V(x)
            A = self.A(x)
            A = A - A.mean(dim=1, keepdim=True)
            return V + A
        else:
            return self.out(x)