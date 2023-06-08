import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, perciever, memory, actor, null_action=None):
        super().__init__()
        self.perciever = perciever
        self.memory = memory
        self.actor = actor
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #self.device = torch.device("cpu")
        self.action = null_action


    def forward(self, x):
        mu, log_var = self.perciever.encode(x)
        z = self.perciever._sample(mu, log_var)

        pred, (hidden, _) = self.memory(torch.cat((z, self.action.view(1, -1).to(self.device)), dim=1), ret_h_c=True)
        action = self.actor(z)
        #action = self.actor(torch.cat((z, hidden), dim=1))
        return action
    
    def _get_action(self, x):
        self.action = self.actor(x).squeeze()
        self.action[1] = (self.action[1] + 1) / 2
        self.action[2] = torch.clip(self.action[2], 0, 1)

        return self.action

    def update_actor(self, weights, bias):

        self.actor.update_params(weights.to(self.device), bias.to(self.device))

    def encode_obs(self, x):
        mu, log_var = self.perciever.encode(x)
        z = self.perciever._sample(mu, log_var)
        return z, (mu, log_var)
    
    def get_pred(self, z, action):
        x = torch.cat((z, action), dim=1)
        pred, (hidden, _) = self.memory(x, ret_h_c=True)
        return pred, hidden
    
    def get_action(self, z, hidden):
        x = torch.cat((z, hidden), dim=1)
        #x[1:] = torch.abs(x[1:])
        #x[-1] = 0
        return self._get_action(x)


