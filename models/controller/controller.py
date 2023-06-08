import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gymnasium as gym

from vae import FlexVae
from mdnrnn import MDNRNN
from common import CarDataset


#device = torch.device("cpu")
class Controller(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.actor = nn.Linear(latent_dim + hidden_dim, action_dim)
        
    def forward(self, x):
        #x = (x - x.min()) / (x.max() - x.min())
        actor = F.tanh(self.actor(x))
        return actor
    
    def update_params(self, weight, bias):
        self.actor.weight.data = weight
        self.actor.bias.data = bias

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    obs_transform = CarDataset.img_transform

    vae = FlexVae(32, (3,64, 64))
    vae.load_state_dict(torch.load('vae.pt'))

    rnn = MDNRNN(32, 3)
    rnn.load_state_dict(torch.load('rnn.pt'))

    controller = Controller(32, 3)
    episodes = 25
    env = gym.make("CarRacing-v2", render_mode="human")
    for episode in range(episodes):
        obs, _ = env.reset()
        obs = obs_transform(obs)
        done = False

        optim = torch.optim.Adam(controller.parameters(), lr=1e-3)

        while not done:
            mu, log_var = vae.encode(obs.view(1, 3, 64, 64))
            z = vae._sample(mu, log_var)
            steer, drive = controller(z)
            steer = steer.detach().cpu()[0][0]
            drive = drive.detach().cpu()[0]

            gas = drive[0]
            brake = drive[1]

            action = [steer, gas, brake]
            #print(action)
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            obs = obs_transform(obs)
            if terminated or truncated:
                done = True
            action = torch.tensor(action)
            rnn_in = torch.cat((z, action.view(1, -1)), dim=1)
            rnn_out = rnn.forward(rnn_in)
            pred_reward = rnn_out[-1]
            loss = torch.pow(pred_reward - reward, 2) * 100
            print(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

