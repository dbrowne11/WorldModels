import os
from copy import deepcopy
import torch.multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt

import time
from common import CarDataset

from models.vae.vae import FlexVae
from mdnrnn import MDNRNN
from controller import Controller
from world_model import WorldModel
from cmaes import CMA

from Envs.CarRacingVisualizer import CarRacingViz
from Envs.Parallel_Env_Wrapper import ParallelEnvWrapper

def reshape_params(params):
    weight = params[:-3].reshape(OUTPUT_SIZE, LATENT_DIM + HIDDEN_DIM)
    bias = params[-3:]
    return weight, bias

from torchvision import transforms
im_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            #transforms.Normalize((0,), (1,)),
            ])
def prepare_obs(obs):
    #print("preparing observation", obs.shape, obs.min(), obs.max())
    #obs = obs / 255.0
    #obs = torch.tensor(obs, dtype=torch.float32)
    obs = im_transform(obs)
    obs = (obs - obs.min()) / obs.max()
    #print("prepared observation", obs.shape, obs.min(), obs.max())
    return obs.view(-1, 3, 64, 64)

if __name__ == '__main__':
    INPUT_SIZE = (3, 64, 64)
    OUTPUT_SIZE = 3
    LATENT_DIM = 32
    HIDDEN_DIM = 0
    preprocess = CarDataset.img_transform

    visual = FlexVae(LATENT_DIM, INPUT_SIZE)
    visual.load_state_dict(torch.load('models/BestModels/9vae.pt'))

    # memory = MDNRNN(LATENT_DIM, OUTPUT_SIZE, HIDDEN_DIM)
    # memory.load_state_dict(torch.load('rnn.pt'))

    es = CMA(mean=np.random.uniform(0,1,(HIDDEN_DIM + LATENT_DIM) * 3 + 3), sigma=0.5, population_size = 24)
    es.__setstate__(torch.load('es_state.pt'))
    
    controller = Controller(LATENT_DIM, HIDDEN_DIM, OUTPUT_SIZE)
    w, b = reshape_params(es.ask())
    controller.update_params(torch.tensor(w, dtype=torch.float32), torch.tensor(b, dtype=torch.float32))
    model = WorldModel(visual, None, controller)
    model.load_state_dict(torch.load('models/BestModels/Controller/controller'))
    env = CarRacingViz(render_mode='human')
    #env = ParallelEnvWrapper(4, env=CarRacingViz)
    max_steps = 500
    generations = 1
    reward_hist = {'mean': [], 'max': [], 'min': []}
    gen = 0
    with torch.no_grad():
        while gen < generations:
            print(f"Generation {gen} starting")
            #obs = prepare_obs(obs)# / 255.0
            cumulative_reward = 0
            hidden = torch.zeros((1, 256))
            obs, _ = env.reset()
            steps = 0
            while steps < max_steps:
                o = prepare_obs(obs)
                #print(o.shape, o.min(), o.max())
                z, (mu, log_var) = model.encode_obs(prepare_obs(obs))
                recons = visual.decode(z)
                print(visual.GetLoss(prepare_obs(obs), recons, mu, log_var, 1))
                recons = recons.squeeze().permute(2, 1, 0)
                action = model._get_action(z)
                # action = model.actor(z)
                # action[1] = (self.action[1] + 1) / 2
                # action[2] = torch.clip(self.action[2], 0, 1)
                #print(z.shape, recons.shape, action.shape)
                for _ in range(2):
                    obs, r, term, trunc, info = env.step(action.detach().cpu().numpy(), z=z[0], recons=recons.detach().numpy())
                    cumulative_reward += (r)



                #preds, hidden = model.get_pred(z, torch.tensor(action.view(1,-1), dtype=torch.float32))
                if np.all(np.logical_and(term,trunc) == True):
                    #print("breaking")
                    break
                #env.render_env(obs)
                
                steps += 1