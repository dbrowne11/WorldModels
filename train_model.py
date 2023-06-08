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

from vae import FlexVae
from mdnrnn import MDNRNN
from controller import Controller
from world_model import WorldModel
from cmaes import CMA

from Envs.CarRacingVisualizer import CarRacingViz
from Envs.Parallel_Env_Wrapper import ParallelEnvWrapper
os.environ['TORCH_USE_CUDA_DSA'] = "1"
process = CarDataset.img_transform

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

# Environment related params
INPUT_SIZE = (3, 64, 64)
OUTPUT_SIZE = 3
LATENT_DIM = 32
HIDDEN_DIM = 256

POP_SIZE = 49

NUM_TRIALS = 8


class WorldModelPop:
    def __init__(self, pop_size, visual, memory, controllers):
        self.pop_size = pop_size
        self.visual = visual
        self.memory = memory
        self.controllers = [deepcopy(controller) for controller in controllers]
        self.population = []
        for i in range(pop_size):
            self.population.append(WorldModel(visual, memory, self.controllers[i], null_action=torch.tensor((0, 0, 0))))

    def __iter__(self):
        for indv in self.population:
            yield indv

    def __len__(self):
        return self.pop_size

    def update_weights(self, params):
        for i in range(self.pop_size):
            w, b = params[i]
            #self.population[i].to("cpu")
            self.population[i].actor.update_params(torch.tensor(w, dtype=torch.float32).to(device), torch.tensor(b, dtype=torch.float32).to(device))

            #self.population[i].update_actor(w.to(device), b.to(device))

    def visual_encode_batch(self, x):
        mu, log_var = self.visual.encode(x)
        z = self.visual._sample(mu, log_var)

        return z, (mu, log_var)

    def predct_z_batch(self, zs, actions):
        x = torch.cat((zs, actions), dim=1)
        preds, hidden = [], []
        for i, world_model in enumerate(self.population):
            pred, h = world_model.get_pred(zs[i].view(1,-1), actions[i].view(1,-1))
            preds.append(pred)
            hidden.append(h)
        return torch.vstack(preds), torch.vstack(hidden)
    
    def get_pop_actions(self, x):
        actions = []
        for i, world_model in enumerate(self.population):
            actions.append(world_model._get_action(x[i]).detach().cpu())
        return np.vstack(actions)

def update_res_output(*args):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(*args)

    
from torchvision import transforms
im_transform = transforms.Compose([
            # transforms.ToPILImage(),
            #transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize((0,), (1,)),
            ])

def prepare_obs(obs):
    ob = []
    for o in obs:
        ob.append(im_transform(torch.tensor(o, dtype=torch.float32).view(3, 96, 96)))
    obs = torch.stack(ob)
    obs = (obs - obs.min()) / obs.max()
    return torch.stack(ob).to(device)


def reshape_params(params):
    weight = params[:-3].reshape(OUTPUT_SIZE, LATENT_DIM + HIDDEN_DIM)
    bias = params[-3:]
    return weight, bias

def draw_population(es, popsize):
    controllers = []
    all_params = []
    for i in range(popsize):
        params = es.ask()
        all_params.append(params)
        weight, bias = reshape_params(params)
        controller = Controller(LATENT_DIM, HIDDEN_DIM, OUTPUT_SIZE)
        controller.update_params(torch.tensor(weight, dtype=torch.float32), torch.tensor(bias, dtype=torch.float32))
        controllers.append(controller.to(device))
    return controllers, np.vstack(all_params)

def update_population(es, population: WorldModelPop):
    w_bs = []
    params = []
    for i in range(population.pop_size):
        params.append(es.ask())
        w_bs.append((reshape_params(params[i])))

    population.update_weights(w_bs)
    return np.vstack(params)

def build_controller_input(mode, z, h):
    if mode == 0:
        return z
    elif mode == 1:
        return torch.cat((z, h), dim=1)
    else:
        raise NotImplementedError("Only z and z+h currently supported")

INPUT_Z = 0
INPUT_H = 1

def initialize_models(args):

    visual = FlexVae(LATENT_DIM, INPUT_SIZE)
    if args['vision_path'] is not None:
        visual.load_state_dict(torch.load(args['vision_path']))
    visual.to(device)
    memory = None
    if args['memory_path'] is not None and args['input_mode'] != 0:
        memory = MDNRNN(LATENT_DIM, OUTPUT_SIZE, HIDDEN_DIM)    
        memory.load_state_dict(torch.load('rnn.pt'))
        memory.to(device)

    es = CMA(mean=np.random.uniform(0,1,(HIDDEN_DIM + LATENT_DIM) * 3 + 3), sigma=0.5, population_size = POP_SIZE)
    if args["es_path"] is not None:
        es.__setstate__(torch.load(args["es_path"]))

    initial_controllers, params = draw_population(es, POP_SIZE)

    pop = WorldModelPop(POP_SIZE, visual, memory, initial_controllers)
    return pop, es, params


def train(pop, env, es, params, args):

    mode = args['input_mode']
    env = ParallelEnvWrapper(POP_SIZE, env, render_mode = 'tiling',
                               render_scale=128,  max_threads=12)

    max_steps = 350
    generations = args['generations']
    reward_hist = {'mean': [], 'max': [], 'min': []}
    gen = 0
    best_score = 0
    with torch.no_grad():
        while gen < generations:
            max_steps += 1
            print(f"Generation {gen} starting")

            cumulative_reward = np.zeros((POP_SIZE))
            hidden = torch.zeros((POP_SIZE, 256)).to(device)

            for trial in range(NUM_TRIALS):
                obs, _ = env.reset({"random_start":True})
                steps = 0
                while steps < max_steps:
                    #print("step", steps)
                    z, (mu, log_var) = pop.visual_encode_batch(prepare_obs(obs))
                    x = build_controller_input(mode, z, hidden)
                    actions = pop.get_pop_actions(x)
                    for _ in range(1):
                        obs, r, term, trunc, info = env.step(actions)
                        cumulative_reward += (r)# + 1e-3 * actions[:, 2]

                    if mode != 0:
                        preds, hidden = pop.predct_z_batch(z, torch.tensor(actions, dtype=torch.float32).to(device))
                    if np.all(np.logical_and(term,trunc) == True):
                        break

                    env.render_env(obs)
                    steps += 1

            cumulative_reward = cumulative_reward / NUM_TRIALS
            reward_hist['mean'].append(cumulative_reward.mean())
            reward_hist['max'].append(cumulative_reward.max())
            reward_hist['min'].append(cumulative_reward.min())
            
            # save model stuff if better than current best
            if cumulative_reward.max() > best_score:
                best_score = cumulative_reward.max()
                index = np.argmax(cumulative_reward)
                best_controller = pop.population[index]
                torch.save(best_controller.state_dict(), "models/BestModels/Controller/controller.pt")
                torch.save(es.__getstate__(), f'models/BestModels/Controller/es_state.pt')

            es.tell([(param, reward) for (param, reward) in zip(params, -cumulative_reward)])
            params = update_population(es, pop)

            if gen % 10 == 0:
                torch.save(es.__getstate__(), f'models/Checkpoints/es_state{gen}.pt')
            print(f"gen: {gen} complete with max: {cumulative_reward.max()}, mean: {cumulative_reward.mean()}")
            gen += 1

    torch.save(es.__getstate__(), 'es_state.pt')
    plt.plot(reward_hist["mean"], label="Mean reward")
    plt.plot(reward_hist["max"], label = "Max reward")
    plt.plot(reward_hist["min"], label="min")
    plt.legend()
    plt.savefig('rew_hist.png')
    env.close()

if __name__ == '__main__':
    import argparse
    parser  = argparse.ArgumentParser()
    parser.add_argument('-g', '--generations', default=10, help="Number of generations to train for", type=int)
    parser.add_argument('-t', '--trials', default=2, help="number of trials per generation", type=int)
    parser.add_argument('--pop_size', default=49, help="population_size", type=int)
    parser.add_argument('--input_mode', default=0, help="Input mode to the controller", type=int)
    parser.add_argument('--vision_path', default='models/BestModels/9vae.pt', help="path to the saved vision model, must be a valid path")
    parser.add_argument('--memory_path', default='rnn.pt', help="Path to memory model, or None if input_mode == 0")
    parser.add_argument('--es_path', default=None, help="Path to the saved CMA-es state, will start fresh training if not specified or None")
    parser.add_argument('-z', '--latent_dim', default=32, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('-a', '--action_dim', default=3, type=int)

    args = vars(parser.parse_args())
    print(args)
    LATENT_DIM = args["latent_dim"]
    HIDDEN_DIM = int(args['hidden_dim'])
    OUTPUT_SIZE = args['action_dim']

    pop, es, params = initialize_models(args)
    try:
        train(pop, 'CarRacing-v2', es, params, args)
    except KeyboardInterrupt:
        torch.save(es.__getstate__(), 'es_state_interrupt.pt')
        exit()
    exit()
    print(args)
    