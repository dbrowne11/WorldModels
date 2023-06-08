import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym
import torch
import time, timeit
import pygame

import pandas as pd
import matplotlib.pyplot as plt
# """I train on a single gpu system with very limited memory, cpu parallelizing the environment stuff, and batching the gpu calculations as much as possible are goals
# additionally a pa"""


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_action(z, controller):
    with torch.no_grad():
        #print(z)
        #controller.to(device)
        # mu, log_var = visual.encode(obs_transform(obs).view(-1, 3, 64, 64))
        # z = visual._sample(mu, log_var)
        action = controller(z)
        action[1:] = np.abs(action[1:])
        action_loss = 0
        if (action[1] > 0 and 1e-3 < torch.abs(action[0])) or action[2] > 0:
           #action[0] = 0
            action_loss = 1
        #action[-1] = 0
        #print(action.numpy())
        return action.numpy(), action_loss
    
from torchvision import transforms
im_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.ToTensor(),
            transforms.Resize((64, 64)),
            #transforms.Normalize((0,), (1,)),
            ])
def encode_obs(obs):
    obs = torch.tensor(obs, dtype=torch.float32)
    obs = im_transform(obs.view(-1, 3, 96, 96)).to(device)
    obs = obs / 255.0
    mu, log_var = visual.encode(obs)
    return visual._sample(mu, log_var), (mu, log_var)

def encode_decode_obs_imgs(obs):
    z, (mu, log_var) = encode_obs(obs)
    obs = torch.tensor(obs, dtype=torch.float32)
    obs = im_transform(obs.view(-1, 3, 96, 96)).to(device)
    recons = visual.decode(z)
    return obs, recons, mu, log_var

def prepare_obs(obs):
    obs = torch.tensor(obs, dtype=torch.float32)
    obs = im_transform(obs.view(-1, 3, 96, 96))
    obs = obs / 255.0
    return obs.to(device)

# Works on global params, es, and updates controller weights, returns params
def sample_controller(controller):
    params = es.ask()
    weight = params[:-3].reshape(OUTPUT_SIZE, LATENT_DIM)
    bias = params[-3:]
    #print(torch.tensor())
    controller.update_params(torch.tensor(weight, dtype=torch.float32), torch.tensor(bias,dtype=torch.float32))
    controller.to(device)
    return params

if __name__ == "__main__":
    ### Imports ###
    from common import CarDataset
    from vae import FlexVae
    from mdnrnn import MDNRNN
    from controller import Controller
    from cmaes import CMA
    from Envs.Parallel_Env_Wrapper import ParallelEnvWrapper as ParallelEnv
    ### Hyperparams ###
    INPUT_SIZE = (3, 64, 64)
    OUTPUT_SIZE = 3
    LATENT_DIM = 32
    HIDDEN_DIM = 0

    POP_SIZE = 64

    GENERATIONS = 30
    NUM_TRIALS = 2
    ### model loading ###
    visual = FlexVae(LATENT_DIM, INPUT_SIZE)
    visual.load_state_dict(torch.load('models/vae_hist/8vae.pt'))
    visual.to(device)

    memory = MDNRNN(LATENT_DIM, OUTPUT_SIZE)
    memory.load_state_dict(torch.load('rnn.pt'))
    memory.to(device)


    mp.set_start_method('spawn')

    #cart_pole_executer = ParallelEnvExecutor(POP_SIZE, 'CartPole-v1')

    sample_env = gym.make('CarRacing-v2')
    env_executer = ParallelEnv(POP_SIZE, 'CarRacing-v2', render_mode = 'tiling',
                               render_scale=96,  max_threads=12)
    print("initialized env executor")

    es = CMA(mean=np.random.uniform(0,1,(99)), sigma=0.5, population_size = POP_SIZE)
    es.__setstate__(torch.load('es_state.pt'))
    #es._popsize = POP_SIZE

    obs_transform = CarDataset.img_transform

    params = np.empty((POP_SIZE, 99), dtype=np.float32)
    controllers = [Controller(LATENT_DIM, HIDDEN_DIM, OUTPUT_SIZE) for _ in range(POP_SIZE)]
    for i, controller in enumerate(controllers):
        params[i] = sample_controller(controller)
        # #controller.load_state_dict(torch.load("models/controller_hist/c14024.pt"))
        # controller.to(device)
    
    max_steps = 250
    visual_optim = torch.optim.Adam(visual.parameters(), lr=1e-3)
    reward_hist = {'mean': [], 'max': [], 'min': []}
    for gen in range(GENERATIONS):
        #max_steps += gen
        print(f"Generation {gen} starting...")
        cumulative_reward = np.zeros((POP_SIZE))
        #num_trials = min(gen, NUM_TRIALS)
        max_steps += gen
        for trial in range(NUM_TRIALS):
            trial_start_time = time.time()
            obs, info = env_executer.reset(options={'random_start':False})
            models_time = 0
            steps_time = 0
            render_time = 0

            for step in range(0, min(max_steps, 1500), 1):
                print(f"step {step}")
                models_start_time = time.time()
                for _ in range(2):
                    z, (mu, log_var) = encode_obs(obs)

                    actions = np.stack([controller(z[j]).detach().cpu() for j, controller in enumerate(controllers)])
                    obs, rewards, terminated, truncated, info = env_executer.step(actions)
                    cumulative_reward += rewards
                
                if np.all(np.logical_and(terminated,truncated) == True):
                    print("breaking")
                    break
                env_executer.render_env(obs)

            trial_end_time = time.time()
            print(f"Timings:: Trial time={(trial_end_time - trial_start_time)}")
            #print(f"act_time={models_time / max_steps},step time={steps_time / max_steps}, render time={render_time / max_steps}")
            print(f"{max_steps} steps completed. Best Reward: {cumulative_reward.max()}, Mean Reward: {cumulative_reward.mean()}")
        # mean cumulative reward
        cumulative_reward = cumulative_reward / NUM_TRIALS
        reward_hist['mean'].append(cumulative_reward.mean())
        reward_hist['max'].append(cumulative_reward.max())
        reward_hist['min'].append(cumulative_reward.min())
        
        weight_loss = np.sum(params, axis=1)    # this should be a regularizer
        es_fitness = -1 * (cumulative_reward - 0.0001 *  weight_loss)
        print(params.shape, es_fitness.shape)
        solns = [(param, reward) for (param, reward) in zip(params, es_fitness)]
        print(len(solns))
        es.tell([(param, reward) for (param, reward) in zip(params, -cumulative_reward)])
        for i, controller in enumerate(controllers):
            params[i] = es.ask()
            weight = params[i][:-3].reshape(OUTPUT_SIZE, LATENT_DIM)
            bias = params[i][-3:]
            controller.update_params(torch.tensor(weight), torch.tensor(bias))
            controller.to(device)
            if gen % 10 == 0:
                torch.save(controller.state_dict(), f'models/controller_hist/c{gen}{i}.pt')
                torch.save(es.__getstate__(), 'es_state.pt')
        print("updated network parameters")

    env_executer.close()
    
    reward_hist = pd.DataFrame(reward_hist)
    reward_hist.to_csv('model_train_hist.csv')
    torch.save(es.__getstate__(), 'es_state.pt')
    plt.plot(reward_hist["mean"], label="Mean reward")
    plt.plot(reward_hist["max"], label = "Max reward")
    plt.plot(reward_hist["min"], label="min")
    plt.legend()
    plt.savefig('rew_hist.png')
