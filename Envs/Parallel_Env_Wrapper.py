import pygame
import gymnasium as gym
import numpy as np
import cv2
import torch.multiprocessing as mp
from math import isqrt

def env_worker(worker, remote, envs, work_num):
    # Initialize vars for each env in envs
    num_envs = len(envs)
    
    awaiting_reset = [True] * num_envs
    neg_reward_counter = [0] * num_envs
    step = [0] * num_envs
    cur_obs = [None] * num_envs
    cum_rewards = [0] * num_envs
    while True:

        cmd, data = worker.recv()

        if cmd == 'step':
            results = []
            for i, env in enumerate(envs):
                if awaiting_reset[i]:
                    reward, terminated, truncated = -0, True, True
                    results.append((cur_obs[i], reward, terminated, truncated, info))
                    continue

                obs, reward, terminated, truncated, info = env.step(data[i])
                cur_obs[i] = obs
                step[i] += 1
                cum_rewards[i] += reward
                if reward < 0:
                    neg_reward_counter[i] += 1
                else:
                    neg_reward_counter[i] = 0

                if terminated or truncated or neg_reward_counter[i] > 50:
                    awaiting_reset[i] = True
                    print(f"worker {work_num} env: {i} awaiting reset. {env.tile_visited_count} Tiles visited. {cum_rewards[i]} total reward" )
                    end_step = step[i]
                    #if step < 250:
                        #reward -= 100
                    #reward += ((env.tile_visited_count - 5) * 10 + (step[i] - 300) * 1e-3)
                    #reward += env.tile_visited_count 
                    #reward -= 10 / end_step 

                results.append((obs, reward, terminated, truncated, info))
            worker.send(results)
                

        if cmd == 'reset':
            results = []
            cum_rewards = [0] * num_envs
            for i, env in enumerate(envs):
                obs, info = env.reset(options=data)
                results.append((obs, info))
                awaiting_reset[i] = False
            #info["need_reset"] = False
                neg_reward_counter[i] = 0
            worker.send(results)

        if cmd == 'render':
            worker.send(env.render())

        if cmd == 'close':
            worker.close()
            break

class ParallelEnvWrapper:
    def __init__(self, pop_size, env_name=None, env=None,
                 render_mode='tiling', render_scale=96 * 2, 
                 max_threads=None):
        
        if env_name is None and env is None:
            raise Exception("Must set either env_name or env as the environment name \
                            or class reference respectively")

        self.waiting = False
        self.closed = False

        self.pop_size = pop_size
        if env is None:
            self.env_name = env_name
            self.envs = [gym.make(env_name, max_episode_steps=5000) for _ in range(self.pop_size)]
        else:
            self.envs = [env(render_mode='human') for _ in range(self.pop_size)]
        # Generally parallelization benefits flatten once you reach num_threads/processes > num_cores
        # Therefore, we include this param to allow for reducing synchronization
        if max_threads is not None:
            self.max_threads = max_threads
            self.num_threads = min(self.max_threads, self.pop_size)
        else:
            self.num_threads = self.pop_size
        
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_threads)])
        self.chunksize = int(np.ceil(self.pop_size / self.num_threads))
        self.render_mode = render_mode
        self.i = 0
        if render_mode is not None:
            pygame.init()
            self.render_scale = render_scale
            self.n_cols = isqrt(self.pop_size)
            self.n_rows = self.pop_size // self.n_cols + np.ceil((self.pop_size % self.n_cols) / self.n_cols) # 1 or 0 if no remainder
            self.render_frame_skip = 8
            self.display = pygame.display.set_mode((self.n_cols * self.render_scale, self.n_rows * self.render_scale ))


        # Generate processes
        self.processes = []
        proc_num = 0
        for i, (worker, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            start = proc_num * self.chunksize
            chunk = self.envs[start :start + self.chunksize]
            proc = mp.Process(target = env_worker, args = (worker, remote, chunk, i))
            self.processes.append(proc)
            proc_num += 1

        for p in self.processes:
            p.daemon = True
            p.start()

        # for remote in self.work_remotes:
        #     remote.close()

    def step_async(self, actions):
        if self.waiting:
            raise Exception("step_async called while waiting")
        self.waiting = True
        for i, remote in enumerate(self.remotes):
            start = i * self.chunksize
            chunk = actions[start : start + self.chunksize]
            remote.send(('step', chunk))
            #print("step sent")

    def step_wait(self):
        if not self.waiting:
            raise Exception("Must be stepping to wait")
        self.waiting = False
        results = []
        for remote in self.remotes:
            results.extend(remote.recv())

        #results = [remote.recv() for remote in self.remotes]
        #print("results recieved")
        obs, rewards, terminated, truncated, info = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(terminated), np.stack(truncated), info
    
    def step(self, actions):
        self.step_async(actions)
        obs, rewards, terminated, truncated, info = self.step_wait()
        # if self.render_mode is not None and self.i % self.render_frame_skip == 0:
        #     self.render_env(obs)
        self.i += 1
        return obs, rewards, terminated, truncated, info

    def render_env(self, observations):
        obs = list(map(lambda image: cv2.resize(image, (self.render_scale, self.render_scale), interpolation=cv2.INTER_LINEAR), observations))

        observations = np.empty(shape=(int(self.n_cols) * self.render_scale, int(self.n_rows) * self.render_scale, 3))
        x_start, y_start = 0, 0
        for i in range(self.pop_size):
            #print(x_start, y_start, observations[x_start: x_start + self.render_scale, y_start: y_start + self.render_scale].shape, obs[i].shape)
            observations[x_start: x_start + self.render_scale, y_start: y_start + self.render_scale] = np.swapaxes(obs[i], 0, 1)
            x_start += self.render_scale
            if x_start >= (self.n_cols * self.render_scale):
                #print(x_start, y_start)
                y_start += self.render_scale
                x_start = 0

        pygame.surfarray.blit_array(self.display, observations)
        pygame.display.update()

    def reset(self, options=None):
        for remote in self.remotes:
            remote.send(('reset', options))
        results = []
        for remote in self.remotes:
            results.extend(remote.recv())
        obs, info = zip(*results)
        return np.stack(obs), info
            
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for worker in self.remotes:
                worker.recv()
        for worker in self.remotes:
            worker.send(('close',None))
        for p in self.processes:
            p.join()
        self.closed = True

    def __del__(self):
        self.close()
