import torch
import os
from torchvision import transforms
import numpy as np



class CarDataset(torch.utils.data.Dataset):
    img_transform = transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize((0,), (1,)),
            ])
    def __init__(self, root, transform=None, dataloader="img", filesize=32):
        #print(os.listdir(root))
        self.root = root
        self.list_of_paths = os.listdir(root)
        self.dataloader = dataloader
        if filesize is not None and filesize != 1:
            self.data_buffer = []

    def  __len__(self):
        return len(self.list_of_paths)

    def __getitem__(self, x):
        if len(self.data_buffer) == 0:
            path = self.list_of_paths[x]  # Gives the path to an image
            data = np.load(self.root + '/' + path, allow_pickle=True)
            obs = data["obs"]
            action = data["actions"]
            next_obs = data["next_obs"]
            reward = data["rewards"]
            #print(obs, action, next_obs, reward)
            self._add_to_buf(obs, action, next_obs, reward)
        
        
        return self.data_buffer.pop()

    def _add_to_buf(self, observations, actions, next_obs, rewards):
        if self.dataloader == 'img':
            imgs = observations
            for img in imgs:
                self.data_buffer.append(self.img_transform(img))
        elif self.dataloader == 'all':
            #print(observations.shape, actions.shape, next_obs.shape, rewards.shape)
            for obs, action, next_obs, reward in zip(observations, actions, next_obs, rewards):
                self.data_buffer.append((self.img_transform(obs), 
                                        torch.tensor(action, dtype=torch.float32),
                                        self.img_transform(next_obs),
                                        torch.tensor(reward, dtype=torch.float32)))
        else:
            print('Dataloader must be one of [ "img", "all"]')
            return
        