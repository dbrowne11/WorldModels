import torch
import torch.nn as nn
import torch.nn.functional as F

# Good article motivating VAEs and explaining the reparameterization trick
# (which is what happens in sample)
# https://www.baeldung.com/cs/vae-reparameterization
# Default paramaters are used from 'World Models' by 

"""
example: in a 96x96x3 img
COnv1 - 47x47x16
Conv2 - 23x23x32
Conv3 - 11x11x64
"""

class FlexVae(nn.Module):
    def __init__(self, latent_dim, input_shape, ConvParams=None, DeConvParams=None, 
                 linParams=None):
        """
        latent_dim
        input_shape C x H x W
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.channels = input_shape[0]
        self.image_shape=input_shape[1:]

        if ConvParams is None:
            ConvParams = self._defaultConvParams()
        self.convs = nn.ModuleList(
            [nn.Conv2d(**convParam) for convParam in ConvParams]
            )
        self.convEmbedShape = self._calcConvEmbeddingShape(ConvParams)

        self.mu = nn.Linear(self.convEmbedShape[0] * self.convEmbedShape[1] * self.convEmbedShape[2],
                            out_features=latent_dim)
        self.log_var = nn.Linear(self.convEmbedShape[0] * self.convEmbedShape[1] * self.convEmbedShape[2],
                            out_features=latent_dim)
        
        if DeConvParams is None:
            DeConvParams = self._defaultDeConvParams()

        self.deconvs = nn.ModuleList(
            [nn.ConvTranspose2d(**deconvParam) for deconvParam in DeConvParams]
        )

        self.deConvShape = self._calcDeConvEmbeddingShape(DeConvParams)
        self.dec_lin = nn.Linear(in_features=latent_dim, out_features=self.convEmbedShape[0] * self.convEmbedShape[1] * self.convEmbedShape[2])



    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self._sample(mu, log_var)
        recons = self.decode(z)
        return recons, mu, log_var

    def encode(self, x):
        for module in self.convs:
            x = module(x)
            x = F.relu(x)
            #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def _sample(self, mu, log_var):
        eps  = torch.randn_like(log_var)
        z = mu + torch.exp(log_var / 2) * eps
        return z
    
    def decode(self, z):
        x = self.dec_lin(z)
        #x = torch.unflatten(x, 1, self.convEmbedShape)
        x = x.view(-1, 1024, 1, 1)

        for i, deconv in enumerate(self.deconvs):
            #print(x.shape)
            x = deconv(x)
            if i != len(self.deconvs) - 1:
                x = F.relu(x)

        return F.sigmoid(x)

    @staticmethod
    def GetLoss(image, reconstruction, mu, log_var, kl_factor):
        recons_loss = F.mse_loss(reconstruction, image)
        recons_loss = torch.mean(torch.sum(torch.square(reconstruction - image), dim=[1,2,3]))
        kl_loss  = -0.5 * torch.mean(1 + log_var - mu.pow(2) - torch.square(log_var.exp()))
        kl_loss = - 0.5 * torch.sum((1 + log_var - mu ** 2 - log_var.exp()), dim=1)
        #kl_loss = torch.maximum(kl_loss, 0.5 * 32)
        kl_loss = torch.mean(kl_loss)
        #print(recons_loss, kl_loss, kl_div)
        #kl_loss = 
        return  recons_loss +  kl_loss, (recons_loss, kl_loss)

    def _defaultConvParams(self):
        convParams = [
            {
                "in_channels": 3,
                "out_channels": 32,
                "kernel_size": 4,
                "stride": 2,

            },
            {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 4,
                "stride": 2,

            },
            {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 4,
                "stride": 2,

            },
            {
                "in_channels": 128,
                "out_channels": 256,
                "kernel_size": 4,
                "stride": 2
            },
        ]
        return convParams

    def _defaultDeConvParams(self):
        deConvParams = [
            {
                "in_channels": 1024,
                "out_channels": 128,
                "kernel_size": 5,
                "stride": 2
            },
            {
                "in_channels": 128,
                "out_channels": 64,
                "kernel_size": 5,
                "stride": 2
            },
            {
                "in_channels": 64,
                "out_channels": 32,
                "kernel_size": 6,
                "stride": 2
            },
            {
                "in_channels": 32,
                "out_channels": 3,
                "kernel_size": 6,
                "stride": 2
            },
        ]
        return deConvParams
    
    def _calcConvEmbeddingShape(self, convParams):
        H, W = self.image_shape
        C = self.channels
        for params in convParams:
            p = params.get("padding", 0)
            s = params.get("stride", 0)
            H_dim = (H - params.get("kernel_size") + 2 * p) // s + 1
            if H == W:
                H, W = H_dim, H_dim
            else:
                W_dim = (W - params.get("kernel_size") + 2 * p) // s + 1
                H, W = H_dim, W_dim
            C = params.get("out_channels")
        return [C, H, W]
    
    def _calcDeConvEmbeddingShape(self, deconvParams):
        C, H, W = self.convEmbedShape
        for params in deconvParams:
            p = params.get("padding", 0)
            s = params.get("stride", 0)
            H_dim = s * (H - 1) + params.get("kernel_size") - 2*p
            if H == W:
                H, W = H_dim, H_dim
            else:
                W_dim = s * (W - 1) + params.get("kernel_size") - 2*p
                H, W = H_dim, W_dim
            C = params.get("out_channels")
        return [C, H, W]
            




# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from torchvision import datasets, transforms
#     import matplotlib.pyplot as plt

#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     data_path='~/data/mnist'
#     batch_size=100
#     epochs=25
#     vae = FlexVae(2, (3, 96, 96))
#     vae.to(device=device)
#     transform = transforms.Compose([
#                 transforms.Resize((28, 28)),
#                 transforms.Grayscale(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0,), (1,)),
#                 transforms.RandomRotation(8),
#                 transforms.RandomResizedCrop(28, (0.8, 1))])

#     mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
#     mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

#     # Create DataLoaders
#     train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

#     optimizer = torch.optim.Adam(params=vae.parameters(), lr=3e-5, betas=(0.9, 0.99))
#     for epoch in range(epochs):
#         for i, (image, label) in enumerate(train_loader):
#             image, label = image.to(device), label.to(device)
#             if i >= 599:
#                 print("500 steps completed epoch", epoch)

#                 recons, _, _ = vae(image)
#                 print(recons.shape)
#                 fig, axes = plt.subplots(nrows=3,ncols=3)
#                 #print("plt created")
#                 for i in range(9):
#                     #print("setting ax", i)
#                     axes[i//3, i%3].imshow(recons[i].view(28, 28).detach().cpu().numpy())
#                     axes[i//3, i%3].set_title(label[i].detach().cpu().numpy())
#                 print("saving")
#                 plt.xticks([])
#                 plt.yticks([])
#                 fig.suptitle(f"Epoch: {epoch} reconstructions")
#                 plt.savefig(f"plots/Recontructed_images{epoch}.png")
#                 break
                

#             recons, mu, log_var = vae(image)
#             z = vae.reparameterize(mu, log_var)
#             loss = vae.GetLoss(image, recons, mu, log_var, z, label)
#             print(f"Step {i} loss: {loss}")
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()


