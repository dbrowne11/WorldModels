import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.vae.vae import FlexVae
from common import CarDataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Default training parameters
data_path='Datasets/ChunkedCarRacing'
batch_size=100
epochs=10

initial_kl_factor = 0
final_kl_factor = 1


input_size = (3,64,64)
latent_dim = 32


def train(dataloader, model):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        steps = 0
        samples = None
        kl_factor = final_kl_factor * epoch / epochs
        loss_total = 0
        for i, image in enumerate(dataloader):
            #print(image.shape, image.min(), image.max())
            steps += 1
            kl_factor +=5e-4 if kl_factor < final_kl_factor else final_kl_factor
            image = image.to(device)
            samples = image

            recons, mu, log_var = model(image)
            
            loss, (recons_loss, kl_loss) = model.GetLoss(image, recons, mu, log_var, kl_factor=kl_factor)
            print(f"Step {i} loss: {loss}, recons {recons_loss}, kl {kl_loss}")
            loss_total += loss.detach().cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} completed with ave loss {loss_total / steps}, kl_factor {kl_factor}")

        recons, _, _ = model(samples)
        fig, axes = plt.subplots(nrows=4,ncols=2)
        for j in range(4):
            axes[j, 0].imshow(image[j].permute(1,2,0).detach().cpu().numpy())
            axes[j, 1].imshow(recons[j].permute(1,2,0).detach().cpu().numpy())
        plt.xticks([])
        plt.yticks([])
        fig.suptitle(f"Epoch{epoch} reconstructions")
        plt.savefig(f"plots/Car/Recontructed_images_{epoch}.png")
        plt.close()
        torch.save(model.state_dict(), f"models/Checkpoints/vae_hist/{epoch}vae.pt")


if __name__ == "__main__":
    import argparse
    parser  = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=epochs, help="Number of epochs to train for", type=int)
    parser.add_argument('-b', '--batch_size', default=100, help="Batch Size", type=int)
    parser.add_argument('--model_path', default=None, help="path to the saved vision model or None to train a new model")
    parser.add_argument('-z', '--latent_dim', default=latent_dim, type=int)
    args = vars(parser.parse_args())

    latent_dim = args["latent_dim"]
    epochs = args['epochs']
    batch_size = args['batch_size']
    # create and optionally load model
    model = FlexVae(latent_dim, input_size)
    if args['model_path'] is not None and args["model_path"] != "None":
        model.load_state_dict(torch.load(args['model_path']))
    model.to(device)

    # Create DataLoaders
    car_train = CarDataset(data_path, dataloader="img", filesize=32)
    train_loader = DataLoader(car_train, batch_size=batch_size, shuffle=True)

    train(train_loader, model)


    
