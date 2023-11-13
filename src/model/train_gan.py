import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.gan_architecture import Generator, Discriminator
from model.early_stopping import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparameters
lr_g = 0.0001
lr_d = 0.0004
batch_size = 32
epochs = 10000
noise_dim = 128
output_dim = 42
es_patience = 200
es_delta = 0.001
es_flag = True
resume_from_epoch=None
GRAD_PENALTY_WEIGHT = 10


def gradient_penalty(critic, real, fake, device="cpu"):
    alpha = torch.rand((real.size(0), 1)).expand_as(real).to(device)
    interpolated = alpha * real.detach() + (1 - alpha) * fake.detach()
    interpolated.requires_grad_(True)

    out = critic(interpolated)

    grad_out = grad(outputs=out, inputs=interpolated, grad_outputs=torch.ones_like(out), create_graph=True,
                    retain_graph=True, only_inputs=True)[0]
    penalty = ((grad_out.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


def train(dataset, lr_g=lr_g, lr_d=lr_d, epochs=epochs, noise_dim=noise_dim, output_dim=output_dim,
          resume_from_epoch=resume_from_epoch, print_fct=print, es_flag=es_flag, es_patience=es_patience,
          es_delta=es_delta, batch_size=batch_size):

    start_time = time.time()

    generator = Generator(noise_dim, output_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d)

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=1000, gamma=0.95)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=1000, gamma=0.95)

    # Add custom early stopping
    if es_flag:
        early_stopping = EarlyStopping(patience=es_patience, delta=es_delta)

    if resume_from_epoch is not None:
        generator.load_state_dict(torch.load(f"data/saved_models/generator_epoch_{resume_from_epoch}.pth"))
        discriminator.load_state_dict(torch.load(f"data/saved_models/discriminator_epoch_{resume_from_epoch}.pth"))
        optimizer_g.load_state_dict(torch.load(f"data/saved_models/optimizer_g_epoch_{resume_from_epoch}.pth"))
        optimizer_d.load_state_dict(torch.load(f"data/saved_models/optimizer_d_epoch_{resume_from_epoch}.pth"))

        with open('data/saved_models/d_losses.pkl', 'rb') as f:
            d_losses = pickle.load(f)

        with open('data/saved_models/g_losses.pkl', 'rb') as f:
            g_losses = pickle.load(f)
    else:
        d_losses = []
        g_losses = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(resume_from_epoch if resume_from_epoch is not None else 0, epochs):
        d_loss_total = 0.0
        g_loss_total = 0.0

        for real_data in dataloader:
            current_batch_size = real_data.size(0)
            real_data = real_data.to(device)

            optimizer_d.zero_grad()

            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_data = generator(noise)

            real_labels = torch.ones(current_batch_size, 1).to(device) * (0.9 - 0.1 * torch.rand((current_batch_size, 1)).to(device))
            fake_labels = torch.zeros(current_batch_size, 1).to(device) + 0.1 * torch.rand((current_batch_size, 1)).to(device)

            logits_real = discriminator(real_data)
            logits_fake = discriminator(fake_data.detach())

            loss_real = criterion(logits_real, real_labels)
            loss_fake = criterion(logits_fake, fake_labels)
            penalty = gradient_penalty(discriminator, real_data, fake_data, device=device)

            loss_d = loss_real + loss_fake + GRAD_PENALTY_WEIGHT * penalty
            loss_d.backward()
            optimizer_d.step()

            d_loss_total += loss_d.item()

            optimizer_g.zero_grad()

            # Generate fake data
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_data = generator(noise)
            logits_fake = discriminator(fake_data)

            # Feature matching
            real_features = discriminator.extract_features(real_data)
            fake_features = discriminator.extract_features(fake_data)
            feature_matching_loss = F.l1_loss(fake_features, real_features.detach())

            # Update the generator based on the sum of the original generator's loss and the feature matching loss
            loss_g = criterion(logits_fake, real_labels) + feature_matching_loss
            loss_g.backward()
            optimizer_g.step()

            g_loss_total += loss_g.item()

        d_losses.append(d_loss_total / len(dataloader))
        g_losses.append(g_loss_total / len(dataloader))

        scheduler_g.step()
        scheduler_d.step()

        if epoch % 100 == 0:
            print_fct(f"Epoch [{epoch}/{epochs}]\n"
                  f"Avg Discriminator Loss: {d_losses[-1]:.4f}\n"
                  f"Avg Generator Loss: {g_losses[-1]:.4f}")
            for param_group in optimizer_d.param_groups:
                print_fct(f"Discriminator Learning Rate: {param_group['lr']:.5f}")
            for param_group in optimizer_g.param_groups:
                print_fct(f"Generator Learning Rate: {param_group['lr']:.5f}")

        if epoch % 1000 == 0:
            torch.save(generator.state_dict(), f"data/saved_models/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"data/saved_models/discriminator_epoch_{epoch}.pth")
            torch.save(optimizer_g.state_dict(), f"data/saved_models/optimizer_g_epoch_{epoch}.pth")
            torch.save(optimizer_d.state_dict(), f"data/saved_models/optimizer_d_epoch_{epoch}.pth")

            with open('data/saved_models/d_losses.pkl', 'wb') as f:
                pickle.dump(d_losses, f)

            with open('data/saved_models/g_losses.pkl', 'wb') as f:
                pickle.dump(g_losses, f)

        # Early stopping check
        if es_flag and early_stopping(d_losses[-1]):
            print_fct("Early stopping triggered.")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print_fct(f"Training completed in {elapsed_time:.2f} seconds.")

    return generator, discriminator, d_losses, g_losses
