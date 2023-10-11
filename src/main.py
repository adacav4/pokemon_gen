from joblib import dump
import torch
import matplotlib.pyplot as plt
from src.preprocessing.data_utils import load_data, clean_data, preprocess_data, PokemonDataset
from src.model.train_gan import train_gan


def main():
    # Preprocess the Data
    print("Preprocessing data...")
    filepath = '../data/pokemon.csv'
    data = load_data(filepath)
    data = clean_data(data)
    preprocessed_tensor, type1_labels, type2_labels, scaler = preprocess_data(data)

    # Save scalar as a pickle file
    dump(scaler, '../data/scaler.pkl')

    dataset = PokemonDataset(preprocessed_tensor)

    # Train the GAN
    print("Training GAN model...")
    generator, discriminator, d_losses, g_losses = train_gan(dataset, resume_from_epoch=None)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('../data/plots/training_curve.png', bbox_inches='tight')

    # Save the trained models
    print("Saving models...")
    torch.save(generator.state_dict(), '../data/saved_models/generator.pth')
    torch.save(discriminator.state_dict(), '../data/saved_models/discriminator.pth')


if __name__ == "__main__":
    main()
