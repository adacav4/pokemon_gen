from joblib import load
import torch
import pandas as pd
from src.model.gan_arcitecture import Generator


def load_generator(model_path, noise_dim=128, output_dim=228):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(noise_dim, output_dim).to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    return generator


def generate_pokemon(generator, num_samples=1, noise_dim=228):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn(num_samples, noise_dim).to(device)
    with torch.no_grad():
        generated_data = generator(noise)
    return generated_data


def generate_dataset(generator, scaler, num_samples=20, noise_dim=228, type1_labels=None, type2_labels=None):
    generated_data = generate_pokemon(generator, num_samples=num_samples, noise_dim=noise_dim)
    inverse_transformed_data = inverse_transform_data(generated_data, scaler, type1_labels, type2_labels)
    return inverse_transformed_data


def inverse_transform_data(generated_data, scaler, type1_labels, type2_labels):
    # Split the data into numerical and categorical
    numerical_data = generated_data[:, :7].cpu().numpy()
    categorical_data = generated_data[:, 7:].cpu().numpy()

    # Inverse transform the numerical data
    numerical_data = scaler.inverse_transform(numerical_data)

    # Decode the one-hot encoded categorical data
    type1_decoded = [type1_labels[row.argmax()] for row in categorical_data[:, :len(type1_labels)]]
    type2_decoded = [type2_labels[row.argmax()] if row.max() > 0 else 'None' for row in
                     categorical_data[:, len(type1_labels):]]

    # Combine the decoded data
    dataset = pd.DataFrame(numerical_data,
                           columns=['HP', 'Attack', 'Defense', 'Sp. Attack', 'Sp. Defense', 'Speed', 'BST'])
    dataset['Type 1'] = type1_decoded
    dataset['Type 2'] = type2_decoded

    return dataset


if __name__ == "__main__":
    noise_dim = 128
    output_dim = 42
    generator_path = '../data/saved_models/generator.pth'

    # Load the saved models and scaler
    generator = load_generator(generator_path, noise_dim, output_dim)
    scaler = load('../data/scaler.pkl')

    type1_labels = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass',
                    'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']
    type2_labels = ['None', 'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost',
                    'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']

    new_pokemons_dataset = generate_dataset(generator, scaler, num_samples=20, noise_dim=noise_dim,
                                            type1_labels=type1_labels, type2_labels=type2_labels)

    # Save the generated dataset
    new_pokemons_dataset.to_csv('../data/generated_outputs/generated_pokemon.csv', index=False)
