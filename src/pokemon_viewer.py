import streamlit as st
import os
import pandas as pd
import requests
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from joblib import load, dump
from PIL import Image
from io import BytesIO
from preprocessing.data_utils import load_data, clean_data, preprocess_data, PokemonDataset
from model.train_gan import train
from model.generate_from_gan import load_generator, generate_dataset
from descriptions.generate_descriptions import initialize_description_dataset


def header(text, color="#333", font_color="#f5f5f5", font_size="24px"):
    st.markdown(f'<div style="text-align:center;background-color:{color};color:{font_color};font-size:{font_size};border-radius:5%;padding:5px;">{text}</div>', unsafe_allow_html=True)


def load_data_st():
    df_path = "data/generated_outputs/generated_pokemon_descriptions.csv"
    df = pd.read_csv(df_path)
    df = df[df["Image URL"] != "Image generation failed"]
    return df


def main_page():
    st.title("Pokémon Generator")
    st.write("Welcome to the Pokémon Generator!")


def train_gan_page():
    st.title("Train a new GAN model")

    # User inputs for hyperparameters
    dataset_path = st.text_input("Enter path for entire dataset", "data/pokemon.csv")

    data = load_data(dataset_path)
    data = clean_data(data)
    preprocessed_tensor, type1_labels, type2_labels, scaler = preprocess_data(data)

    # Save scalar as a pickle file
    dump(scaler, 'data/scaler.pkl')

    dataset = PokemonDataset(preprocessed_tensor)

    st.write(f"Dataset:")
    st.write(data)

    epochs = st.number_input("Epochs", 10000)

    es_flag = st.checkbox("Enable Early Stopping: stops training if it the loss delta reaches the specified "
                          "delta threshold", value=True)
    es_patience = st.number_input("Early Stopping patience value", 200)
    es_delta = st.number_input("Early Stopping delta threshold", format="%f", value=0.001)

    if st.button("Start Training"):
        st.write("Starting training...")
        generator, discriminator, d_losses, g_losses = train(dataset, lr_g=0.0001, lr_d=0.0004, epochs=epochs,
                                                             noise_dim=128, output_dim=42, resume_from_epoch=None,
                                                             print_fct=st.write, es_flag=es_flag,
                                                             es_patience=es_patience, es_delta=es_delta)

        st.write("Training finished.")

        plt.figure(figsize=(12, 6))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig('data/plots/training_curve.png', bbox_inches='tight')

        training_curve_img = Image.open('data/plots/training_curve.png')
        st.image(training_curve_img, use_column_width=True, caption="")


def generate_pokemon_page():
    st.title("Generate Pokémon using the latest model")

    num_samples = int(st.number_input("Number of samples to generate"))
    noise_dim = 128
    output_dim = 42
    generator_path = "data/saved_models/generator.pth"
    scaler_path = "data/scaler.pkl"

    if st.button("Generate"):
        st.write("Generating Pokémon...")
        generator = load_generator(generator_path, noise_dim, output_dim)
        scaler = load(scaler_path)

        type1_labels = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass',
                        'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']
        type2_labels = ['None', 'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost',
                        'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']

        new_pokemons_dataset = generate_dataset(generator, scaler, num_samples, noise_dim=noise_dim, type1_labels=type1_labels, type2_labels=type2_labels)

        image_gen = "dalle"
        new_pokemons_dataset.to_csv('data/generated_outputs/generated_pokemon.csv', index=False)

        if image_gen == "hf":
            ...
            # pipeline = DiffusionPipeline.from_pretrained("justinpinkney/pokemon-stable-diffusion")
            # initialize_description_dataset(pokemon_data, image_gen=image_gen, generator=pipeline)
        elif image_gen == "dalle":
            initialize_description_dataset(new_pokemons_dataset, image_gen=image_gen)

        st.write(pd.read_csv('data/generated_outputs/generated_pokemon_descriptions.csv'))
        st.write(f"Generated {num_samples} pokemon")


def display_pokemon_page():
    df = load_data_st()
    selected_name = st.sidebar.selectbox("Select a Pokémon", df["Name"].tolist())
    pokemon_data = df[df["Name"] == selected_name].iloc[0]

    if st.sidebar.button("Reload CSV"):
        st.experimental_rerun()

    # Create Pokémon Card Layout
    st.markdown(f"<h2 style='text-align:center;'>{pokemon_data['Name']}</h2>", unsafe_allow_html=True)

    # Display the Pokémon image at center
    response = requests.get(pokemon_data["Image URL"])
    image = Image.open(BytesIO(response.content))
    st.image(image, use_column_width=True, caption="")

    # Pokémon Types
    st.markdown(f"<div style='text-align:center;'><b>Type 1</b>: {pokemon_data['Type 1']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;margin-bottom:20px;'><b>Type 2</b>: {pokemon_data['Type 2']}</div>",
                unsafe_allow_html=True)

    # Pokémon Description over the stats
    st.markdown(
        f'<div style="border-radius:5px;padding:10px;margin:auto;text-align:center;">'
        f'{pokemon_data["Description"]}</div>', unsafe_allow_html=True)

    # Pokémon Stats in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<div style='text-align:center;'><b>HP</b>: {pokemon_data['HP']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'><b>Attack</b>: {pokemon_data['Attack']}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'><b>Defense</b>: {pokemon_data['Defense']}</div>",
                    unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div style='text-align:center;'><b>Sp. Attack</b>: {pokemon_data['Sp. Attack']}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'><b>Sp. Defense</b>: {pokemon_data['Sp. Defense']}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'><b>Speed</b>: {pokemon_data['Speed']}</div>", unsafe_allow_html=True)

    # Closing the card's div background
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    PAGES = {
        "Home": main_page,
        "Train GAN": train_gan_page,
        "Generate Pokémon": generate_pokemon_page,
        "Display Pokémon": display_pokemon_page
    }

    # Sidebar for navigation
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[choice]()
