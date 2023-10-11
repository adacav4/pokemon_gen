import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Function to load the dataset
def load_data():
    df_path = "data/generated_outputs/generated_pokemon_descriptions.csv"
    return pd.read_csv(df_path)

# Initially load the dataset
df = load_data()

# Add a button to reload the dataset
if st.sidebar.button("Reload CSV"):
    df = load_data()

# Create a sidebar selection using Pokémon names
selected_name = st.sidebar.selectbox("Select a Pokémon", df["Name"].tolist())

# Get the data for the selected Pokémon
pokemon_data = df[df["Name"] == selected_name].iloc[0]

# Display the Pokémon image
response = requests.get(pokemon_data["Image URL"])
image = Image.open(BytesIO(response.content))
st.image(image, caption=pokemon_data["Name"], use_column_width=True)

# Display the Pokémon details
st.write(f"**Name**: {pokemon_data['Name']}")
st.write(f"**Type 1**: {pokemon_data['Type 1']}")
st.write(f"**Type 2**: {pokemon_data['Type 2']}")
st.write(f"**BST**: {pokemon_data['BST']}")
st.write(f"**HP**: {pokemon_data['HP']}")
st.write(f"**Attack**: {pokemon_data['Attack']}")
st.write(f"**Defense**: {pokemon_data['Defense']}")
st.write(f"**Sp. Attack**: {pokemon_data['Sp. Attack']}")
st.write(f"**Sp. Defense**: {pokemon_data['Sp. Defense']}")
st.write(f"**Speed**: {pokemon_data['Speed']}")
st.write(f"**Description**: {pokemon_data['Description']}")
