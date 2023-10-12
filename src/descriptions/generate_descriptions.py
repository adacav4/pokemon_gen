import os
import pandas as pd
import openai
from diffusers import DiffusionPipeline

openai.api_key = "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"


def get_pokemon_description(pokemon):
    system_prompt_init = "You are an expert Pokémon researcher, professor, and storyteller. Your knowledge combines " \
                         "the lore, characteristics, and nuances of every Pokémon species. You are tasked with " \
                         "crafting unique names and detailed descriptions of Pokémon based on their provided stats, " \
                         "ensuring the descriptions encapsulate their appearance, behavior, and any unique " \
                         "attributes. The descriptions should be captivating, imaginative, and paint a vivid picture " \
                         "of the Pokémon, helping anyone visualize them clearly even if they've never encountered " \
                         "them before. Remember, the stats given offer clues about their potential strengths, " \
                         "abilities, and characteristics. Use this data to fuel your creativity. The name and " \
                         "description should be in the format: 'Name|Description', and the description should be all" \
                         "in one line, in one paragraph."

    prompt = f"Using the stats: HP: {pokemon['HP']}, Attack: {pokemon['Attack']}, Defense: " \
             f"{pokemon['Defense']}, Special Attack: {pokemon['Sp. Attack']}, Special Defense: " \
             f"{pokemon['Sp. Defense']}, Speed: {pokemon['Speed']}, Type 1: {pokemon['Type 1']}, and " \
             f"Type 2: {pokemon['Type 2']}, imagine and describe a new, unique Pokémon's appearance and " \
             f"behavior that hasn't been seen before. Respond in a complete and continuous paragraph. " \
             f"Provide its name and description in the format: 'Name|Description', keeping the description as one" \
             f"line and in one paragraph."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt_init},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']


def generate_pokemon_image_dalle(pokemon_name, pokemon_description):
    static_prompt = f"{pokemon_name} - a Pokémon creature resembling "
    post_description = ". Designed in the style of traditional cartoon Pokémon images with a black background."

    # Calculate the maximum space available for the description
    available_space_for_description = 1000 - len(static_prompt) - len(post_description)

    # If the description exceeds the available space, trim it.
    if len(pokemon_description) > available_space_for_description:
        pokemon_description = pokemon_description[:available_space_for_description - 3] + "..."

    # Construct the complete prompt
    prompt = static_prompt + pokemon_description + post_description

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    image_url = response['data'][0]['url']
    return image_url


def generate_pokemon_image_HF(pokemon_name, pokemon_description, generator):
    static_prompt = f"{pokemon_name} - a Pokémon creature resembling "
    post_description = ". Designed in the style of traditional cartoon Pokémon images with a black background."

    # Calculate the maximum space available for the description
    available_space_for_description = 1000 - len(static_prompt) - len(post_description)

    # If the description exceeds the available space, trim it.
    if len(pokemon_description) > available_space_for_description:
        pokemon_description = pokemon_description[:available_space_for_description - 3] + "..."

    # Construct the complete prompt
    prompt = static_prompt + pokemon_description + post_description

    image = generator(prompt).images[0]

    # Save the image to a local path
    image_path = os.path.join(f"data/generated_outputs/images/{pokemon_name}.png")
    image.save(image_path)

    return image_path


def initialize_description_dataset(df, image_gen='dalle', generator=None):
    names = []
    descriptions = []
    image_urls = []  # Storing the URLs of the generated images

    for index, row in df.iterrows():
        print(f"Starting name/description generation for pokemon {index + 1}...")
        description_output = get_pokemon_description(row)
        split_output = description_output.split("|")

        # Check if the expected split resulted in at least 2 parts.
        if len(split_output) > 1:
            name = split_output[0].strip()
            description = split_output[1].strip()

            names.append(name)
            descriptions.append(description)
            image_url = "Image generation failed"

            print(f"Starting Image generation for {name}...")

            if image_gen == 'dalle':
                try:
                    image_url = generate_pokemon_image_dalle(name, description)
                    image_urls.append(image_url)
                except Exception as e:
                    print(f"Failed to generate image for {name}. Reason: {e}\n")
                    image_urls.append(image_url)

                print(f"Name: {name}\nDescription: {description}\nImage: {image_url}\n")
            elif image_gen == 'hf':
                try:
                    image_url = generate_pokemon_image_HF(name, description, generator)
                    image_urls.append(image_url)
                except Exception as e:
                    print(f"Failed to generate image for {name}. Reason: {e}\n")
                    image_urls.append(image_url)
                print(f"Name: {name}\nDescription: {description}\nImage Path: {image_url}\n")
            else:
                raise Exception("Must provide correct image generator: 'dalle' for DALLE or 'hf' for "
                                "HuggingFace Stable Diffusion")

        else:
            names.append("Unknown Name")
            descriptions.append("Description not found.")
            image_urls.append("Image not available.")

    df['Name'] = names
    df['Description'] = descriptions
    df['Image URL'] = image_urls  # Adding a new column for the image URLs
    df = df[['Name', 'Type 1', 'Type 2', 'BST', 'HP', 'Attack', 'Defense', 'Sp. Attack', 'Sp. Defense', 'Speed',
             "Image URL", "Description"]]
    df.to_csv('data/generated_outputs/generated_pokemon_descriptions.csv', index=False)


if __name__ == "__main__":
    image_gen = "dalle"
    pokemon_data = pd.read_csv('../../data/generated_outputs/generated_pokemon.csv')

    if image_gen == "hf":
        pipeline = DiffusionPipeline.from_pretrained("justinpinkney/pokemon-stable-diffusion")
        initialize_description_dataset(pokemon_data, image_gen=image_gen, generator=pipeline)
    elif image_gen == "dalle":
        initialize_description_dataset(pokemon_data, image_gen=image_gen)

    print("Generated Pokémon names and descriptions to 'generated_pokemon_descriptions.csv'.")