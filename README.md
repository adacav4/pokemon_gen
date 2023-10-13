# Pokémon Stats Generator - Using a GAN (Generative Adversarial Network)

This repository contains an implementation of a Generative Adversarial Network (GAN) to generate data for fictional Pokémon. The project consists of a generator and discriminator that are trained together to produce realistic Pokémon data, including stats such as HP, Attack, Defense, and more. 

The user can then utilize the synthetic data generated from the GAN with an LLM (like GPT) and an image diffuser model (like DALL-E) to generate a realistic Pokémon card for each synthetically generated Pokémon, with descriptions and images. 

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
   - [Generator](#generator)
   - [Discriminator](#discriminator)
   - [Streamlit User Interface](#streamlit-user-interface)
5. [Training the Model](#training-the-model)
6. [Generating New Pokémon](#generating-new-pokémon)
7. [Contributions](#contributions)


## Installation

To run the code in this repository, you will need a Python/Anaconda environment. Here are the steps to get it up and running:

1. Clone this repository:
```bash
git clone https://github.com/adacav4/pokemon_gen.git
cd pokemon_gen
```

2. Create an Anaconda virtual environment (optional but recommended):
```bash
conda create -n pokemon_gen python=3.8 -y
conda activate pokemon_gen
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```


## Dataset

The dataset used for this project consists of Pokémon statistics, including various metrics like HP, Attack, Defense, Special Attack, Special Defense, Speed, Base Stat, as well as their types. Below are the first 10 Pokémon entries from the dataset:

| Dex No | Name           | Base Name   | Type 1 | Type 2  | BST | HP | Attack | Defense | Sp. Attack | Sp. Defense | Speed |
|--------|----------------|-------------|--------|---------|-----|----|--------|---------|------------|-------------|-------|
| 0001   | Bulbasaur      | Bulbasaur   | GRASS  | POISON  | 318 | 45 | 49     | 49      | 65         | 65          | 45    |
| 0002   | Ivysaur        | Ivysaur     | GRASS  | POISON  | 405 | 60 | 62     | 63      | 80         | 80          | 60    |
| 0003   | Venusaur       | Venusaur    | GRASS  | POISON  | 525 | 80 | 82     | 83      | 100        | 100         | 80    |
| 0003   | Mega Venusaur  | Venusaur    | GRASS  | POISON  | 625 | 80 | 100    | 123     | 122        | 120         | 80    |
| 0004   | Charmander     | Charmander  | FIRE   | -       | 309 | 39 | 52     | 43      | 60         | 50          | 65    |
| 0005   | Charmeleon     | Charmeleon  | FIRE   | -       | 405 | 58 | 64     | 58      | 80         | 65          | 80    |
| 0006   | Charizard      | Charizard   | FIRE   | FLYING  | 534 | 78 | 84     | 78      | 109        | 85          | 100   |
| 0006   | Mega Charizard X | Charizard  | FIRE   | DRAGON  | 634 | 78 | 130    | 111     | 130        | 85          | 100   |
| 0006   | Mega Charizard Y | Charizard  | FIRE   | FLYING  | 634 | 78 | 104    | 78      | 159        | 115         | 100   |
| 0007   | Squirtle       | Squirtle    | WATER  | -       | 314 | 44 | 48     | 65      | 50         | 64          | 43    |

This dataset was used to train the GAN and generate new Pokémon stats. The dataset can be downloaded at: https://www.kaggle.com/datasets/rounakbanik/pokemon.

**Note**: If you're using your own dataset, make sure to modify the paths in the code accordingly.

## Model Architecture

### Generator

The generator plays a crucial role in the GAN setup, aiming to create data that resembles the real Pokémon statistics. It's designed to accept random noise as input, and after processing through its architecture, it outputs a set of values that correspond to Pokémon statistics.

The generator employs a deep learning architecture and employs Batch Normalization after dense layers to ensure a more stable and efficient training process by reducing the internal covariate shift. Also, using Leaky ReLU activations help the network learn non-linear mappings and prevent the dying ReLU problem where neurons could sometimes get stuck during training and not activate at all.

By the end of the generator's architecture, we have a fully-formed set of Pokémon statistics, ready to be evaluated by the discriminator.

### Discriminator

The discriminator is used to evaluate the authenticity of the data created by the generator. It takes in Pokémon statistics, whether real or generated, and assigns a probability score indicating how real the data looks. 

Layer Normalization is used to stabilize the training of GANs. This stablizes the training dynamic between the generator and discriminator. When training the model, mode collapse was an issue, and layer normalization helps regulate this. Also, a gradient penalty is used, which can also prevent mode collapse.

### Streamlit User Interface

The Streamlit interface has been designed to offer users an easy experience when working with the GAN model and the Pokémon dataset.

#### Train GAN:
- **Dataset Selection**: Users can choose the specific Pokémon dataset they wish to work with, ensuring versatility in model training.
- **Customization Controls**: Adjust various training parameters for any custom fine-tuning like epoch count, starting epoch checkpoint, early stopping parameters, batch size, learning rates.
- **Training Visualization**: Real-time logs and loss graphs provide insights into the GAN model's learning process, ensuring users can track the model's evolution.

#### Generate Pokémon:
- **Generation Count**: Decide how many Pokémon you want to generate in one go, whether it's a single Pokémon or an entire batch.
- **Model Selection**: Opt between previously trained models or use the default one to generate Pokémon.
- **Instant Preview**: As soon as the Pokémon are generated, they are displayed, letting users review the GAN model's capabilities.

#### Display Pokémon:
- **Generated Pokémon Display**: Each Pokémon is showcased in a dedicated card-style layout for easy viewing.
- **DALLE-Generated Image**: Accompanying each Pokémon is an image crafted using the DALLE model, offering a visual representation of the Pokémon.
- **GPT-Generated Description**: For context and clarity, a brief description of the Pokémon, generated by GPT, is provided below the image.
- **Pokémon Statistics**: Essential stats of each Pokémon are clearly laid out, offering insights into their strengths, abilities, and unique features.


## Training the Model

The training proceeds in an alternating fashion. First, the discriminator is updated to get better at distinguishing real data from fake. Then, the generator is updated to attempt to fool the discriminator. The generator goes through this loop, testing out new outputs against the discriminator until convergance.

To kick off the training, use the Streamlit UI and navigate to the "Train GAN" page. There, you can change the training settings and run training. Look into the source code `train_gan.py` for more details.

Alternatively, you can execute the command from a bash shell:

```bash
python run_train.py
```


## Generating New Pokémon

To produce new Pokémon, we just provide the generator with random noise vectors. It then uses its learned parameters to output Pokémon statistics that resemble those from the actual dataset.

To generate new Pokémon statistics, you can use the Streamlit UI and navigate to the "Generate Pokémon" page. You are able to change the number of Pokémon generated through the Streamlit UI and specify any available model checkpoint from training.

Alternatively, you can run the following Python script from a bash shell:

```bash
python generate_from_gan.py
```

The results will be saved in a specified output directory, which will contain the statistics of newly imagined Pokémon.

## Contributions

Contributions to improve this project are always welcome. Whether it's optimizing the model, enhancing the architecture, or even adding new features, feel free to fork the repository, make your changes, and submit a pull request.

Before making any major changes, please open an issue to discuss what you would like to contribute. This will ensure we're on the same page and will save any unnecessary effort on both sides.

If you wish to collaborate on this project, please email me at adarsh.cavale@gmail.com.
