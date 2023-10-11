# Pokémon Stats Generator - Using a GAN (Generative Adversarial Network)

This repository contains an implementation of a Generative Adversarial Network (GAN) to generate data for fictional Pokémon. The project consists of a generator and discriminator that are trained together to produce realistic Pokémon data, including stats such as HP, Attack, Defense, and more. 

The end goal is to use this GAN with a LLM (like GPT) and an image diffuser model (like DALL-E) to generate realistic Pokémon, with descriptions and images. 

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training the Model](#training-the-model)
5. [Generating New Pokémon](#generating-new-pokémon)
6. [Contributions](#contributions)


## Installation

To run the code in this repository, you will need a Python environment. Here are the steps to get it up and running:

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

Spectral Normalization is used to stabilize the training of GANs. By constraining the spectral norm (largest singular value) of the weight matrices in the discriminator, the Lipschitz constant is better controlled, stablizing the training dynamic between the generator and discriminator. When training the model, mode collapse was an issue, and spectral normalization helps regulate this. Also, a gradient penalty is used, which can also prevent mode collapse.

## Training the Model

The training proceeds in an alternating fashion. First, the discriminator is updated to get better at distinguishing real data from fake. Then, the generator is updated to attempt to fool the discriminator. The generator goes through this loop, testing out new outputs against the discriminator until convergance.

To kick off the training, you can execute the command:

```bash
python main.py
```

To start at a model checkpoint please edit the default argument `resume_from_epoch` within `train_gan` to be any 1000th epoch (e.g. `resume_from_epoch=7000`). Ensure that the epoch model file, epoch optimizer files, and the losses files are present within the directory `data/saved_models` or else the model training cannot resume from that ecpoch properly:

```python
def train_gan(dataset, lr_g=0.0001, lr_d=0.0004, epochs=10000, noise_dim=128, output_dim=42, resume_from_epoch=None):
```


## Generating New Pokémon

To produce new Pokémon, we just provide the generator with random noise vectors. It then uses its learned parameters to output Pokémon statistics that resemble those from the actual dataset.

To generate new Pokémon statistics:

```bash
python generate_from_gan.py
```

The results will be saved in a specified output directory, which will contain the statistics of newly imagined Pokémon. It will generate 20 entries by default, but can be changed within the `generate_dataset()` function by changing `num_samples=20` to another value:

```python
def generate_dataset(generator, scaler, num_samples=20, noise_dim=228, type1_labels=None, type2_labels=None):
```

## Contributions

Contributions to improve this project are always welcome. Whether it's optimizing the model, enhancing the architecture, or even adding new features, feel free to fork the repository, make your changes, and submit a pull request.

Before making any major changes, please open an issue to discuss what you would like to contribute. This will ensure we're on the same page and will save any unnecessary effort on both sides.

If you wish to collaborate on this project, please email me at adarsh.cavale@gmail.com.
