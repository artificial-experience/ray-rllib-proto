# Basic Ray RLlib Project Skeleton

This repository provides a streamlined setup for training Reinforcement Learning (RL) models using the Ray RLlib library with environments adhering to the OpenAI Gym API.

The project encapsulates the RL training process within the `Engine` class. This class enables simple setup, configuration, and execution of RL training based on a specified configuration file.

## Key Features:
- **Environment Activation:** Easily set up and activate the working environment using the provided shell script.
- **Gym Environment Compatibility:** The framework is designed to handle environments following the OpenAI Gym API.
- **RL Algorithm Configuration:** You can specify the RL algorithm for training directly within the `Engine` class.
- **Hyperparameter Configuration:** Set and adjust the hyperparameters for the RL algorithm directly via the configuration file.
- **Training Control:** You have the ability to define stopping conditions and checkpointing policies for the training process.
- **Hyperparameter Optimization:** The project seamlessly integrates with Ray Tune for RL model hyperparameter optimization.

## Getting Started:

Firstly, you need to activate the environment by running the following command in your shell:

```bash```
source activate_env.sh

This will set up and activate the working environment necessary for running the project.

Next, provide a YAML configuration file detailing the training parameters. The main script reads this file, initializes the Engine class with the loaded configuration, applies the parameters, and commences the RL training process.
