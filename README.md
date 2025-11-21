# Reinforcement Learning with gym4real & Stable Baselines3

This repository contains a small reinforcement learning project where we train a PPO agent (from Stable Baselines3) on the **Elevator** environment from the **gym4real** library.
The goal is to show an end-to-end setup:

- Installing and wiring a local copy of `gym4real`
- Exploring the environment (observation space, action space, rewards)
- Training a **non-tabular** RL algorithm (PPO with neural networks)
- Evaluating the trained agent and saving models

  We used the official gym4real repository as the base environment implementation:`https://github.com/Daveonwave/gym4ReaL`
  
---

## Project Structure

```text
.
├── gym4ReaL/               # Local copy of Daveonwave/gym4ReaL (installed in editable mode)
├── explore_environment.py   # Step 1: Explore and understand gym4real environments
├── list_environments.py     # Helper: list and inspect registered gym/gymnasium environments
├── train_ppo.py             # Step 2: Train a PPO agent on a selected gym4real env
├── evaluate_model.py        # Step 3: Evaluate a trained PPO agent
├── requirements.txt         # Python dependencies for this project (SB3, gymnasium, etc.)
└── .gitignore               # Ignore models, logs, venv, etc.
```
**`list_environments.py`**
  - Utility script to:
    - Import `gym4real` (using the local `gym4ReaL/` folder).
    - Ask Gymnasium for all registered environments.
    - Filter those that belong to `gym4real` and print their IDs and spaces.
  - This was crucial when we weren’t sure which environment IDs were actually registered.

- **`explore_environment.py`**
  - Creates a chosen gym4real environment (defaults to `gym4real/elevator-v0`).
  - Prints:
    - Observation space details (type, structure, sample)
    - Action space (discrete/continuous, shape, sample)
    - Some random steps with random actions and rewards
  - Used to “get familiar with the environment” before training.

- **`train_ppo.py`**
  - Main training script that:
    - Creates the selected environment with the right **settings** (e.g. elevator world YAML).
    - Detects if the observation space is a `Dict` and automatically selects:
      - `MultiInputPolicy` for dict observations (elevator, etc.)
      - `MlpPolicy` for simple Box/Discrete observations
    - Trains a PPO agent for a configurable number of timesteps.
    - Saves:
      - Final model to `models/ppo_<env_name>.zip`
      - Best model (via evaluation callback) under `models/ppo_<env_name>_best/best_model.zip`
    - Logs to `logs/` and `logs/tensorboard/` for monitoring.

- **`evaluate_model.py`**
  - Loads a trained PPO model and evaluates it for a given number of episodes.
  - Handles several model path patterns:
    - `models/ppo_gym4real_elevator_v0`
    - `models/ppo_gym4real_elevator_v0_best/best_model.zip`
  - Recreates the appropriate environment (with the same settings as during training).
  - Reports:
    - Mean reward
    - Reward standard deviation
    - Episode lengths
  - Has a `--render` flag to visually inspect the agent if the env supports rendering.

---
