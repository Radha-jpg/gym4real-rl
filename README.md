# Reinforcement Learning Project: gym4real with Stable Baselines3

This project implements and tests a non-tabular Reinforcement Learning algorithm (PPO - Proximal Policy Optimization) on the gym4real environment using Stable Baselines3.

## 📚 Learning Resources

**New to RL? Start here:**
- **[QUICK_THEORY.md](QUICK_THEORY.md)** - Quick theory reference (5 min read)
- **[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)** - Complete end-to-end guide with presentation tips

## Project Structure

```
rlc/
├── requirements.txt          # Python dependencies
├── explore_environment.py   # Script to explore and understand gym4real
├── train_ppo.py            # Training script using PPO algorithm
├── evaluate_model.py       # Evaluation and testing script
├── list_environments.py    # Helper script to list available environments
├── gym4ReaL/               # gym4real package (local installation)
└── README.md               # This file
```

## Setup

### 1. Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- `stable-baselines3[extra]` - RL algorithms library
- `gymnasium` - OpenAI Gym API (successor to gym)
- `numpy` - Numerical computing
- `torch` - PyTorch for neural networks
- `tensorboard` - For training visualization

### 2. Install gym4real Environment

The gym4real package is already included in this repository in the `gym4ReaL/` directory. Install it using:

```bash
pip install -e gym4ReaL/
```

This will install the gym4real package and register all available environments:
- `gym4real/elevator-v0` - **Elevator control (recommended for beginners)** - Simplest environment, good for testing
- `gym4real/dam-v0` - Dam water management
- `gym4real/microgrid-v0` - Microgrid energy management
- `gym4real/wds-v0` - Water distribution system
- `gym4real/robofeeder-picking-v0` - Robot feeder picking (v0) - Requires MuJoCo
- `gym4real/robofeeder-picking-v1` - Robot feeder picking (v1) - Requires MuJoCo
- `gym4real/robofeeder-planning` - Robot feeder planning - Requires MuJoCo
- `gym4real/TradingEnv-v0` - Trading environment

**Note:** 
- Some environments may have additional requirements. Check the `gym4ReaL/gym4real/envs/<env_name>/requirements.txt` files for environment-specific dependencies.
- **For Week 8 assignment, we recommend starting with `gym4real/elevator-v0`** as it's the simplest and doesn't require additional dependencies.
- Some environments (like robofeeder) require MuJoCo which may need separate installation.

## Usage

### Step 1: List Available Environments

First, check which gym4real environments are available:

```bash
python list_environments.py
```

This will show all registered gym4real environments and their properties.

### Step 2: Explore the Environment

Before training, it's important to understand the environment structure:

```bash
# Explore the default environment (elevator-v0)
python explore_environment.py

# Or specify a different environment
python explore_environment.py --env gym4real/dam-v0
```

This script will display:
- Observation space (type, shape, sample)
- Action space (type, number of actions, sample)
- Environment metadata
- Test random actions to see how the environment responds

### Step 3: Train the Model

Train a PPO model on a gym4real environment:

```bash
# Train on default environment (elevator-v0)
python train_ppo.py

# Train on a specific environment
python train_ppo.py --env gym4real/dam-v0

# Customize training timesteps and learning rate
python train_ppo.py --env gym4real/elevator-v0 --timesteps 200000 --lr 1e-4
```

**Customizing Training:**

You can modify the training parameters in `train_ppo.py` or use command-line arguments:

```python
model = train_ppo_model(
    env_id='gym4real/elevator-v0',  # Environment ID
    total_timesteps=100000,           # Total training steps
    learning_rate=3e-4,              # Learning rate
    n_steps=2048,                    # Steps per update
    batch_size=64,                   # Batch size
    n_epochs=10,                     # Training epochs per update
    gamma=0.99,                      # Discount factor
    verbose=1                        # Verbosity level
)
```

**Training Output:**
- Model checkpoints saved in `models/checkpoints/`
- Best model saved in `models/ppo_<env_name>_best/`
- Final model saved in `models/ppo_<env_name>.zip`
- Training logs in `logs/`
- TensorBoard logs in `logs/tensorboard/`

**Monitor Training with TensorBoard:**

```bash
tensorboard --logdir logs/tensorboard/
```

Then open your browser to `http://localhost:6006`

### Step 4: Evaluate the Model

Evaluate the trained model:

```bash
# Basic evaluation (no rendering) - uses default model
python evaluate_model.py

# Use the final trained model
python evaluate_model.py --model models/ppo_gym4real_elevator_v0

# Use the best model (recommended - best performance during training)
python evaluate_model.py --model models/ppo_gym4real_elevator_v0_best

# Evaluation with rendering
python evaluate_model.py --model models/ppo_gym4real_elevator_v0_best --render

# Custom number of episodes
python evaluate_model.py --model models/ppo_gym4real_elevator_v0_best --episodes 20

# Specify environment explicitly
python evaluate_model.py --model models/ppo_gym4real_elevator_v0 --env gym4real/elevator-v0

# Use stochastic actions instead of deterministic
python evaluate_model.py --model models/ppo_gym4real_elevator_v0_best --stochastic
```

**Note:** If you get an error about model not found, the script will automatically list all available models in the `models/` directory.

## Algorithm: PPO (Proximal Policy Optimization)

PPO is a **non-tabular** policy gradient method that:
- Uses neural networks to approximate the policy and value functions
- Works with both discrete and continuous action spaces
- Is sample-efficient and stable
- Clips the policy update to prevent large policy changes

**Why PPO is non-tabular:**
- Tabular methods (like Q-learning with Q-tables) store values for each state-action pair
- PPO uses function approximation (neural networks) to generalize across states
- This allows it to handle large or continuous state/action spaces

**Policy Selection:**
- The training script automatically detects the observation space type
- For **Dict observation spaces** (like gym4real/elevator-v0): Uses `MultiInputPolicy`
- For **Box/Discrete observation spaces**: Uses `MlpPolicy`
- This ensures compatibility with all gym4real environments

## Project Requirements Checklist

✅ **Get familiar with the environment**
- Use `explore_environment.py` to understand observation/action spaces

✅ **Use a non-tabular RL algorithm**
- PPO (Proximal Policy Optimization) is implemented
- Uses neural network function approximation

✅ **Test on gym4real**
- Training script works with gym4real environments
- Evaluation script tests the trained model
- Supports multiple gym4real environments (elevator, dam, microgrid, wds, etc.)

## Customization (Optional)

**For Week 8 Assignment: Customization is NOT required.**

The gym4real environments come with well-designed reward functions that work out of the box. However, if you want to experiment:

- **Reward Function:** The environments have built-in rewards (see `CUSTOMIZATION_GUIDE.md` for details)
- **Environment Settings:** Can be customized via parameter generators (reward coefficients, environment parameters, etc.)
- **Custom Wrappers:** Can be added for advanced reward shaping

See `CUSTOMIZATION_GUIDE.md` for detailed customization options (optional, not required for assignment).

## Troubleshooting

### Environment Not Found Error

If you get an error like `gym.error.Error: Could not create gym4real environment`:

1. **Install gym4real:**
   ```bash
   pip install -e gym4ReaL/
   ```

2. **Verify gym4real is installed and environments are registered:**
   ```bash
   python list_environments.py
   ```

3. **Check if the environment ID is correct:**
   - Use the format: `gym4real/<env-name>-v0`
   - Available environments: `elevator-v0`, `dam-v0`, `microgrid-v0`, `wds-v0`, etc.
   - Run `python list_environments.py` to see all available environments

4. **If using a local installation, make sure the gym4ReaL directory is accessible:**
   - The scripts will automatically try to import from the local `gym4ReaL/` directory
   - Make sure you're running scripts from the project root directory

### Training Takes Too Long

- Reduce `total_timesteps` for faster testing
- Reduce `n_steps` to update more frequently (but may be less stable)
- Use a smaller network architecture (modify policy in `train_ppo.py`)

### Out of Memory

- Reduce `batch_size` and `n_steps`
- Use a smaller network architecture

## Additional Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Notes

- The default training uses 100,000 timesteps, which may take some time depending on your hardware
- For faster testing, you can reduce `total_timesteps` to 10,000-20,000
- The model automatically saves checkpoints during training
- Best model (based on evaluation) is saved separately for easy access

