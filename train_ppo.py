"""
Training script using PPO (Proximal Policy Optimization) algorithm.
PPO is a non-tabular RL algorithm suitable for continuous and discrete action spaces.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys

# Available gym4real environments
AVAILABLE_ENVS = [
    'gym4real/elevator-v0',
    'gym4real/dam-v0',
    'gym4real/microgrid-v0',
    'gym4real/wds-v0',
    'gym4real/robofeeder-picking-v0',
    'gym4real/robofeeder-picking-v1',
    'gym4real/robofeeder-planning',
    'gym4real/TradingEnv-v0',
]

def train_ppo_model(
    env_id='gym4real/elevator-v0',
    total_timesteps=100000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    save_path=None
):
    """
    Train a PPO model on the gym4real environment.
    
    Args:
        env_id: Environment ID (default: 'gym4real/elevator-v0')
        total_timesteps: Total number of timesteps to train
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps to run per update
        batch_size: Minibatch size
        n_epochs: Number of epochs when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range: Clipping parameter for PPO
        ent_coef: Entropy coefficient for exploration
        verbose: Verbosity level (0: no output, 1: info, 2: debug)
        save_path: Path to save the trained model (default: models/ppo_{env_name})
    """
    
    print("=" * 60)
    print(f"Training PPO Model on gym4real Environment: {env_id}")
    print("=" * 60)
    
    # Import gym4real to register environments
    try:
        import gym4real
    except ImportError:
        print("\nWarning: gym4real package not found in Python path.")
        print("Trying to import from local directory...")
        gym4real_path = os.path.join(os.path.dirname(__file__), 'gym4ReaL')
        if os.path.exists(gym4real_path):
            sys.path.insert(0, gym4real_path)
            import gym4real
            print("Successfully imported gym4real from local directory.")
        else:
            raise ImportError("Could not find gym4real package.")
    
    # Set default save path if not provided
    if save_path is None:
        env_name = env_id.replace('/', '_').replace('-', '_')
        save_path = f"models/ppo_{env_name}"
    
    # Create directories for logs and models
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("evaluations", exist_ok=True)
    
    def create_env(env_id):
        """Helper function to create environment with appropriate settings."""
        if 'elevator' in env_id:
            from gym4real.envs.elevator.utils import parameter_generator
            gym4real_path = os.path.join(os.path.dirname(__file__), 'gym4ReaL', 'gym4real', 'envs', 'elevator', 'world.yaml')
            params = parameter_generator(world_options=gym4real_path)
            return gym.make(env_id, **{'settings': params})
        elif 'dam' in env_id:
            from gym4real.envs.dam.utils import parameter_generator
            dam_world = os.path.join(os.path.dirname(__file__), 'gym4ReaL', 'gym4real', 'envs', 'dam', 'world_train.yaml')
            dam_lake = os.path.join(os.path.dirname(__file__), 'gym4ReaL', 'gym4real', 'envs', 'dam', 'lake.yaml')
            params = parameter_generator(world_options=dam_world, lake_params=dam_lake)
            return gym.make(env_id, settings=params)
        elif 'microgrid' in env_id:
            from gym4real.envs.microgrid.utils import parameter_generator
            params = parameter_generator()
            return gym.make(env_id, **{'settings': params})
        elif 'wds' in env_id:
            from gym4real.envs.wds.utils import parameter_generator
            params = parameter_generator()
            return gym.make(env_id, **{'settings': params})
        else:
            # Try without settings first
            try:
                return gym.make(env_id)
            except TypeError:
                # If it fails, try with empty settings
                return gym.make(env_id, settings={})
    
    try:
        # Create the environment
        env = create_env(env_id)
        
        # Wrap environment with Monitor for logging
        env = Monitor(env, "logs/")
        
        # Create evaluation environment
        eval_env = create_env(env_id)
        eval_env = Monitor(eval_env, "evaluations/")
        
        print(f"\nEnvironment: {env.spec.id if env.spec else env_id}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Determine the appropriate policy based on observation space
        # Dict observation spaces require MultiInputPolicy
        if isinstance(env.observation_space, gym.spaces.Dict):
            policy = 'MultiInputPolicy'
            print("\nDetected Dict observation space - using MultiInputPolicy")
        else:
            policy = 'MlpPolicy'
            print("\nDetected Box/Discrete observation space - using MlpPolicy")
        
        # Initialize PPO model
        print(f"\nInitializing PPO model with {policy}...")
        model = PPO(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            tensorboard_log="logs/tensorboard/"
        )
        
        print(f"Model architecture: {model.policy}")
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{save_path}_best",
            log_path="logs/results/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        env_name_short = env_id.split('/')[-1].replace('-', '_')
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="models/checkpoints/",
            name_prefix=f"ppo_{env_name_short}"
        )
        
        # Train the model
        print(f"\nStarting training for {total_timesteps} timesteps...")
        print("This may take a while. Progress will be logged to logs/")
        print("You can monitor training with: tensorboard --logdir logs/tensorboard/")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save the final model
        print(f"\nSaving final model to {save_path}...")
        model.save(save_path)
        
        print("\nTraining completed!")
        print(f"Model saved to: {save_path}")
        print(f"Best model saved to: {save_path}_best")
        print(f"Logs available in: logs/")
        print(f"TensorBoard logs: logs/tensorboard/")
        
        env.close()
        eval_env.close()
        
        return model
        
    except (gym.error.Error, ImportError) as e:
        print(f"\nError: Could not create gym4real environment.")
        print(f"Details: {e}")
        print(f"\nAvailable environments:")
        for env in AVAILABLE_ENVS:
            print(f"  - {env}")
        print("\nPlease ensure gym4real is installed correctly.")
        return None
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO model on gym4real environment")
    parser.add_argument(
        "--env",
        type=str,
        default='gym4real/elevator-v0',
        choices=AVAILABLE_ENVS,
        help="Environment ID to train on"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total number of timesteps to train"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # You can adjust these hyperparameters as needed
    model = train_ppo_model(
        env_id=args.env,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        verbose=1
    )

