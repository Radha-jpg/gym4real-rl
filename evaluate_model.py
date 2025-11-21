"""
Script to evaluate and test a trained RL model on the gym4real environment.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
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

def evaluate_model(model_path, env_id=None, n_episodes=10, render=False, deterministic=True):
    """
    Evaluate a trained model on the gym4real environment.
    
    Args:
        model_path: Path to the saved model
        env_id: Environment ID (if None, will try to infer from model or use default)
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
    """
    
    print("=" * 60)
    print("Evaluating Trained Model")
    print("=" * 60)
    
    # Check for model file (could be .zip or in a best_model directory)
    model_file = model_path + ".zip"
    best_model_file = os.path.join(model_path + "_best", "best_model.zip")
    
    actual_model_path = None
    
    if os.path.exists(model_file):
        actual_model_path = model_path
    elif os.path.exists(best_model_file):
        actual_model_path = best_model_file.replace(".zip", "")
        print(f"Using best model from {best_model_file}")
    elif os.path.exists(model_path):
        # Check if it's a directory with best_model.zip
        best_in_dir = os.path.join(model_path, "best_model.zip")
        if os.path.exists(best_in_dir):
            actual_model_path = best_in_dir.replace(".zip", "")
            print(f"Using best model from {best_in_dir}")
        else:
            # List available models
            print(f"\nError: Model file not found at {model_path}.zip or {best_model_file}")
            print("\nAvailable models in models/ directory:")
            if os.path.exists("models"):
                for item in os.listdir("models"):
                    item_path = os.path.join("models", item)
                    if os.path.isfile(item_path) and item.endswith(".zip"):
                        print(f"  - {item_path.replace('.zip', '')}")
                    elif os.path.isdir(item_path):
                        best_model = os.path.join(item_path, "best_model.zip")
                        if os.path.exists(best_model):
                            print(f"  - {item_path} (contains best_model.zip)")
            print("\nPlease train a model first using train_ppo.py or use one of the available models above.")
            return None
    else:
        # List available models
        print(f"\nError: Model file not found at {model_path}.zip or {best_model_file}")
        print("\nAvailable models in models/ directory:")
        if os.path.exists("models"):
            for item in os.listdir("models"):
                item_path = os.path.join("models", item)
                if os.path.isfile(item_path) and item.endswith(".zip"):
                    print(f"  - {item_path.replace('.zip', '')}")
                elif os.path.isdir(item_path):
                    best_model = os.path.join(item_path, "best_model.zip")
                    if os.path.exists(best_model):
                        print(f"  - {item_path} (contains best_model.zip)")
        print("\nPlease train a model first using train_ppo.py or use one of the available models above.")
        return None
    
    if actual_model_path is None:
        print(f"\nError: Could not locate model file.")
        return None
    
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
    
    # Infer environment ID from model path if not provided
    if env_id is None:
        # Try to infer from model path
        if 'elevator' in model_path:
            env_id = 'gym4real/elevator-v0'
        elif 'dam' in model_path:
            env_id = 'gym4real/dam-v0'
        elif 'microgrid' in model_path:
            env_id = 'gym4real/microgrid-v0'
        elif 'wds' in model_path:
            env_id = 'gym4real/wds-v0'
        else:
            env_id = 'gym4real/elevator-v0'  # Default
    
    def create_env(env_id):
        """Helper function to create environment with appropriate settings."""
        import os
        if 'elevator' in env_id:
            from gym4real.envs.elevator.utils import parameter_generator
            gym4real_path = os.path.join(os.path.dirname(__file__), 'gym4ReaL', 'gym4real', 'envs', 'elevator', 'world.yaml')
            params = parameter_generator(world_options=gym4real_path)
            return gym.make(env_id, **{'settings': params})
        elif 'dam' in env_id:
            from gym4real.envs.dam.utils import parameter_generator
            params = parameter_generator(
                world_options='gym4real/envs/dam/world_train.yaml',
                lake_params='gym4real/envs/dam/lake.yaml'
            )
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
        
        # Load the trained model
        print(f"\nLoading model from {actual_model_path}...")
        model = PPO.load(actual_model_path, env=env)
        
        print(f"Model loaded successfully!")
        print(f"Environment: {env.spec.id if env.spec else env_id}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Evaluate the policy
        print(f"\nEvaluating policy over {n_episodes} episodes...")
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            render=render
        )
        
        print(f"\nEvaluation Results:")
        print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Over {n_episodes} episodes")
        
        # Test the model interactively
        if render:
            print("\nRunning interactive test (press Ctrl+C to stop)...")
            test_model_interactive(model, env, max_steps=1000, render=render)
        else:
            print("\nTo see the model in action, run with --render flag")
        
        env.close()
        
        return mean_reward, std_reward
        
    except (gym.error.Error, ImportError) as e:
        print(f"\nError: Could not create gym4real environment.")
        print(f"Details: {e}")
        print(f"\nAvailable environments:")
        for env in AVAILABLE_ENVS:
            print(f"  - {env}")
        return None
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_interactive(model, env, max_steps=1000, render=False):
    """
    Test the model interactively with rendering.
    
    Args:
        model: The trained model
        env: The environment
        max_steps: Maximum number of steps to run
        render: Whether to render the environment
    """
    
    obs, info = env.reset()
    total_reward = 0
    episode_count = 0
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if render:
            env.render()
        
        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} completed. Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_gym4real_elevator_v0",
        help="Path to the model file (without .zip extension)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=AVAILABLE_ENVS,
        help="Environment ID (if not provided, will try to infer from model path)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        env_id=args.env,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic
    )

