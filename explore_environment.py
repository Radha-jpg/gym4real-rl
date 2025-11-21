"""
Script to explore and understand the gym4ReaL environment.
This helps us get familiar with the environment's structure, 
observation space, action space, and reward structure.
"""

import gymnasium as gym
import numpy as np
import argparse

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

def explore_environment(env_id='gym4real/elevator-v0'):
    """
    Explore the gym4real environment to understand its properties.
    
    Args:
        env_id: The environment ID to explore (default: 'gym4real/elevator-v0')
    """
    
    print("=" * 60)
    print(f"Exploring gym4real Environment: {env_id}")
    print("=" * 60)
    
    try:
        # Import gym4real to register environments
        try:
            import gym4real
        except ImportError:
            print("\nWarning: gym4real package not found in Python path.")
            print("Trying to import from local directory...")
            import sys
            import os
            # Add the gym4ReaL directory to path if it exists
            gym4real_path = os.path.join(os.path.dirname(__file__), 'gym4ReaL')
            if os.path.exists(gym4real_path):
                sys.path.insert(0, gym4real_path)
                import gym4real
                print("Successfully imported gym4real from local directory.")
            else:
                raise ImportError("Could not find gym4real package.")
        
        # Create the environment with appropriate settings
        # Some environments require settings parameters
        if 'elevator' in env_id:
            from gym4real.envs.elevator.utils import parameter_generator
            import os
            # Get the correct path to world.yaml
            gym4real_path = os.path.join(os.path.dirname(__file__), 'gym4ReaL', 'gym4real', 'envs', 'elevator', 'world.yaml')
            params = parameter_generator(world_options=gym4real_path)
            env = gym.make(env_id, **{'settings': params})
        elif 'dam' in env_id:
            from gym4real.envs.dam.utils import parameter_generator
            params = parameter_generator(
                world_options='gym4real/envs/dam/world_train.yaml',
                lake_params='gym4real/envs/dam/lake.yaml'
            )
            env = gym.make(env_id, settings=params)
        elif 'microgrid' in env_id:
            from gym4real.envs.microgrid.utils import parameter_generator
            params = parameter_generator()
            env = gym.make(env_id, **{'settings': params})
        elif 'wds' in env_id:
            from gym4real.envs.wds.utils import parameter_generator
            params = parameter_generator()
            env = gym.make(env_id, **{'settings': params})
        else:
            # Try without settings first
            try:
                env = gym.make(env_id)
            except TypeError:
                # If it fails, try with empty settings
                env = gym.make(env_id, settings={})
        
        print("\n1. Environment Information:")
        print(f"   Environment ID: {env.spec.id if env.spec else 'N/A'}")
        print(f"   Environment: {env}")
        
        print("\n2. Observation Space:")
        print(f"   Type: {type(env.observation_space)}")
        print(f"   Shape: {env.observation_space.shape}")
        print(f"   Dtype: {env.observation_space.dtype}")
        print(f"   Sample observation: {env.observation_space.sample()}")
        
        print("\n3. Action Space:")
        print(f"   Type: {type(env.action_space)}")
        if hasattr(env.action_space, 'n'):
            print(f"   Number of actions: {env.action_space.n}")
        if hasattr(env.action_space, 'shape'):
            print(f"   Shape: {env.action_space.shape}")
        print(f"   Dtype: {env.action_space.dtype}")
        print(f"   Sample action: {env.action_space.sample()}")
        
        print("\n4. Testing Environment Step:")
        obs, info = env.reset()
        print(f"   Initial observation shape: {obs.shape if isinstance(obs, np.ndarray) else type(obs)}")
        print(f"   Initial observation sample: {obs}")
        print(f"   Info: {info}")
        
        # Take a few random steps
        print("\n5. Testing Random Actions:")
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"   Step {step + 1}: Action={action}, Reward={reward:.4f}, Done={terminated or truncated}")
            if terminated or truncated:
                obs, info = env.reset()
                print(f"   Episode ended, resetting environment")
        
        print(f"\n   Total reward over 5 steps: {total_reward:.4f}")
        
        print("\n6. Environment Metadata:")
        if hasattr(env, 'metadata'):
            print(f"   Metadata: {env.metadata}")
        if hasattr(env, 'reward_range'):
            print(f"   Reward range: {env.reward_range}")
        
        env.close()
        print("\n" + "=" * 60)
        print("Environment exploration complete!")
        print("=" * 60)
        
    except (gym.error.Error, ImportError) as e:
        print(f"\nError: Could not create gym4real environment.")
        print(f"Details: {e}")
        print(f"\nAvailable environments:")
        for env in AVAILABLE_ENVS:
            print(f"  - {env}")
        print("\nPlease ensure gym4real is installed correctly.")
        print("If using local installation, make sure the gym4ReaL directory is accessible.")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore gym4real environment")
    parser.add_argument(
        "--env",
        type=str,
        default='gym4real/elevator-v0',
        choices=AVAILABLE_ENVS,
        help="Environment ID to explore"
    )
    args = parser.parse_args()
    explore_environment(args.env)

