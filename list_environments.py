"""
Helper script to list all available Gymnasium/Gym environments.
This helps identify the correct name for the gym4real environment.
"""

import gymnasium as gym
import os
import sys

def list_all_environments():
    """List all registered Gymnasium environments, with focus on gym4real."""
    
    print("=" * 60)
    print("Available Gymnasium Environments")
    print("=" * 60)
    
    # Try to import gym4real
    try:
        import gym4real
        print("\n[OK] Successfully imported gym4real package")
    except ImportError:
        print("\n[WARNING] gym4real package not found in Python path.")
        print("Trying to import from local directory...")
        gym4real_path = os.path.join(os.path.dirname(__file__), 'gym4ReaL')
        if os.path.exists(gym4real_path):
            sys.path.insert(0, gym4real_path)
            try:
                import gym4real
                print("[OK] Successfully imported gym4real from local directory")
            except ImportError as e:
                print(f"[ERROR] Failed to import gym4real: {e}")
                gym4real = None
        else:
            print("[ERROR] gym4ReaL directory not found")
            gym4real = None
    
    # Get all registered environments
    try:
        # Try new gymnasium API
        all_envs = list(gym.envs.registry.values())
        all_env_names = [env_spec.id for env_spec in all_envs]
    except (AttributeError, TypeError):
        # Fallback for different API versions
        all_env_names = list(gym.envs.registry.keys())
        all_envs = [gym.envs.registry[env_id] for env_id in all_env_names]
    
    # Filter for gym4real environments
    gym4real_envs = []
    
    for env_id in all_env_names:
        # Check for gym4real environments
        if 'gym4real' in env_id.lower():
            gym4real_envs.append(env_id)
    
    print(f"\nTotal environments found: {len(all_env_names)}")
    
    if gym4real_envs:
        print(f"\n{'=' * 60}")
        print(f"gym4real Environments ({len(gym4real_envs)} found):")
        print("=" * 60)
        for env_id in sorted(gym4real_envs):
            print(f"  [OK] {env_id}")
            # Try to get more info
            try:
                test_env = gym.make(env_id)
                obs_space = test_env.observation_space
                act_space = test_env.action_space
                print(f"    - Observation: {obs_space}")
                print(f"    - Action: {act_space}")
                test_env.close()
            except Exception as e:
                print(f"    - Could not instantiate: {str(e)[:100]}")
    else:
        print("\n[WARNING] No gym4real environments found.")
        print("\nExpected environments:")
        expected = [
            'gym4real/elevator-v0',
            'gym4real/dam-v0',
            'gym4real/microgrid-v0',
            'gym4real/wds-v0',
            'gym4real/robofeeder-picking-v0',
            'gym4real/robofeeder-picking-v1',
            'gym4real/robofeeder-planning',
            'gym4real/TradingEnv-v0',
        ]
        for env_id in expected:
            print(f"  - {env_id}")
        print("\nTo register environments:")
        print("1. Make sure gym4real is installed: pip install -e gym4ReaL/")
        print("2. Or add gym4ReaL to Python path")
    
    print("\n" + "=" * 60)
    print("Other environments containing 'real' or '4real':")
    print("=" * 60)
    real_envs = [eid for eid in all_env_names if ('real' in eid.lower() or '4real' in eid.lower()) and 'gym4real' not in eid.lower()]
    if real_envs:
        for env_id in sorted(real_envs):
            print(f"  - {env_id}")
    else:
        print("  None found")
    
    print("\n" + "=" * 60)
    print("First 20 registered environments (alphabetically):")
    print("=" * 60)
    for env_id in sorted(all_env_names)[:20]:
        print(f"  - {env_id}")
    
    if len(all_env_names) > 20:
        print(f"\n... and {len(all_env_names) - 20} more environments")
    
    return gym4real_envs if gym4real_envs else None

if __name__ == "__main__":
    list_all_environments()

