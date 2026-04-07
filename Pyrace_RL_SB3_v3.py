import gymnasium as gym
import gym_race
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

VERSION_NAME = 'SB3_PPO_v3'

def main():
    # 1. Create the environment
    # Pyrace-v3 uses gym.spaces.Box for continuous observations and gym.spaces.Discrete for actions
    env = gym.make("Pyrace-v3")
    
    # We unwrap the environment briefly to disable rendering during training for faster processing
    if hasattr(env.unwrapped, 'set_view'):
        env.unwrapped.set_view(False)

    # 2. Instantiate the model
    # We use PPO (Proximal Policy Optimization), a highly advanced Policy Gradient algorithm
    # that handles our existing Discrete(3) action limits seamlessly and learns efficiently.
    print("Initializing SB3 PPO Model...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./runs/{VERSION_NAME}/")

    # 3. Train the model
    print("Training the PPO policy...")
    # 50,000 timesteps as an arbitrary training horizon to demonstrate compatibility
    model.learn(total_timesteps=50_000, progress_bar=True)

    # 4. Save the model
    models_dir = f"models_{VERSION_NAME}"
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "ppo_pyrace")
    model.save(save_path)
    print(f"Model successfully saved to {save_path}.zip")

    # 5. Evaluate and render graphically
    print("Evaluating trained policy...")
    
    # Turning rendering back on to visualize the outcome
    if hasattr(env.unwrapped, 'set_view'):
        env.unwrapped.set_view(True)
        # mode=2 is continuous view
        env.unwrapped.pyrace.mode = 2
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=False)
    print(f"Evaluation complete! Mean reward: {mean_reward} +/- {std_reward}")

    print("Playing continuous demo run with trained PPO agent...")
    obs, _ = env.reset()
    for _ in range(2000):
        # Setting deterministic to true makes PPO use its purely exploitative optimal policy
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
