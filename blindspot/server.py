import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time

class HighwayRiskAssessmentEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(HighwayRiskAssessmentEnv, self).__init__()
        
        # Define the observation and action space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # Example: 3 possible actions

        self.state = np.random.random((5, 5))  # Random initial state
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.random((5, 5))
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        reward = np.random.random()  # Example reward logic
        terminated = self.steps >= self.max_steps
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self, mode="human"):
        print(f"Rendering... Steps: {self.steps}")

    def close(self):
        pass

# ✅ Environment setup
def make_env():
    env = HighwayRiskAssessmentEnv()
    return env

# ✅ Wrap the environment with DummyVecEnv
env = DummyVecEnv([make_env])

# ✅ Train the model
model = PPO("MlpPolicy", env, verbose=1)

print("🚀 Training RL model...")
model.learn(total_timesteps=50000)

# ✅ Save the model
model.save("highway_risk_model")
print("✅ Model saved successfully!")

# 🚦 **Testing Section**
print("\n🔎 Running Testing Episodes...")

num_episodes = 5  # Number of test episodes

for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0

    print(f"\n🎯 Episode {episode + 1}")

    for step in range(50):  # Simulate 50 steps per episode
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)

        total_reward += reward[0]

        print(f"Step: {step + 1}, Action: {action[0]}, Reward: {reward[0]:.4f}")

        if done[0] or truncated[0]:
            print(f"✅ Episode finished with total reward: {total_reward:.2f}")
            break

        time.sleep(0.2)  # Add a delay to simulate real-time testing

# ✅ Close the environment
env.close()
print("\n🎉 Simulation complete!")
