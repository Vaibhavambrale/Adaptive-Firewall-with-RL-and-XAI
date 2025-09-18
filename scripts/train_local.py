from app.core.environment.firewall_env import FirewallEnv
from app.core.agents.dqn_agent import DQNAgent

def main():
    env = FirewallEnv(n_features=8, max_steps=50, malicious_prob=0.2, seed=123)
    agent = DQNAgent(env)
    print("Starting training...")
    model_path = agent.train(total_timesteps=2000, checkpoint_freq=1000)
    print(f"Training finished. Model saved at {model_path}")

if __name__ == "__main__":
    main()
