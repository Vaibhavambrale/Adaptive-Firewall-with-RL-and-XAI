from app.core.environment.firewall_env import FirewallEnv

def main():
    env = FirewallEnv(n_features=6, max_steps=10, malicious_prob=0.2, seed=42)
    obs, _ = env.reset()
    print("Initial obs shape:", obs.shape)
    for i in range(8):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {i+1:02d} | action={action} reward={reward:.2f} label={info['label']}")
        if terminated:
            print("terminated after step", i+1)
            break
    print("env test finished")

if __name__ == "__main__":
    main()
