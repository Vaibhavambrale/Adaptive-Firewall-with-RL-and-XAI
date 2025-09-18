# scripts/test_predict.py
from app.core.environment.firewall_env import FirewallEnv
from app.core.inference.predictor import Predictor
import numpy as np

def main():
    # create a small env to sample a real-shaped observation
    env = FirewallEnv(n_features=8, max_steps=10, malicious_prob=0.1, seed=999)
    obs, _ = env.reset()
    print("Sample observation (shape):", obs.shape)
    print("Sample observation:", obs.tolist())

    # load predictor (adjust path if your model is in another location)
    predictor = Predictor(model_path="models/dqn_firewall_final.zip")
    action = predictor.predict(obs)
    print("Predicted action:", action)
    action_meaning = {0: "noop", 1: "block", 2: "throttle"}
    print("Action meaning:", action_meaning.get(action, "unknown"))

if __name__ == "__main__":
    main()
