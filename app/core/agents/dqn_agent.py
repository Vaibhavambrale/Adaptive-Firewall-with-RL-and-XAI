from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os

class DQNAgent:
    def __init__(self, env, model_dir="models", log_dir="logs"):
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        # Wrap env with Monitor so SB3 can record stats
        self.env = Monitor(env)
        self.model_dir = model_dir
        self.log_dir = log_dir
        # Create DQN model with default params
        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,  # ensures logs are printed
            tensorboard_log=self.log_dir
        )

    def train(self, total_timesteps=10000, checkpoint_freq=5000):
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=self.model_dir,
            name_prefix="dqn_firewall"
        )
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
        final_path = os.path.join(self.model_dir, "dqn_firewall_final")
        self.model.save(final_path)
        return final_path
