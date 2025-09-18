# app/core/inference/predictor.py
import os
import numpy as np
from stable_baselines3 import DQN

class Predictor:
    """
    Loads a saved Stable-Baselines3 DQN model and offers a predict(observation) method.
    """

    def __init__(self, model_path: str = "models/dqn_firewall_final.zip"):
        if not os.path.exists(model_path):
            # try with appended .zip (SB3 often saves with .zip)
            if os.path.exists(model_path + ".zip"):
                model_path = model_path + ".zip"
            else:
                raise FileNotFoundError(f"Model not found at '{model_path}' or '{model_path}.zip'")

        # load the model
        try:
            self.model = DQN.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model at {model_path}: {e}")

    def predict(self, observation):
        """
        observation: 1D array-like of shape (n_features,)
        returns: integer action (e.g., 0=noop,1=block,2=throttle)
        """
        obs = np.asarray(observation, dtype=np.float32)
        # SB3 expects a 1D or 2D array; predict accepts a single observation
        action, _state = self.model.predict(obs, deterministic=True)
        return int(action)
