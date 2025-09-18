# app/core/environment/firewall_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FirewallEnv(gym.Env):
    """
    Minimal simulated firewall environment (MVP).

    Observation: fixed-length numeric vector (n_features)
    Actions: 0 = noop, 1 = block, 2 = throttle
    Reward: +1 if block a malicious flow, -1 if block a benign; small reward/cost for throttle.

    This is purely simulated (no system firewall calls) so it's safe to run locally.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_features: int = 8, max_steps: int = 200, malicious_prob: float = 0.05, seed: int | None = None):
        super().__init__()
        self.n_features = n_features
        self.action_space = spaces.Discrete(3)   # 0: noop, 1: block, 2: throttle
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.max_steps = int(max_steps)
        self.malicious_prob = float(malicious_prob)
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self._obs = None
        self._label = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # follow Gymnasium API
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self._obs, self._label = self._sample_traffic()
        return self._obs, {}

    def step(self, action):
        # compute reward using a simple rule:
        # block on malicious -> +1, block benign -> -1
        # throttle malicious -> +0.5, throttle benign -> -0.2
        # noop -> 0
        if action == 1:  # block
            reward = 1.0 if self._label == 1 else -1.0
        elif action == 2:  # throttle
            reward = 0.5 if self._label == 1 else -0.2
        else:
            reward = 0.0

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False

        # sample next traffic example for next step
        self._obs, self._label = self._sample_traffic()
        info = {"label": int(self._label)}
        return self._obs, float(reward), bool(terminated), bool(truncated), info

    def _sample_traffic(self):
        # Simple synthetic features + binary label (malicious or benign)
        # You will replace this sampling with real feature extractor later.
        features = self.rng.normal(loc=0.0, scale=1.0, size=(self.n_features,)).astype(np.float32)
        label = 1 if self.rng.random() < self.malicious_prob else 0
        return features, label

    def render(self, mode="human"):
        # optional: implement if you want textual printing of state
        print(f"step={self.step_count}, last_label={getattr(self,'_label',None)}")
