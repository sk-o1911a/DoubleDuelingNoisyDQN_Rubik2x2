import random
import numpy as np
import torch
from collections import deque
from typing import Deque, Tuple


# Define a transition tuple [state, action, reward, next_state, done]
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]

class ReplayBuffer:
    def __init__(self, capacity: int, n_step: int = 1, gamma: float = 0.99):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        # will store n-step transitions
        self._nq: Deque[Transition] = deque()

    def __len__(self) -> int:
        return len(self.buffer)

    def reset_episode(self):
        self._nq.clear()

    def add(self, s: np.ndarray,
            a: int,
            r: float,
            s2: np.ndarray,
            done: bool):

        self._nq.append((s, a, r, s2, done))
        if self.n_step == 1:
            self.buffer.append((s, a, r, s2, done))
            if done:
                self._nq.clear()
            else:
                self._nq.popleft()
            return

        if len(self._nq) < self.n_step and not self._nq[-1][4]:
            return

        R, s0, a0, sN, doneN = 0.0, None, None, None, False
        for i, (si, ai, ri, s2i, di) in enumerate(self._nq):
            if i == 0:
                s0, a0 = si, ai
            R += (self.gamma ** i) * ri
            sN, doneN = s2i, di
            if di:
                break

        self.buffer.append((s0, a0, R, sN, doneN))

        # remove the oldest transition
        if self._nq[0][4]:
            self._nq.clear()
        else:
            self._nq.popleft()

    def sample(self, batch_size: int = 256, device: str = "cpu"):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)

        s_arr = np.asarray(s, dtype=np.float32)
        s2_arr = np.asarray(s2, dtype=np.float32)
        a_arr = np.asarray(a, dtype=np.int64)
        r_arr = np.asarray(r, dtype=np.float32)
        d_arr = np.asarray(d, dtype=np.bool_)

        # pin memory for faster transfer to GPU
        s_t = torch.from_numpy(s_arr).pin_memory()
        a_t = torch.from_numpy(a_arr).pin_memory()
        r_t = torch.from_numpy(r_arr).pin_memory()
        s2_t = torch.from_numpy(s2_arr).pin_memory()
        d_t = torch.from_numpy(d_arr).pin_memory()
        return (
            s_t.to(device, non_blocking=True),
            a_t.to(device, non_blocking=True),
            r_t.to(device, non_blocking=True),
            s2_t.to(device, non_blocking=True),
            d_t.to(device, non_blocking=True),
        )