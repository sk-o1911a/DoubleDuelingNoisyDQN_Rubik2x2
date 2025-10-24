import torch
import torch.nn.functional as F
import copy
from Q_Network import QNetwork

class DQNAgent:
    def __init__(self,
                 input_dim=144,
                 num_actions=12,
                 lr=1e-3,
                 gamma=0.99,
                 tau=0.01,          # hệ số soft-update
                 dueling=True,
                 noisy=True,
                 sigma0=0.5,
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.num_actions = num_actions
        self.q = QNetwork(input_dim=input_dim,
                        num_actions=num_actions,
                        dueling=dueling,
                        noisy=noisy,
                        sigma0=sigma0).to(self.device)

        self.target = copy.deepcopy(self.q).to(self.device)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, action_mask=None, eval_mode=False):
        if eval_mode:
            self.q.eval()
        else:
            self.q.train()

        x = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        q = self.q(x)[0]

        if action_mask is not None:
            mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
            q[~mask] = -float("inf")

        a = int(torch.argmax(q).item())
        return a

    def update(self, batch, double=True, max_grad_norm=10.0):
        s, a, r, s2, d = batch

        self.q.train()
        q_all = self.q(s)                      # (B, A)
        q_sa  = q_all.gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            if double:
                a2 = torch.argmax(self.q(s2), dim=1)        # chọn a2 từ online
                q2 = self.target(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
            else:
                q2 = torch.max(self.target(s2), dim=1).values

            target = r + (~d).float() * self.gamma * q2

        #huber loss
        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_grad_norm)
        self.opt.step()

        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target.parameters()):
                tp.data.lerp_(p.data, self.tau)

        return float(loss.item())

    def save(self, path):
        torch.save({
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
        }, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["q"])
        self.target.load_state_dict(ckpt["target"])
        self.opt.load_state_dict(ckpt["opt"])
