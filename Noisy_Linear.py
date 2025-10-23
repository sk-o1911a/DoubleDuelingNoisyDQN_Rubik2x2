import math
import torch
import torch.nn as nn

class NoisyLinear(nn.Module):

    # base: y = xW^T + b
    # noisy: y = x(muy_W + sigma_w * epsilon_w)^T + (muy_b + sigma_b * epsilon_b) as muy: mean, sigma: std, epsilon: noise
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()

        self.in_f = in_features
        self.out_f = out_features

        self.mu_w    = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b    = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))

        # buffers để chứa noise mỗi lần forward
        self.register_buffer("eps_w", torch.empty(out_features, in_features))
        self.register_buffer("eps_b", torch.empty(out_features))

        self.reset_parameters(sigma0)

    def reset_parameters(self, sigma0):
        mu_range = 1 / math.sqrt(self.in_f)
        nn.init.uniform_(self.mu_w,-mu_range, mu_range)
        nn.init.uniform_(self.mu_b, -mu_range, mu_range)

        nn.init.constant_(self.sigma_w, sigma0 / math.sqrt(self.in_f))
        nn.init.constant_(self.sigma_b, sigma0 / math.sqrt(self.out_f))

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()  # factorized trick

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate new noise each forward pass (automatic exploration)
        eps_in = torch.randn(self.in_f, device=x.device, dtype=x.dtype)
        eps_out = torch.randn(self.out_f, device=x.device, dtype=x.dtype)
        f_in = self._f(eps_in)
        f_out = self._f(eps_out)

        # outer product -> factorized noise
        self.eps_w = torch.ger(f_out, f_in)
        self.eps_b = f_out

        if not self.training:
            w = self.mu_w
            b = self.mu_b
            return x @ w.t() + b

        w = self.mu_w + self.sigma_w * self.eps_w
        b = self.mu_b + self.sigma_b * self.eps_b
        return x @ w.t() + b
