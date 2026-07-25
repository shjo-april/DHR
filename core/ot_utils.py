import torch

from torch.nn import functional as F

class OptimalTransport:
    def __init__(self, max_iter=100, eps=0.1):
        self.max_iter = max_iter
        self.eps = eps

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        
        thresh = 1e-2
        for _ in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T
    
    def apply(self, sim):
        M, N = sim.shape

        # set a cosine distance
        wdist = 1.0 - sim.unsqueeze(0) # sim (-1~1), wdist (0~2)

        # set two distributions (default: uniform)
        xx = torch.zeros(1, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy = torch.zeros(1, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)
        
        # calculate the optimal transport plans (T)
        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK, xx, yy) # B*C, M, N
        
        min_v = torch.min(T[0], dim=0, keepdim=True)[0]
        max_v = torch.max(T[0], dim=0, keepdim=True)[0]
        return (T[0] - min_v) / (max_v - min_v).clip(min=1e-5) 