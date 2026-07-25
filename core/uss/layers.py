import torch
from torch import nn
from torch.nn import functional as F

def transform(x, ratio):
    """
    B, P, D => B, D, root(P), root(P)
    Ex) 128, 400, 768 => 128, 768, 20, 20
    """
    B, P, D = x.shape
    return x.permute(0, 2, 1).view(B, D, *ratio)

def untransform(x):
    """
    B, D, P, P => B, P*P, D,
    Ex) 128, 768, 20, 20 => 128, 400, 768
    """
    return x.view(*x.shape[:2], -1).permute(0, 2, 1)

def cos_distance_matrix(z, c):
    z_flattened = z.contiguous().view(-1, z.shape[-1])
    norm_z = F.normalize(z_flattened, dim=1)
    norm_embed = F.normalize(c, dim=1)
    return torch.einsum("ab,cb->ac", norm_z, norm_embed).view(*z.shape[:-1], -1)

def codebook_index(z, c):
    # computing distance & codebook index
    return cos_distance_matrix(z, c).argmax(dim=2)

def vqt(z, c):
    """
    Return Vector-Quantized Tensor
    """
    return c[codebook_index(z, c)].view(*z.shape[:-1], c.shape[1])

class TRDecoder(nn.Module):
    def __init__(self, dim, reduced_dim, hidden_dim=2048, nhead=1, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.f1 = nn.Conv2d(dim, reduced_dim, (1, 1))
        self.f2 = nn.Sequential(nn.Conv2d(dim, dim, (1, 1)), nn.ReLU(), nn.Conv2d(dim, reduced_dim, (1, 1)))
    
    def forward(self, tgt, memory, pos, ratio):
        q = k = tgt + pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + pos, key=memory, value=memory)[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = memory + self.norm3(tgt)
        tgt = transform(tgt.transpose(0, 1), ratio)
        tgt = self.f1(tgt) + self.f2(tgt)
        tgt = untransform(tgt)
        return tgt

class CauseDecoder(nn.Module):
    def __init__(self, num_queries, dim, reduced_dim):
        super().__init__()
        self.codebook = None

        # TR decoder
        self.query_pos = nn.Parameter(torch.randn(num_queries, dim))
        self.tr = TRDecoder(dim, reduced_dim)

    def forward(self, feat, ratio, pe):
        feat = self.tr(
            vqt(feat, self.codebook).transpose(0, 1), 
            feat.transpose(0, 1), pe.unsqueeze(1), ratio
        )
        return transform(feat, ratio)

class Segment_TR(nn.Module):
    def __init__(self, dim, reduced_dim, num_queries):
        super().__init__()

        # TR Decoder Head for training
        # self.head = CauseDecoder(num_queries, dim, reduced_dim)
        
        # TR Decoder EMA Head
        self.head_ema = CauseDecoder(num_queries, dim, reduced_dim)
        