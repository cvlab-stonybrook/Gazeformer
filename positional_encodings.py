import math
import torch
from torch import nn

class PositionEmbeddingSine1d(nn.Module):
    def __init__(self, max_len, hidden_dim=768, temperature=1000, normalize=False, scale=None, device = "cuda:0"):
        super(PositionEmbeddingSine1d, self).__init__()
        normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.device = device
        position = torch.arange(max_len).unsqueeze(1)
        if normalize:
            eps = 1e-6
            position = position / (max_len - 1 + eps) * scale
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(temperature) / hidden_dim))
        self.pos = torch.zeros(max_len, hidden_dim)
        self.pos[:, 0::2] = torch.sin(position * div_term)
        self.pos[:, 1::2] = torch.cos(position * div_term)
        self.pos = self.pos.unsqueeze(1).to('cpu')

    def forward(self, x):
        return x + self.pos[:x.size(0), :].to(self.device)
        
class PositionEmbeddingSine2d(nn.Module):
    def __init__(self, spatial_dim, hidden_dim=768, temperature=10000, normalize=False, scale=None, flatten = True, device = "cuda:0"):
        super(PositionEmbeddingSine2d, self).__init__()
        self.num_pos_feats = hidden_dim // 2
        normalize = normalize
        self.h, self.w = spatial_dim
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.device = device
        position_y = torch.arange(self.h).unsqueeze(1)
        position_x = torch.arange(self.w).unsqueeze(1)
        if normalize:
            eps = 1e-6
            position_y = position_y / (self.h - 1 + eps) * scale
            position_x = position_x / (self.w - 1 + eps) * scale
        div_term = torch.exp(torch.arange(0, self.num_pos_feats, 2) * (-math.log(temperature) / self.num_pos_feats))
        pe_y = torch.zeros(self.h, 1, self.num_pos_feats)
        pe_x = torch.zeros(1, self.w, self.num_pos_feats)
        pe_y[:, 0, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 0, 1::2] = torch.cos(position_y * div_term)
        pe_x[0, :, 0::2] = torch.sin(position_x * div_term)
        pe_x[0, :, 1::2] = torch.cos(position_x * div_term)
        pe_y = pe_y.repeat(1, self.w, 1)
        pe_x = pe_x.repeat(self.h, 1, 1)
        self.pos = torch.cat((pe_y, pe_x), dim=-1).permute(2, 0, 1)
        if flatten:
            self.pos =  self.pos.view(hidden_dim, -1).permute(1,0).unsqueeze(1)
        else:
            self.pos = self.pos.permute(1,2,0)
        del pe_y, pe_x, position_y, position_x

    def forward(self, x):
        return x.to(self.device) + self.pos.to(self.device)


class FixationEmbeddingLearned2d(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, spatial_dim, hidden_dim = 768, device = "cuda:0"):
        super(FixationEmbeddingLearned2d, self).__init__()
        self.h, self.w = spatial_dim
        self.row_embed = nn.Embedding(self.h, hidden_dim//2)
        self.col_embed = nn.Embedding(self.w, hidden_dim//2)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, token):
        x_emb = self.col_embed(token[:, :, 1])
        y_emb = self.row_embed(token[:, :, 0])
        pos = torch.cat([y_emb, x_emb], dim = -1).to(self.device)
        return pos


