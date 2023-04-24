import torch
import torch.nn.functional as F
from torch import nn, Tensor

from positional_encodings import PositionEmbeddingSine2d

class gazeformer(nn.Module):
    def __init__(self, transformer, spatial_dim, dropout=0.4, max_len = 7, patch_size  = 16, device = "cuda:0"):
        super(gazeformer, self).__init__()
        self.spatial_dim = spatial_dim
        self.transformer = transformer.to(device)
        self.hidden_dim = transformer.d_model
        #fixation embeddings
        self.querypos_embed = nn.Embedding(max_len,self.hidden_dim).to(device)
        #2D patch positional encoding
        self.patchpos_embed = PositionEmbeddingSine2d(spatial_dim, hidden_dim=self.hidden_dim, normalize=True, device = device)
        #2D pixel positional encoding for initial fixation
        self.queryfix_embed = PositionEmbeddingSine2d((spatial_dim[0] * patch_size, spatial_dim[1] * patch_size), hidden_dim=self.hidden_dim, normalize=True, flatten = False, device = device).pos.to(device)
        #classify fixation, or PAD tokens
        self.token_predictor = nn.Linear(self.hidden_dim, 2).to(device)
        #Gaussian parameters for x,y,t
        self.generator_y_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_x_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_t_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_y_logvar = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_x_logvar = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_t_logvar = nn.Linear(self.hidden_dim, 1).to(device)
        
        self.device = device
        self.max_len = max_len
        
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.LogSoftmax(dim=-1).to(device)
        #projection for first fixation encoding
        self.firstfix_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    #reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, src: Tensor, tgt: Tensor, task: Tensor):
        src = src.to(self.device)
        tgt_input = torch.zeros(self.max_len, src.size(0), self.hidden_dim).to(self.device)#Notice that this where we convert target input to zeros

        tgt_input[0, :, :] = self.firstfix_linear(self.queryfix_embed[tgt[:, 0], tgt[:,1], :])
        outs = self.transformer(src=src, tgt=tgt_input, tgt_mask= None, tgt_key_padding_mask = None, 
        task = task.to(self.device), querypos_embed = self.querypos_embed.weight.unsqueeze(1), patchpos_embed = self.patchpos_embed)

        outs = self.dropout(outs)
        #get Gaussian parameters for (x,y,t)
        y_mu, y_logvar, x_mu, x_logvar, t_mu, t_logvar = self.generator_y_mu(outs),self.generator_y_logvar(outs), self.generator_x_mu(outs), self.generator_x_logvar(outs), self.generator_t_mu(outs), self.generator_t_logvar(outs)

        return self.softmax(self.token_predictor(outs)), self.activation(self.reparameterize(y_mu, y_logvar)),self.activation(self.reparameterize(x_mu, x_logvar)), self.activation(self.reparameterize(t_mu, t_logvar))
        
