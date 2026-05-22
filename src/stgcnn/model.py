import torch
import torch.nn as nn

class ST_GCNN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.tcn = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, 1), 
            padding=(kernel_size // 2, 0)
        )
        self.prelu = nn.PReLU()

    def forward(self, x, A):
        # x: [Batch, Channels, Time, Nodes]
        # A: [Batch, Time, Nodes, Nodes]
        
        # 1. Spatial Graph Convolution
        spatial_out = torch.einsum('btvw, bctw -> bctv', A, x)
        
        # 2. Temporal Convolution
        temporal_out = self.prelu(self.tcn(spatial_out))
        
        return temporal_out


class TXP_CNN(nn.Module):
    def __init__(self, in_channels, obs_len=8, pred_len=12):
        super().__init__()

        self.conv = nn.Conv2d(obs_len, pred_len, kernel_size=(3, 1), padding=(1, 0))
        
    def forward(self, x):
        # x: [Batch, Channels, Time, Nodes]
        x = x.permute(0, 2, 1, 3) # -> [Batch, Time, Channels, Nodes]
        x = self.conv(x)          # -> [Batch, Pred_Time, Channels, Nodes]
        x = x.permute(0, 2, 1, 3) # -> [Batch, Channels, Pred_Time, Nodes]
        return x


class Social_STGCNN(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=64, obs_len=8, pred_len=12):
        super().__init__()
        
        self.st_gcnn1 = ST_GCNN_Layer(in_channels, hidden_dim)
        self.st_gcnn2 = ST_GCNN_Layer(hidden_dim, hidden_dim)
        self.st_gcnn3 = ST_GCNN_Layer(hidden_dim, hidden_dim)
        
        self.txp = TXP_CNN(hidden_dim, obs_len, pred_len)
        
        # Final projection (5 params)
        self.predict = nn.Conv2d(hidden_dim, 5, kernel_size=1)

    def forward(self, x, A):
        x = self.st_gcnn1(x, A)
        x = self.st_gcnn2(x, A)
        x = self.st_gcnn3(x, A)
        
        # Temporal extrapolation (8 -> 12)
        x = self.txp(x) 
        out = self.predict(x) # [Batch, 5_params, 12_frames, 11_nodes]
        
        # Split and apply constraints
        mu_x = out[:, 0, :, :]
        mu_y = out[:, 1, :, :]
        
        # Clip exponential so variance doesn't explode
        log_sig_x = torch.clamp(out[:, 2, :, :], min=-4.6, max=6.9) # exp(-4.6) ~= 0.01, exp(6.9) ~= 1000
        log_sig_y = torch.clamp(out[:, 3, :, :], min=-4.6, max=6.9)
        sig_x = torch.exp(log_sig_x)
        sig_y = torch.exp(log_sig_y)
        
        # Clip correlation to avoid -1/1
        rho = torch.clamp(torch.tanh(out[:, 4, :, :]), min=-0.99, max=0.99)
        
        return mu_x, mu_y, sig_x, sig_y, rho