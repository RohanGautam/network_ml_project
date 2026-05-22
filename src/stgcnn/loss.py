import torch

def bivariate_loss(pred, target):
    """
    Calculates the Negative Log-Likelihood for a bivariate Gaussian distribution.
    """
    mu_x, mu_y, sig_x, sig_y, rho = pred
    target_x = target[..., 0]
    target_y = target[..., 1]
    
    norm_x = target_x - mu_x
    norm_y = target_y - mu_y
    
    eps = 1e-5
    sig_x = sig_x + eps
    sig_y = sig_y + eps
    
    cor_sq = rho ** 2
    cor_sq = torch.clamp(cor_sq, min=0.0, max=1.0 - eps)
    
    z = (norm_x / sig_x)**2 + (norm_y / sig_y)**2 - 2 * rho * norm_x * norm_y / (sig_x * sig_y)
    
    loss = z / (2 * (1 - cor_sq)) + torch.log(sig_x) + torch.log(sig_y) + 0.5 * torch.log(1 - cor_sq)
    
    return torch.mean(loss)