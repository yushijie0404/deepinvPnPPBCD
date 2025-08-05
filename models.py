import torch
import tensorly as tl
from deepinv.optim.prior import Prior, RED

def vectorized_prox_lp(S_matrix, normb, mu, p):
    v = mu * normb ** (p - 2)
    tn = solve_lp_vectorized(v, p)
    return tn * S_matrix

def solve_lp_vectorized(v, p, max_step=10, theta=1e-10):
    beta = (v * p * (1 - p)) ** (1 / (2 - p))
    t1n = (1 + beta) / 2
    converged = torch.zeros_like(v, dtype=torch.bool)
    
    for _ in range(max_step):
        tn = t1n - (v * p * t1n ** (p - 1) + t1n - 1) / (v * p * (p - 1) * t1n ** (p - 2) + 1)
        current_converged = (torch.sum((tn - t1n) ** 2, dim=-1) / torch.sum(t1n ** 2, dim=-1)) < theta
        converged = converged | current_converged
        
        if converged.all():
            break
            
        t1n = torch.where(converged, t1n, tn)
        
    tn = torch.max(beta, torch.min(tn, torch.ones_like(tn)))
    return tn

class L2pPrior(Prior):
    def __init__(self, *args, l2_axis=1, l2p_p=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.l2_axis = l2_axis
        self.l2p_p = l2p_p

    def prox(self, x, *args, temp_tau=1.0, **kwargs):
        S_matrix = tl.unfold(x, self.l2_axis)
        normb = torch.norm(S_matrix, dim=0, keepdim=True)
        p0 = 1 / (2 - self.l2p_p)
        b0 = temp_tau ** p0 * (2 - self.l2p_p) / ((2 * (1 - self.l2p_p)) ** (1 - p0))
        
        mask = normb <= b0
        S_matrix_output = torch.zeros_like(S_matrix)
        S_matrix_output[:, mask[0]] = 0
        
        if not mask.all():
            S_matrix_output[:, ~mask[0]] = vectorized_prox_lp(
                S_matrix[:, ~mask[0]], normb[:, ~mask[0]], temp_tau, self.l2p_p
            )
            
        return tl.fold(S_matrix_output, self.l2_axis, x.shape)

class StiefelPrior(Prior):
    def __init__(self, *args, rank, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.rank = rank

    def prox(self, x, *args, **kwargs):
        U, _, V = torch.linalg.svd(x, full_matrices=False)
        SIGone = torch.diag(torch.ones(self.rank, dtype=torch.float32).to(x.device))
        return U @ SIGone @ V

class GSPnP(RED):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

def noise_estimate_batch_new(im_batch, pch_size=8):
    B, C, H, W = im_batch.shape
    pch = im2patch_new(im_batch, pch_size, 3)
    B, C, _, _, num_pch = pch.shape
    pch = pch.view(B, -1, num_pch)
    
    d = pch.shape[1]
    mu = pch.mean(dim=2, keepdim=True)
    X = pch - mu
    sigma_X = torch.bmm(X, X.transpose(1, 2)) / num_pch
    sig_values, _ = torch.linalg.eigh(sigma_X)
    sig_values = sig_values.sort(dim=1).values
    
    noise_levels = []
    for sig_value in sig_values:
        for ii in range(-1, -len(sig_value)-1, -1):
            temp_tau = sig_value[:ii].mean()
            above = (sig_value[:ii] > temp_tau).sum()
            below = (sig_value[:ii] < temp_tau).sum()
            if above == below:
                noise_levels.append(torch.sqrt(temp_tau))
                break
        else:
            noise_levels.append(torch.sqrt(sig_value.mean()))
            
    return torch.stack(noise_levels).to(im_batch.device).view(B, 1)

def im2patch_new(im, pch_size, stride=1):
    if isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        pch_H, pch_W = pch_size
        
    if isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        stride_H, stride_W = stride
        
    B, C, H, W = im.shape
    patches_unfold = torch.nn.functional.unfold(im, kernel_size=pch_size, stride=stride)
    num_patches = patches_unfold.shape[-1]
    patches = patches_unfold.view(B, C, pch_H, pch_W, num_patches)
    
    return patches