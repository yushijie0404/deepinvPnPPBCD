from deepinv.optim.optim_iterators import OptimIterator, fStep, gStep
import torch
import tensorly as tl

class PBCDIteration(OptimIterator):
    r"""Single iteration of Plug-and-Play Block Coordinate Descent (PBCD)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eigenmode = kwargs.get('eigenmode', 1)
        self.l2_axis = kwargs.get('l2_axis', 1)
        self.H_step = HStepPBCD(**kwargs)
        self.S_step = l2pStepPBCD(**kwargs)
        self.E_step = ProjStepPBCD(**kwargs)
        self.Z_step = gStepPBCD(**kwargs)

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics=None):
        S_prev, E_prev, Z_prev = X["est"]
        
        # Compute intermediate parameters
        cur_params["tilde_alpha_S"] = cur_params["delta"] / (cur_params["delta"] + cur_params["alpha_S"])
        cur_params["tilde_tau"] = cur_params["tau"] / (cur_params["delta"] + cur_params["alpha_S"])
        cur_params["tilde_alpha_E"] = 1 / cur_params["alpha_E"]
        cur_params["tilde_alpha_Z"] = cur_params["delta"] / (cur_params["delta"] + cur_params["alpha_Z"])
        cur_params["lambda"] = 1 / cur_params["tilde_alpha_Z"]
        
        # Update S
        S_temp = y - tl.tenalg.mode_dot(Z_prev, E_prev, self.eigenmode, transpose=False)
        S = self.S_step(
            self.H_step(S_prev, cur_data_fidelity, cur_params["tilde_alpha_S"], S_temp, physics),
            cur_prior[0],
            cur_params
        )
        
        # Update E
        E_temp = tl.unfold(y - S, self.eigenmode) @ tl.unfold(Z_prev, self.eigenmode).permute(1, 0)
        E = self.E_step(
            self.H_step(E_prev, cur_data_fidelity, cur_params["tilde_alpha_E"], E_temp, physics),
            cur_prior[1]
        )
        
        # Update Z
        Z_temp = tl.tenalg.mode_dot(y - S, E, self.eigenmode, transpose=True)
        Z_shift = self.H_step(Z_prev, cur_data_fidelity, cur_params["tilde_alpha_Z"], Z_temp, physics)
        
        # Normalize and estimate noise
        Z_shift_max, Z_shift_min = Z_shift.max(), Z_shift.min()
        Z_shift = (Z_shift - Z_shift_min) / (Z_shift_max - Z_shift_min)
        cur_params["g_param"] = cur_params["noiselevelparams"] * noise_estimate_batch_new(
            Z_shift.permute(1, 0, 2, 3), pch_size=8
        ).to(self.device)
        
        Z = self.Z_step(Z_shift, cur_prior[2], cur_params)
        Z = Z * (Z_shift_max - Z_shift_min) + Z_shift_min
        
        return {"est": (S, E, Z)}

class HStepPBCD(fStep):
    r"""PBCD H-step module."""
    def forward(self, x, cur_data_fidelity, alpha, y, physics=None):
        if physics is None:
            grad = alpha * cur_data_fidelity.grad_d(x, y)
        else:
            grad = alpha * cur_data_fidelity.grad(x, y, physics)
        return x - grad

class ProjStepPBCD(gStep):
    r"""PBCD projection step module."""
    def forward(self, x, cur_prior):
        return cur_prior.prox(x)

class gStepPBCD(gStep):
    r"""PBCD g-step module."""
    def forward(self, x, cur_prior, cur_params):
        return cur_prior.prox(
            x.permute(1, 0, 2, 3),
            cur_params["g_param"],
            gamma=cur_params["tilde_alpha_Z"]
        ).permute(1, 0, 2, 3)

class l2pStepPBCD(gStep):
    r"""PBCD L2P step module."""
    def forward(self, x, cur_prior, cur_params):
        return cur_prior.prox(x, tilde_tau=cur_params["tilde_tau"])