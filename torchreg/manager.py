from dataclasses import dataclass
import torch
from torch import nn
from .image import Image
from .functions import gaussian_kernel

@dataclass
class Resolution:
    scales: float = 1.
    smoothings: float = 1.
    iterations: int = 200
    loss: object = nn.SmoothL1Loss()


class Manager(object):
    @torch.no_grad()
    def __init__(
        self,
        reference: Image, moving: Image,
        model, model_loss_w,
        optimizer, convergence_val: float, convergence_win: int=5,
        resolutions: tuple=(Resolution(),)
    ):
        self.optimizer = optimizer
        self.model = model
        self.convergence_val = convergence_val
        self.convergence_win = convergence_win
        self.model_loss_w = model_loss_w
        self.stages = [
            (
                res.iterations,
                *self._makeresfun(reference.sdimension(), res.smoothings, res.scales)(reference, moving),
                res.loss,
                res.scales**2 if isinstance(res.scales, float) else (sum(res.scales) / len(res.scales))**2
            )
            for res in resolutions
        ]

    def fit(self):
        res = []
        # torch.cuda.memory._record_memory_history(
        #     max_entries=100000
        # )
        for iterations, ref, mov, metric, mean_scale in self.stages:
            ref.detach()
            mov.detach()
            losses = torch.full((iterations,), torch.nan, requires_grad=False).to(ref.device)
            grid_r = ref.grid()[...,:-1]
            self.model[-1].unflattened_size = (1, *grid_r.shape[:-1])
            model_loss_w = torch.tensor(self.model_loss_w, device=ref.device)
            for step in range(iterations):
                self.optimizer.zero_grad(set_to_none=True)
                pred = mov.sample(self.model(grid_r)[0])
                loss = metric(ref, pred) + (model_loss_w * self.model.loss().to(ref.device)).sum()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    losses[step] = loss.detach()
                    #print(f'{losses[step-self.convergence_win:step].diff()}')
                    if losses[step-self.convergence_win:step].diff().mean().abs() < (self.convergence_val / mean_scale):
                        break
            #self.optimizer = type(self.optimizer)(**self.optimizer.param_groups[0])
            torch.cuda.empty_cache()
            print(f'steps: {step}, final loss: {losses[step]}')
            res.append(losses.detach().cpu().numpy())
        return res

    @staticmethod
    @torch.no_grad()
    def _makeresfun(ndim, smoothings, scales):
        if type(smoothings) is float and smoothings == 1.:
            convfn = lambda x, _: x
            smoothings=(1,)
        else: 
            convfn = getattr(nn.functional, f'conv{ndim}d')
        if type(scales) is float and scales == 1:
            samplefn = lambda x, scale_factor=None: x
        else: 
            samplefn = nn.functional.interpolate
        # TODO create filter kernel based on ndim
        return lambda *imgs: tuple(
                samplefn(
                    convfn(img, gaussian_kernel(torch.tensor(smoothings, device=img.device), dtype=img.dtype, device=img.device), padding='same'),#(tmp:=torch.ones((1,1,*smoothings),dtype=img.dtype, device=img.device))/tmp.sum()),
                    scale_factor=scales
                )
                for img in imgs
            )
        
