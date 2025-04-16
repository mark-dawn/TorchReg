import torch
from torch import nn
from .image import SamplingGrid
from .functions import gaussian_kernel

class Sequential(nn.Sequential):
    def loss(self):
        return torch.cat([l.loss().view(1) for l in self if hasattr(l, 'loss')])

class Translation(nn.Module):
    bias: torch.Tensor
    def __init__(self, ndim, device=None, dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(ndim, **factory_kwargs))

    def forward(self, input1):
        return input1 + self.bias

    def loss(self):
        return torch.linalg.norm(self.bias)

class Affine(nn.Linear):
    @torch.no_grad()
    def __init__(self, ndim, device=None, dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(ndim, ndim, **factory_kwargs)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.bias.fill_(0.)
        self.weight.fill_(0.)
        self.weight.fill_diagonal_(1.)

    def loss(self):
        #return torch.det(self.weight).abs()
        return (1-self.weight.trace()/3).abs()

class DeformGrid(nn.Module):
    __constants__ = ['ndim', 'spacing', 'kernel', '_input_rescaler']
    ndim: int
    bias: torch.Tensor
    spacing: torch.Tensor
    _input_rescaler: torch.Tensor
    kernel: int # torch.Tensor

    def __init__(self, size, spacing, device=None, dtype=None, kernel=1) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.ndim = len(size)
        self.bias = nn.Parameter(torch.zeros((1, self.ndim, *size,),**factory_kwargs))
        self.register_buffer('spacing', spacing.detach().clone())
        self.register_buffer('_input_rescaler', (spacing * (torch.tensor(size, **factory_kwargs))/2).detach()[[2,1,0]].clone())
        #self.register_buffer('kernel', gaussian_kernel((kernel/spacing[1:]).round().int().detach(), variance=kernel/3, channels=3, groups=3, **factory_kwargs).detach())
        self.kernel = kernel

    def forward(self, input1):
        # rescale the input to a 5-D, float, between -1,1 image for grid sample
        input_rescaled = (input1.view(*(1,) * (2 + self.ndim - input1.dim()), *input1.shape) / self._input_rescaler)
        deltas = nn.functional.grid_sample(self.bias, input_rescaled, padding_mode='zeros', align_corners=False, mode='nearest')
        kernels = (4 * self.spacing / input1.spacing).round().int() -1
        if torch.any(kernels>8):
            for dim in range(1, self.ndim)[::-1]:
                kernel = gaussian_kernel(
                    kernels[dim].view(1),
                    variance=kernels[dim], channels=3, groups=3,
                    device=input1.device, dtype=input1.dtype
                ).detach()
             #   print(kernel.shape)
                deltas = nn.functional.conv3d(deltas, kernel.view(*kernel.shape[:2], *torch.where(torch.arange(self.ndim)==dim,-1,1)), padding='same', groups=3)
        else:
            kernel = gaussian_kernel(
                    kernels,
                    variance=kernels.max(), channels=3, groups=3,
                    device=input1.device, dtype=input1.dtype
                ).detach()
            #print(kernel.shape)
            deltas = nn.functional.conv3d(deltas, kernel, padding='same', groups=3)
        return deltas.moveaxis(1,-1) + input1

    def loss(self):
        # loss =  torch.gradient(self.bias[0,0], dim=-1, spacing=self.spacing[0])[0] + \
        #         torch.gradient(self.bias[0,1], dim=-2, spacing=self.spacing[1])[0] + \
        #         torch.gradient(self.bias[0,2], dim=-3, spacing=self.spacing[2])[0]
        # return (loss / 3).abs().mean()
        return torch.zeros(1, device=self.bias.device, dtype=self.bias.dtype)
        # return self.bias.abs().mean()

class ProbeAffine(nn.Module):
    __constants__ = ["ndim"]
    ndim: int
    weight: torch.Tensor
    center: torch.Tensor
    r2: torch.Tensor
    init_r: float
    init_c: tuple

    def __init__(self, ndim, device=None, dtype=None, init_r=1000., init_c=(0.,0.,0.)) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.ndim = ndim
        self.weight = nn.Parameter(torch.empty((ndim, 1, ndim), **factory_kwargs))
        self.center = nn.Parameter(torch.empty(ndim, **factory_kwargs))
        self.r2 = nn.Parameter(torch.empty(tuple(), **factory_kwargs))
        self.init_c = init_c
        self.init_r = init_r
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.r2.fill_(self.init_r)
        self.center.fill_(0.)
        self.center.add_(torch.tensor(self.init_c, device=self.center.device, dtype=self.center.dtype))
        self.weight.fill_(0.)
        #self.weight[:,0,:].fill_diagonal_(.01)

    def forward(self, input1) -> torch.Tensor:
        input_b = input1 - self.center
        falloff = nn.functional.softplus(self.r2 - ((input_b) ** 2).sum(dim=-1))[:, None]
        return nn.functional.bilinear(nn.functional.normalize(falloff, p=2), input_b, self.weight) + input1

    def loss(self):
        return torch.det(self.weight[:,0,:]).abs()
