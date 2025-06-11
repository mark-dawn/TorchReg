import torch
from torch import nn
import numpy as np


class Image_(object):
    def __init__(self, t: torch.Tensor, spacing):
        pass


class SpacedTensor(torch.Tensor):
    tensorfuncs_spacing = (torch.Tensor.to, )
    @staticmethod
    def __new__(cls, t: torch.Tensor, spacing, *args, **kwargs):
        return super().__new__(cls, t, *args, **kwargs)

    def __init__(self, img, spacing, **kwargs):
        self.spacing = spacing.clone().detach()

    def to(self, *args, **kwargs):
        new_obj = super(torch.Tensor, self).to(*args, **kwargs)
        if new_obj is self:
            return self
        new_obj.spacing = self.spacing.to(*args, **kwargs)
        return new_obj

    def __repr__(self, *args, tensor_contents=None):
        representation = super().__repr__(*args, tensor_contents=tensor_contents)
        return representation[:-1] + f", spacing={str(self.spacing)})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        newobj = super().__torch_function__(func, types, args, kwargs)
        if type(newobj) is cls and func not in cls.tensorfuncs_spacing:
            if func is nn.functional.interpolate:
                newobj.spacing = (args[0].spacing / torch.tensor(kwargs['scale_factor'], device=args[0].device)).to(newobj.device)
            else:
                for arg in args:
                    if type(arg) is cls:
                        newobj.spacing = arg.spacing.clone().to(device=newobj.device)
        return newobj


class Image(SpacedTensor):
    tensorfuncs_spacing = (torch.Tensor.to, )
    @staticmethod
    def __new__(cls, img, spacing, *args, **kwargs):
        if 2 <= img.ndimension() < 4:
            return super().__new__(cls, img[None, None, ...], spacing, *args, **kwargs)
        else: raise TypeError('Images must be 2 or 3 dimensional')

    def sdimension(self):
        return self.ndimension() - 2

    @property
    def spatial_shape(self,):
        return self.shape[-self.sdimension():]

    def sample(self, grid, spacing, padding_mode='zeros', mode='bilinear', align_corners=False):
        sampled =  nn.functional.grid_sample(
            self,
            torch.Tensor(grid[None]) / (self.spacing * (torch.tensor(self.spatial_shape, device=self.device)-1)/2)[[2,1,0]],
            padding_mode=padding_mode, mode=mode, align_corners=align_corners
        )
        sampled.spacing = spacing.detach().to(sampled.device)
        return sampled

    def napari_metadata(self):
        spacing = self.spacing.cpu().numpy()
        return dict(scale=spacing, translate=-np.array(self.spatial_shape)*spacing/2)

    def grid(self, align_corners=False):
        if self.sdimension() == 3:
            grid = torch._decomp.decompositions._make_base_grid_5d(self, *self.spatial_shape, align_corners)
        elif self.sdimension() == 2:
            grid = torch._decomp.decompositions._make_base_grid_4d(self, *self.spatial_shape, align_corners)
        else:
            raise TypeError('SamplingGrids must be built from Images')

        extended_spacing = torch.cat((
            self.spacing * (torch.tensor(self.spatial_shape, device=self.device)-1)/2,
            torch.tensor([1], device=self.device),
        ))

        return grid * extended_spacing[[2,1,0,3]]

