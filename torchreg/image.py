import torch
from torch import nn
import numpy as np


class Image_(object):
    def __init__(self, t: torch.Tensor, spacing):
        pass


class SpacedTensor(torch.Tensor):
    tensorfuncs_spacing = (torch.Tensor.to, )
    @staticmethod
    def __new__(cls, t: torch.tensor, spacing, *args, **kwargs):
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

    def grid(self, align_corners=False):
        return SamplingGrid(self, align_corners=align_corners)

    def sample(self, grid, padding_mode='zeros', mode='bilinear', align_corners=False):
        sampled =  nn.functional.grid_sample(
            self,
            torch.Tensor(grid) / (self.spacing * (torch.tensor(self.spatial_shape, device=self.device)-1)/2)[[2,1,0]],
            padding_mode=padding_mode, mode=mode, align_corners=align_corners
        )
        sampled.spacing = grid.spacing.clone().detach()
        return sampled

    def napari_metadata(self):
        spacing = self.spacing.cpu().numpy()
        return dict(scale=spacing, translate=-np.array(self.spatial_shape)*spacing/2)


class SamplingGrid(SpacedTensor):
    @staticmethod
    def __new__(cls, img, *args, translation_ones=False, align_corners=False, **kwargs):
        if img.sdimension() == 3:
            grid = torch._decomp.decompositions._make_base_grid_5d(img, *img.spatial_shape, align_corners)
        elif img.sdimension() == 2:
            grid = torch._decomp.decompositions._make_base_grid_4d(img, *img.spatial_shape, align_corners)
        else:
            raise TypeError('SamplingGrids must be built from Images')

        extended_spacing = torch.cat((
            img.spacing * (torch.tensor(img.spatial_shape, device=img.device)-1)/2,
            torch.tensor([1], device=img.device),
        ))

        return super().__new__(cls, grid * extended_spacing[[2,1,0,3]], img.spacing, *args, **kwargs) 

    def __init__(cls, img, *args, translation_ones=False, align_corners=False, **kwargs):
        super().__init__(img, img.spacing, **kwargs) 

        