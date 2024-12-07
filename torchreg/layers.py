import torch
from torch import nn
from .image import Image, SamplingGrid

class Sequential(nn.Sequential):
    def loss(self):
        return torch.tensor([l.loss() for l in self if hasattr(l, 'loss')])

class Affine_(nn.Module):
    @torch.no_grad()
    def __init__(self, ndim: int, pad_ones=True):
        super().__init__()
        transform = torch.zeros((1, ndim, ndim+1), dtype=torch.double)
        transform[0, :ndim, :ndim].fill_diagonal_(1.)
        self.transform = nn.Parameter(transform)
        self.pad_ones = pad_ones

    def forward(self, grid: SamplingGrid) -> Image:
        # n, _, d, h, w = self.output_size
        # base_grid shape is (d, h, w, 4) and theta shape is (n, 3, 4)
        # We do manually a matrix multiplication which is faster than mm()
        # (d * h * w, 4, 1) * (n, 1, 4, 3) -> (n, d * h * w, 3)
        # in einsum:
        # torch.einsum('dDo,noDt->ndt', grid.view(-1,4,1), self.transform.mT.unsqueeze(1))
        grid_affined = (
                (grid.view(-1, self.transform.shape[-1], 1) * self.transform.mT.unsqueeze(1)).sum(-2)
            ).view(1,*grid.shape[-4:-1], -1)
        if self.pad_ones:
            grid_affined = nn.functional.pad(grid_affined, (0,1), value=1) 
        return grid_affined
    
    def loss(self):
        return torch.det(self.transform[:, :self.transform.shape[1], :self.transform.shape[1]]).abs()


class DeformGrid(nn.Module):
    @torch.no_grad()
    def __init__(self, ndim: int, node_spacing: float, pad_ones=True):
        super().__init__()
        self.pad_ones = pad_ones
        self.mode = 'trilinear' if ndim == 3 else 'bilinear'
        self.displacements = nn.Parameter(torch.zeros(1, ndim, *node_spacing)  )
        #smooth = torch.ones((3,1,3,19,19))
        #self.smooth = smooth/smooth.sum()

    def forward(self, grid:SamplingGrid) -> Image:
        grid_deformed = grid[...,:self.displacements.shape[1]] + \
            nn.functional.interpolate(self.displacements, size=grid.shape[-4:-1], mode=self.mode).movedim(1,-1)
            # nn.functional.conv3d(
            #     nn.functional.interpolate(self.displacements, size=self.target_grid.shape[:-1], mode=self.mode),
            #     self.smooth, groups=3, padding='same'
            # ).movedim(1,-1)
        if self.pad_ones:
            grid_deformed = nn.functional.pad(grid_deformed, (0,1), value=1) 
        return grid_deformed

    # def to(self, device):
    #     obj = super().to(device)
    #     obj.smooth = self.smooth.to(device)
    #     return obj

    def loss(self):
        loss =  torch.gradient(self.displacements[0,0], dim=-1)[0] + \
                torch.gradient(self.displacements[0,1], dim=-2)[0] + \
                torch.gradient(self.displacements[0,2], dim=-3)[0]
        return loss.abs().mean()
