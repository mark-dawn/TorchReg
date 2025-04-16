import torch
import torch.nn as nn

def match_hist(dst, ref, bins=255):
    # B C
    B, C = dst.size()[:2]
    # assertion
    assert dst.device == ref.device
    # [B*C 256]
    hist_dst = cal_hist(dst)
    hist_ref = cal_hist(ref)
    # [B*C 256]
    tables = cal_trans_batch(hist_dst, hist_ref, max=255)
    # [B C ...]
    rst = torch.empty_like(dst)
    dst = (dst * bins).long()
    for b in range(B):
        for c in range(C):
            rst[b,c] = tables[b*c, (dst[b,c])]
    return rst# / bins


def cal_hist(img, bins=256):
    B, C = img.size()[:2]
    # [B*C bins]
    if torch.is_grad_enabled():
        hists = soft_histc_batch(img, bins=bins, min=0., max=1., sigma=3*25)
    else:
        hists = torch.stack([torch.histc(img[b,c], bins=bins, min=0., max=1.) for b in range(B) for c in range(C)])
    hists = nn.functional.normalize(hists.float(), p=1)
    # BC bins
    #bc, n = hists.size()
    # [B*C bins bins]
    #triu = torch.ones(bc, n, n, device=hists.device).triu()
    # [B*C bins]
    #hists = torch.bmm(hists[:,None,:], triu)[:,0,:]
    return hists

def soft_histc_batch(x, bins=256, min=0, max=256, sigma=3*25):
    # B C
    B, C = x.size()[:2]
    # [B*C ...]
    x = x.view(B*C, -1)
    delta = float(max - min) / float(bins)
    # [1 bins 1]
    centers = float(min) + delta * (torch.arange(bins, device=x.device, dtype=torch.bfloat16) + 0.5)[None,:,None]
    # [B*C 1 ...]
    x = torch.unsqueeze(x, 1)
    # [B*C bins ...]
    x = (x - centers).type(torch.bfloat16)
    # [B*C bins ...]
    x = torch.sigmoid(sigma * (x + delta/2)) - torch.sigmoid(sigma * (x - delta/2))
    # [B*C bins]
    x = x.sum(dim=2)
    # prevent oom
    # torch.cuda.empty_cache()
    return x.type(torch.float32)

def cal_trans_batch(hist_dst, hist_ref, bins=256, min=0, max=1.0):
    # [B*C bins bins]
    hist_dst = hist_dst[:,None,:].repeat(1,bins,1)
    # [B*C bins bins]
    hist_ref = hist_ref[:,:,None].repeat(1,1,bins)
    # [B*C bins bins]
    table = hist_dst - hist_ref
    # [B*C bins bins]
    table = torch.where(table>=0, 0., 1.)
    # [B*C bins]
    #table = torch.sum(table, dim=1) - 1
    # [B*C bins]
    #table = torch.clamp(table, min=min, max=max)
    return table.mean(dim=1)

def make_kernel(kernel_size, fn, channels=1, groups=1, device=None, dtype=None, ):
    dims = len(kernel_size)
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    cord = [torch.arange(ks, device=device, dtype=dtype) for ks in kernel_size]
    grid = torch.stack(torch.meshgrid(*cord), dim=-1, ) - (kernel_size - 1)/2.

    kernel = fn(grid)
    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to batch, ndim convolutional weight
    return kernel.view(1, 1, *kernel_size).repeat(channels, channels//groups, *[1]*dims)

def gaussian_kernel(kernel_size, mean=0, variance=1, channels=1, groups=1, device=None, dtype=None):
    def gauss(k):
        var=variance
        m=mean
        a = 1./2.*torch.pi*var
        return a * torch.exp(-torch.sum((k-m)**2, dim=-1) / (2.*var))
    return make_kernel(kernel_size, gauss, channels=channels, groups=groups, device=device, dtype=dtype)
