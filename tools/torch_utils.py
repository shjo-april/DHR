import copy
import torch
import random
import numpy as np

from torch import nn
from torch.nn import functional as F

normalize = F.normalize
cosine = F.cosine_similarity
einsum = torch.einsum

def get_minmax(masks):
    min_v = torch.min(masks.view(masks.shape[0], -1), dim=1)[0]
    max_v = torch.max(masks.view(masks.shape[0], -1), dim=1)[0]
    return min_v[:, None, None], max_v[:, None, None]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_numpy(tensor):
    return tensor.cpu().detach().numpy()

def calculate_parameters(params):
    return sum(param.numel() for param in params)/1000000.0

def get_learning_rate(optimizer, option='first'):
    lr_list = [group['lr'] for group in optimizer.param_groups]
    if option == 'max': return max(lr_list)
    elif option == 'min': return min(lr_list)
    elif option == 'first': return lr_list[0]
    else: return lr_list

def normalize_masks(masks):
    flat_v = masks.reshape(masks.shape[0], -1)
    max_v, min_v = flat_v.max(dim=1)[0][:, None, None], flat_v.min(dim=1)[0][:, None, None]
    return (masks - min_v) / (max_v - min_v).clip(min=1e-5)

def resize(tensors, size=None, scale=1.0, mode='bilinear', align_corners=True):
    # for mask
    is_mask = len(tensors.size()) == 2
    if is_mask: tensors = tensors.float().unsqueeze(0)

    # for batch
    without_batch = len(tensors.size()) == 3
    if without_batch: tensors = tensors.unsqueeze(0)

    # resize
    if size is None: size = tensors.size()[2:]
    if scale != 1.: size = [int(s * scale) for s in size]
    if mode == 'nearest': align_corners = None
    
    _, _, h, w = tensors.size()
    if size[0] != h or size[1] != w:
        tensors = F.interpolate(tensors, size, mode=mode, align_corners=align_corners)
    
    # for batch
    if without_batch: tensors = tensors[0]

    # for mask
    if is_mask: tensors = tensors.long()[0]

    return tensors

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

def load_model(model, model_path, strict=True, map_location='cpu'):
    if is_parallel(model):
        model = de_parallel(model)
    model.load_state_dict(torch.load(model_path, map_location=map_location), strict=strict)

def save_model(model, model_path):
    if is_parallel(model):
        model = de_parallel(model)
    torch.save(model.state_dict(), model_path)

class ExponentialMovingAverage:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(de_parallel(model))
        self.ema.eval()
        
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.decay = decay
    
    @torch.no_grad()
    def update(self, model):
        current = model.state_dict() if not is_parallel(model) else model.module.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= self.decay
                v += (1. - self.decay) * current[k].detach()
    
    def get(self): return self.ema
    def to(self, device): self.ema.to(device)
