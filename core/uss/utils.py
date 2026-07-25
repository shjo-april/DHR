import torch
from torch.nn import functional as F

def get_feature_size(size, patch_size):
    return list(map(lambda x: x // patch_size, size))

def interpolate_pe(pe, size):
    return F.interpolate(pe.permute(1, 0)[None], size, mode='linear')[0].permute(1, 0)

def resize(tensors, size=None, scale=1.0, mode='bilinear', align_corners=False):
    without_batch = len(tensors.size()) == 3

    if without_batch: tensors = tensors.unsqueeze(0)
    if size is None: size = tensors.size()[2:]
    
    size = list(size)
    size[0] = int(size[0] * scale)
    size[1] = int(size[1] * scale)

    if mode == 'nearest': align_corners = None
    
    _, _, h, w = tensors.size()
    if size[0] != h or size[1] != w:
        tensors = F.interpolate(tensors, size, mode=mode, align_corners=align_corners)
    
    if without_batch: tensors = tensors[0]
    return tensors

def resize_for_aspect_ratio(image: torch.Tensor, image_size: int=0):
    oh, ow = image.shape[1:]

    if image_size > 0: # resize
        short, long = (ow, oh) if ow <= oh else (oh, ow)
        new_short, new_long = image_size, int(image_size * long / short)
        nw, nh = (new_short, new_long) if ow <= oh else (new_long, new_short)
        image = resize(image, (nh, nw))
    else: # use original image
        image_size = min(oh, ow)
    
    return image, image_size