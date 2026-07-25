import torch

from .models import DINOv1
from .models import DINOv2
from .models import STEGO
from .models import CAUSE

from .utils import resize, resize_for_aspect_ratio

@torch.no_grad()
def inference(model: CAUSE, image: torch.Tensor, scales: list=[2.0, 1.0, 1.5, 0.5], hflip: bool=True, image_size: int=512, stride: int=2, mode: str=''):
    image = image.to(model.device)

    if mode == 'sliding':
        image, image_size = resize_for_aspect_ratio(image, image_size)

        stride = image_size // stride
        grid_fn = lambda x: max(x-image_size+stride-1, 0) // stride+1
    
    ms_feat = None
    ih, iw = image.shape[1:]
    
    for scale in scales:
        # with sliding
        if mode == 'sliding':
            images = image[None] if scale == 1.0 else resize(image[None], scale=scale)
            if hflip: images = torch.cat([images, images.flip(-1)], dim=0)

            b, _, h, w = images.shape

            feat = images.new_zeros((b, model.output_dim, h // model.patch_size, w // model.patch_size))
            cnt = images.new_zeros((b, 1, h // model.patch_size, w // model.patch_size))

            for i in range(b):
                h_grids, w_grids = map(grid_fn, [h, w])
                for h_idx in range(h_grids):
                    for w_idx in range(w_grids):
                        y1, x1 = h_idx * stride, w_idx * stride
                        x2, y2 = min(x1 + image_size, w), min(y1 + image_size, h)
                        x1, y1 = max(x2 - image_size, 0), max(y2 - image_size, 0)
                        
                        crop_feat = model(images[i:i+1, :, y1:y2, x1:x2])
                        # crop_feat = resize(crop_feat, (y2-y1, x2-x1))
                        
                        feat[i:i+1, :, y1 // model.patch_size:y2 // model.patch_size, x1 // model.patch_size:x2 // model.patch_size] += crop_feat
                        cnt[i:i+1, :, y1 // model.patch_size:y2 // model.patch_size, x1 // model.patch_size:x2 // model.patch_size] += 1

            if hflip:
                feat[1] = feat[1].flip(-1)
                cnt[1] = cnt[1].flip(-1)

            feat = torch.mean(feat / cnt, dim=0)
        
        # without sliding
        else:
            sh, sw = int(ih*scale), int(iw*scale)
            while sw % model.patch_size != 0: sw += 1
            while sh % model.patch_size != 0: sh += 1

            images = resize(image[None], (sh, sw))
            if hflip: images = torch.cat([images, images.flip(-1)], dim=0)

            feat = model(images)

            if hflip: 
                feat[1] = feat[1].flip(-1)
            
            feat = feat.mean(dim=0)

        # accumulate
        if ms_feat is None: ms_feat = feat
        else: ms_feat += resize(feat, ms_feat.shape[1:])
    
    return ms_feat / len(scales)
