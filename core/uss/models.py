import torch
from torch import nn

from .dinov1 import vit_base
from .layers import Segment_TR
from .utils import interpolate_pe, get_feature_size

class DINOv1(nn.Module):
    def __init__(self, arch='ViT-B', patch_size=8, device=torch.device('cpu')):
        super().__init__()
        self.backbone = vit_base(patch_size, num_classes=0)
        
        state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dino/" + f"dino_vitbase{patch_size}_pretrain/dino_vitbase{patch_size}_pretrain.pth")
        self.backbone.load_state_dict(state_dict, strict=True)

        self.device = device
        self.patch_size = patch_size
        self.embed_dim = self.output_dim = self.backbone.embed_dim
        
        self.eval()
        self.to(device)

    def to(self, device):
        super().to(device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, image):
        if len(image.shape) == 3: image = image[None]

        fh, fw = get_feature_size(image.shape[2:], self.patch_size)
        image_feat, _, _ = self.backbone.get_intermediate_feat(image)
        
        image_feat = image_feat[0][:, 1:, :]
        image_feat = image_feat.reshape(image_feat.shape[0], fh, fw, -1).permute(0, 3, 1, 2)
        
        return image_feat

class STEGO(nn.Module):
    def __init__(self, arch, dataset='VOC2012', patch_size=8, dim=70, device=torch.device('cpu'), pt_dir='./weights/'):
        super().__init__()
        self.model = DINOv1(arch, patch_size, device)

        self.device = device
        self.output_dim = dim
        self.patch_size = patch_size
        
        self.cluster1 = self.make_linear_cluster(self.model.embed_dim, dim)
        self.cluster2 = self.make_nonlinear_cluster(self.model.embed_dim, dim)

        checkpoint = torch.load(pt_dir + f'STEGO_{dataset}.pt', map_location='cpu')
        self.cluster1.load_state_dict(checkpoint['cluster1'])
        self.cluster2.load_state_dict(checkpoint['cluster2'])

        self.eval()
        self.to(device)

    def to(self, device):
        super().to(device)
        self.device = device

    def make_linear_cluster(self, inc, ouc, k=(1, 1)):
        return nn.Sequential(nn.Conv2d(inc, ouc, k))

    def make_nonlinear_cluster(self, inc, ouc, k=(1, 1)):
        return nn.Sequential(nn.Conv2d(inc, inc, k), nn.ReLU(), nn.Conv2d(inc, ouc, k))

    @torch.no_grad()
    def forward(self, images):
        image_feat = self.model(images)
        image_feat = self.cluster1(image_feat) + self.cluster2(image_feat)
        return image_feat

class DINOv2(nn.Module):
    def __init__(self, arch='ViT-S', patch_size=14, device=torch.device('cpu')):
        super().__init__()

        if arch == 'ViT-S': tag = f'dinov2_vits{patch_size}'
        elif arch == 'ViT-B': tag = f'dinov2_vitb{patch_size}'
        elif arch == 'ViT-L': tag = f'dinov2_vitl{patch_size}'
        elif arch == 'ViT-G': tag = f'dinov2_vitg{patch_size}'
        else: raise ValueError("Unknown arch")
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', tag) 
        
        self.device = device
        self.patch_size = patch_size
        self.embed_dim = self.output_dim = self.backbone.embed_dim

        self.eval()
        self.to(device)

    def to(self, device):
        super().to(device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, image):
        if len(image.shape) == 3: image = image[None]
        return self.backbone.get_intermediate_layers(image, n=1, reshape=True)[0]
    
    @torch.no_grad()
    def forward_features(self, x, masks=None):
        if len(x.shape) == 3: x = x[None]

        if isinstance(x, list):
            return self.backbone.forward_features_list(x, masks)

        x = self.backbone.prepare_tokens_with_masks(x, masks)
        for blk in self.backbone.blocks:
            x = blk(x)

        return self.backbone.norm(x)

class CAUSE(nn.Module):
    def __init__(self, arch='ViT-B', dataset='VOC2012', patch_size=14, dim=90, device=torch.device('cpu'), pt_dir='./weights/'):
        super().__init__()

        self.backbone = DINOv2(arch, patch_size, device)

        self.device = device
        self.output_dim = dim
        self.patch_size = patch_size

        self.segment = Segment_TR(self.backbone.embed_dim, dim, 322**2 // self.patch_size**2)

        checkpoint = torch.load(pt_dir + f'CAUSE_{dataset}.pt', map_location='cpu')
        self.segment.load_state_dict(checkpoint['decoder'])
        self.segment.head_ema.codebook = checkpoint['codebook']

        self.pe = self.segment.head_ema.query_pos.clone()
        
        self.eval()
        self.to(device)

    def to(self, device):
        super().to(device)

        self.device = device
        self.pe = self.pe.to(device)
        self.segment.head_ema.codebook = self.segment.head_ema.codebook.to(device)
    
    @torch.no_grad()
    def forward(self, images):
        image_feat = self.backbone.forward_features(images)[:, 1:, :]

        fh, fw = get_feature_size(images.shape[2:], self.patch_size)
        pe = interpolate_pe(self.pe, fh*fw)

        return self.segment.head_ema(image_feat, (fh, fw), pe)
