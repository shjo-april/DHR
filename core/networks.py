import torch

from torch import nn
from torch.nn import functional as F

from .layers import Backbone, DeepLabv1_Head, DeepLabv2_Head, DeepLabv3plus_Head
from tools import torch_utils

class DeepLabv1(Backbone):
    def __init__(self, backbone, num_classes=20+1, output_stride=8):
        super().__init__(backbone, last_stride=1, output_stride=output_stride)
        self.seg_decoder = DeepLabv1_Head(self.in_channels[-1], num_classes)

    def forward(self, x):
        logits = self.seg_decoder(self.model(x)[-1])
        return {'logits': logits}
    
    @torch.no_grad()
    def apply_ms(self, image, scales=[1.0, 0.5, 1.5, 2.0], hflip=True, interpolation='bilinear'):
        size = None
        pred_masks = []

        for scale in scales:
            # rescale
            images = torch_utils.resize(image, image.shape[2:], scale=scale)
            if hflip: images = torch.cat([images, images.flip(-1)], dim=0)
            
            # inference
            output_dict = self.forward(images)
            if size is None: size = output_dict['logits'].shape[2:]
            
            # segmentation masks
            masks = F.softmax(output_dict['logits'], dim=1)
            masks = torch_utils.resize(masks, size, mode=interpolation)
            
            pred_masks.append(masks[0])
            if hflip: pred_masks.append(masks[1].flip(-1))
        
        return {'pred_mask': torch_utils.resize(torch.mean(torch.stack(pred_masks), dim=0), image.shape[2:])}

class DeepLabv2(Backbone):
    def __init__(self, backbone, num_classes=20+1, output_stride=8):
        super().__init__(backbone, last_stride=1, output_stride=output_stride)
        self.seg_decoder = DeepLabv2_Head(self.in_channels[-1], num_classes)
    
    def forward(self, x):
        logits = self.seg_decoder(self.model(x)[-1])
        return {'logits': logits}
    
    @torch.no_grad()
    def apply_ms(self, image, scales=[1.0, 0.5, 1.5, 2.0], hflip=True, interpolation='bilinear'):
        size = None
        pred_masks = []

        for scale in scales:
            # rescale
            images = torch_utils.resize(image, image.shape[2:], scale=scale)
            if hflip: images = torch.cat([images, images.flip(-1)], dim=0)
            
            # inference
            output_dict = self.forward(images)
            if size is None: size = output_dict['logits'].shape[2:]
            
            # segmentation masks
            masks = F.softmax(output_dict['logits'], dim=1)
            masks = torch_utils.resize(masks, size, mode=interpolation)
            
            pred_masks.append(masks[0])
            if hflip: pred_masks.append(masks[1].flip(-1))

        return {'pred_mask': torch_utils.resize(torch.mean(torch.stack(pred_masks), dim=0), image.shape[2:])}

class DeepLabv3plus(Backbone):
    def __init__(self, backbone, num_classes=20+1, feature_size=256, low_channels=48, output_stride=8):
        super().__init__(backbone, last_stride=1, output_stride=output_stride)
        self.seg_decoder = DeepLabv3plus_Head(
            self.in_channels[-1], feature_size, 
            self.in_channels[1] if len(self.in_channels) == 5 else self.in_channels[0], 
            low_channels, output_stride, num_classes
        )
        
    def forward(self, x, with_features=False):
        f_list = self.model(x)
        if len(f_list) == 4: C2, C3, C4, C5 = f_list
        else: C1, C2, C3, C4, C5 = f_list
        
        logits, _ = self.seg_decoder(C5, C2)
        
        output_dict = {'logits': logits}
        if with_features: output_dict['features'] = {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}
        return output_dict
    
    @torch.no_grad()
    def apply_ms(self, image, scales=[1.0, 0.5, 1.5, 2.0], hflip=True, interpolation='bilinear', with_features=False):
        size = None
        pred_masks = []

        if with_features: 
            pred_features = None
        
        for scale in scales:
            # rescale
            images = torch_utils.resize(image, image.shape[2:], scale=scale)
            if hflip: images = torch.cat([images, images.flip(-1)], dim=0)
            
            # inference
            output_dict = self.forward(images, with_features)

            if size is None: size = output_dict['logits'].shape[2:]
            
            # segmentation masks
            masks = F.softmax(output_dict['logits'], dim=1)
            masks = torch_utils.resize(masks, size, mode=interpolation)
            
            pred_masks.append(masks[0])
            if hflip: pred_masks.append(masks[1].flip(-1))

            if with_features: 
                if pred_features is None:
                    pred_features = [[] for _ in range(len(output_dict['features']))]
                
                for i, f in enumerate(output_dict['features'].values()):
                    f = torch_utils.resize(f, size, mode=interpolation)
                    pred_features[i].append(f[0])
                    if hflip: pred_features[i].append(f[1].flip(-1))
        
        output_dict = {'pred_mask': torch_utils.resize(torch.mean(torch.stack(pred_masks), dim=0), image.shape[2:])}
        
        if with_features: 
            for i in range(len(pred_features)):
                pred_features[i] = F.normalize(torch.mean(torch.stack(pred_features[i]), dim=0), dim=0)
            output_dict['pred_features'] = pred_features
        
        return output_dict

class DeepLabv3plus_with_DHR(Backbone):
    def __init__(self, backbone, num_seg_classes=21, num_cls_classes=20, feature_size=256, low_channels=48, output_stride=8):
        super().__init__(backbone, last_stride=1, output_stride=output_stride)
        self.seg_decoder = DeepLabv3plus_Head(self.in_channels[-1], feature_size, self.in_channels[1], low_channels, output_stride, num_seg_classes)
        self.classifier = nn.Conv2d(self.in_channels[-1], num_cls_classes, kernel_size=1, bias=False)

    def forward(self, x, return_cam=False, with_features=False):
        C1, C2, C3, C4, C5 = self.model(x)

        cams = self.classifier(torch_utils.resize(C5, None, scale=0.5))
        output_dict = {'cams': cams}
        
        if with_features: output_dict['features'] = {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}

        if return_cam:
            return output_dict
        
        logits, _ = self.seg_decoder(C5, C2)
        output_dict['logits'] = logits
        
        return output_dict

    @torch.no_grad()
    def apply_ms(self, image, scales=[1.0, 0.5, 1.5, 2.0], hflip=True, interpolation='bilinear', with_features=False, with_cam=False, with_seg=True):
        size = None
        if with_seg:
            pred_masks = []

        if with_features: 
            pred_features = None
        
        cam_size = None
        if with_cam:
            pred_cams = []
            pred_classes = []
        
        for scale in scales:
            # rescale
            images = torch_utils.resize(image, image.shape[2:], scale=scale)
            if hflip: images = torch.cat([images, images.flip(-1)], dim=0)
            
            # inference
            output_dict = self.forward(images, not with_seg, with_features)

            if size is None and with_seg: size = output_dict['logits'].shape[2:]
            if cam_size is None and with_cam: cam_size = output_dict['cams'].shape[2:]
            
            # segmentation masks
            if with_seg:
                masks = F.softmax(output_dict['logits'], dim=1)
                masks = torch_utils.resize(masks, size, mode=interpolation)

                pred_masks.append(masks[0])
                if hflip: pred_masks.append(masks[1].flip(-1))

            if with_cam:
                if scale == 1.0:
                    pred_classes.append(torch.sigmoid(F.adaptive_avg_pool2d(output_dict['cams'], (1, 1))[:, :, 0, 0].mean(dim=0)))
                
                cam = F.relu(torch_utils.resize(output_dict['cams'], cam_size, mode=interpolation))
                pred_cams.append(cam[0])
                if hflip: pred_cams.append(cam[1].flip(-1))

            if with_features: 
                if pred_features is None:
                    pred_features = [[] for _ in range(len(output_dict['features']))]

                for i, f in enumerate(output_dict['features'].values()):
                    f = torch_utils.resize(f, cam_size, mode=interpolation)
                    pred_features[i].append(f[0])
                    if hflip: pred_features[i].append(f[1].flip(-1))
        
        output_dict = {}

        if with_seg:
            output_dict['pred_mask'] = torch_utils.resize(torch.mean(torch.stack(pred_masks), dim=0), image.shape[2:])

        if with_cam: 
            pred_cam = torch.sum(torch.stack(pred_cams), dim=0)
            pred_cam /= (F.adaptive_max_pool2d(pred_cam, (1, 1)) + 1e-5)
            output_dict['pred_cam'] = pred_cam
            output_dict['pred_class'] = torch.stack(pred_classes).mean(dim=0)

        if with_features: 
            for i in range(len(pred_features)):
                pred_features[i] = F.normalize(torch.mean(torch.stack(pred_features[i]), dim=0), dim=0)
            output_dict['pred_features'] = pred_features
        
        return output_dict
