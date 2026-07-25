import cv2
import torch
import random
import numpy as np

from PIL import Image

from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform(da_list, args):
    transform_dict = {}
    if 'RandomRescale' in da_list: transform_dict['RandomRescale'] = RandomRescale(args.min_scale, args.max_scale)
    if 'RandomResize' in da_list: transform_dict['RandomResize'] = RandomResize(args.min_image, args.max_image)
    if 'RandomHFlip' in da_list: transform_dict['RandomHFlip'] = RandomHFlip()
    if 'RandomBlur' in da_list: transform_dict['RandomBlur'] = RandomBlur()
    if 'RandomGray' in da_list: transform_dict['RandomGray'] = RandomGray()
    if 'ColorJitter' in da_list: transform_dict['ColorJitter'] = ColorJitter(args.b_factor, args.c_factor, args.s_factor, args.h_factor)
    if 'Normalize' in da_list: transform_dict['Normalize'] = Normalize()
    if 'RandomCrop' in da_list: transform_dict['RandomCrop'] = RandomCrop(args.image)
    return Compose([transform_dict[da] for da in da_list.split(',')])

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, output_dict):
        isdict = isinstance(output_dict, dict)
        if not isdict: output_dict = {'image': output_dict}

        for transform in self.transforms: 
            output_dict = transform(output_dict)
        
        if not isdict: output_dict = output_dict['image']
        return output_dict
    
    def __repr__(self):
        text = 'Compose(\n'
        for transform in self.transforms:
            text += '\t{}\n'.format(transform)
        text += ')'
        return text

class Resize:
    def __init__(self, image_size=512, interpolation=Image.BICUBIC):
        self.image_size = image_size
        self.interpolation = interpolation
        self.key_dict = {'image': interpolation, 'mask': Image.NEAREST}
    
    def resize(self, image, size, mode):
        w, h = image.size
        if w < h: scale = size / h
        else: scale = size / w
        
        size = (int(round(w*scale)), int(round(h*scale)))
        if size[0] == w and size[1] == h: return image
        else: return image.resize(size, mode)
    
    def __call__(self, output_dict):
        for key in output_dict:
            if 'image' in key: output_dict[key] = self.resize(output_dict[key], self.image_size, self.key_dict['image'])
            elif 'mask' in key: output_dict[key] = self.resize(output_dict[key], self.image_size, self.key_dict['mask'])
        return output_dict

    def __repr__(self):
        return f'Resize ({self.image_size}, {self.interpolation})'

class ConditionalResize(Resize):
    def __init__(self, image_size=512):
        super().__init__(image_size)
    
    def __call__(self, output_dict):
        w, h = output_dict['image'].size
        
        if max(w, h) > self.image_size:
            for key in output_dict:
                if 'image' in key: output_dict[key] = self.resize(output_dict[key], self.image_size, self.key_dict['image'])
                elif 'mask' in key: output_dict[key] = self.resize(output_dict[key], self.image_size, self.key_dict['mask'])
        
        return output_dict

    def __repr__(self):
        return f'ConditionalResize ({self.image_size})'

class RandomResize(Resize):
    def __init__(self, min_image=320, max_image=640):
        super().__init__(min_image)
        self.min_image = min_image
        self.max_image = max_image
    
    def __call__(self, output_dict):
        w, h = output_dict['image'].size
        image_size = random.randint(self.min_image, self.max_image)
        
        for key in output_dict:
            if 'image' in key: output_dict[key] = self.resize(output_dict[key], image_size, self.key_dict['image'])
            elif 'mask' in key: output_dict[key] = self.resize(output_dict[key], image_size, self.key_dict['mask'])
        
        return output_dict

    def __repr__(self):
        return f'RandomResize (min={self.min_image}, max={self.max_image})'

class RandomRescale(Resize):
    def __init__(self, min_scale, max_scale):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def get_scale(self):
        return self.min_scale + random.random() * (self.max_scale - self.min_scale)
    
    def resize(self, image, scale, mode):
        w, h = image.size
        size = (int(round(w*scale)), int(round(h*scale)))

        if size[0] == w and size[1] == h: return image
        else: return image.resize(size, mode)
    
    def __call__(self, output_dict):
        scale = self.get_scale()
        for key in output_dict:
            if 'image' in key: output_dict[key] = self.resize(output_dict[key], scale, self.key_dict['image'])
            elif 'mask' in key: output_dict[key] = self.resize(output_dict[key], scale, self.key_dict['mask'])
        return output_dict

    def __repr__(self):
        return f'RandomRescale ({self.min_scale}, {self.max_scale})'

class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, output_dict):
        if np.random.rand() <= self.p:
            for key in output_dict.keys():
                if 'image' in key or 'mask' in key:
                    output_dict[key] = output_dict[key].transpose(Image.FLIP_LEFT_RIGHT)
        return output_dict

    def __repr__(self):
        return f'RandomHFlip (p={self.p})'

class ColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    def __call__(self, output_dict):
        for key in output_dict:
            if 'image' in key: output_dict[key] = super().__call__(output_dict[key])
        return output_dict

class RandomGray(T.RandomGrayscale):
    def __init__(self, p=0.1):
        super().__init__(p)

    def __call__(self, output_dict):
        for key in output_dict:
            if 'image' in key: output_dict[key] = super().__call__(output_dict[key])
        return output_dict

class RandomBlur(T.GaussianBlur):
    def __init__(self, p=0.5, kernel=(5, 5)):
        super().__init__(kernel)
        self.p = p

    def __call__(self, output_dict):
        if np.random.rand() <= self.p:
            for key in output_dict:
                if 'image' in key: output_dict[key] = super().__call__(output_dict[key])
        return output_dict

class RandomCrop:
    def __init__(self, size, channels=3, bg_mask=255):
        if isinstance(size, int):
            self.w_size = size
            self.h_size = size
        elif isinstance(size, tuple):
            self.w_size, self.h_size = size

        self.bg_mask = bg_mask
        self.shape = (channels, self.h_size, self.w_size)

    def get_random_boxes(self, image):
        h, w = image.shape[1:]
        
        crop_h = min(self.h_size, h)
        crop_w = min(self.w_size, w)

        w_space = w - self.w_size
        h_space = h - self.h_size

        img_top = 0
        img_left = 0
        cont_top = 0
        cont_left = 0

        if w_space > 0: img_left = random.randrange(w_space + 1)
        else: cont_left = random.randrange(-w_space + 1)

        if h_space > 0: img_top = random.randrange(h_space + 1)
        else: cont_top = random.randrange(-h_space + 1)

        src_bbox = [img_left, img_top, img_left+crop_w, img_top+crop_h]
        dst_bbox = [cont_left, cont_top, cont_left+crop_w, cont_top+crop_h]

        return src_bbox, dst_bbox
    
    def __call__(self, output_dict):
        for key in output_dict:
            if 'image' in key:
                (sxmin, symin, sxmax, symax), (dxmin, dymin, dxmax, dymax) = self.get_random_boxes(output_dict[key])
                break
        
        for key in output_dict:
            if 'image' in key:
                cropped_image = np.zeros(self.shape, output_dict[key].dtype)
                cropped_image[:, dymin:dymax, dxmin:dxmax] = output_dict[key][:, symin:symax, sxmin:sxmax]
                output_dict[key] = cropped_image
            elif 'mask' in key:
                cropped_mask = np.ones(self.shape[1:], output_dict[key].dtype) * self.bg_mask
                cropped_mask[dymin:dymax, dxmin:dxmax] = output_dict[key][symin:symax, sxmin:sxmax]
                output_dict[key] = cropped_mask
        
        output_dict['crop_bbox'] = [dxmin, dymin, dxmax, dymax]
        
        return output_dict

    def __repr__(self):
        size = (self.w_size, self.h_size)
        return f'RandomCrop (size={size})'

class Normalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, output_dict):
        for key in output_dict:
            if 'image' in key: 
                image = np.asarray(output_dict[key], dtype=np.float32); h, w, c = image.shape

                # transpose (h, w, c) to (c, h, w) & normalize
                norm_image = np.zeros((c, h, w), np.float32)
                norm_image[0, ...] = (image[..., 0] / 255. - self.mean[0]) / self.std[0]
                norm_image[1, ...] = (image[..., 1] / 255. - self.mean[1]) / self.std[1]
                norm_image[2, ...] = (image[..., 2] / 255. - self.mean[2]) / self.std[2]

                output_dict[key] = norm_image
            elif 'mask' in key:
                output_dict[key] = np.asarray(output_dict[key], dtype=np.int64)
        return output_dict

    def __repr__(self):
        return f'Normalize (mean={self.mean}, std={self.std})'

class Denormalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray): 
            image = image.cpu().detach().numpy()

        image = image.transpose((1, 2, 0))
        image = (image * self.std) + self.mean
        image = (image * 255).astype(np.uint8)
        return image
    
    def __repr__(self):
        return f'Denormalize (mean={self.mean}, std={self.std})'

class ToTensor:
    def __init__(self):
        pass

    def __call__(self, output_dict):
        for key in output_dict:
            if 'image' in key: output_dict[key] = torch.from_numpy(output_dict[key])
            elif 'mask' in key: output_dict[key] = torch.from_numpy(output_dict[key])
        return output_dict
    
    def __repr__(self):
        return f'ToTensor()'