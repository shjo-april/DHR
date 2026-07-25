import torch
import numpy as np
from PIL import Image

from tools import data_utils, xml_utils, io_utils

def get_onehot(tags, classes, class_dict):
    """Embedding labels to one-hot form.
    """
    vector = np.zeros((classes), dtype=np.float32)
    for tag in tags: vector[class_dict[tag]] = 1.
    return vector

class ConventionalDataset:
    def __init__(self, root_dir: str, domain: str, dataset: data_utils.Dataset, transform=None, return_formats=[]):
        self.domain = domain
        self.dataset = dataset
        self.transform = transform

        self.return_formats = return_formats
        self.return_dict = {
            'id': self.get_id,
            'image': self.get_image,
            'mask': self.get_mask,
            'tags': self.get_tags,
        }
        
        self.image_dir = root_dir + f'{dataset.tag}/{domain}/image/'
        self.mask_dir = root_dir + f'{dataset.tag}/{domain}/mask/'
        self.xml_dir = root_dir + f'{dataset.tag}/{domain}/xml/'
        self.json_dir = root_dir + f'{dataset.tag}/{domain}/json/'

        self.image_names = io_utils.listdir(self.image_dir)
        if 'tags' in return_formats: self.image_names, self.balance_dict = self.reject_empty_tags()

    def __len__(self): return len(self.image_names)
    def get_id(self, image_name: str): return image_name.replace('.jpg', '')
    def get_image(self, image_name: str): return Image.open(self.image_dir + image_name).convert('RGB')
    def get_mask(self, image_name: str): return Image.open(self.mask_dir + image_name.replace('.jpg', '.png'))
    def get_tags(self, image_name: str): return xml_utils.read_tags(self.xml_dir + image_name.replace('.jpg', '.xml'))
    
    def __getitem__(self, i):
        image_name = self.image_names[i]
        output_dict = {fmt: self.return_dict[fmt](image_name) for fmt in self.return_formats}
        if self.transform is not None: output_dict = self.transform(output_dict)
        return output_dict
    
    def reject_empty_tags(self): 
        image_names = []
        balance_dict = {}

        for name in io_utils.progress(self.image_names):
            tags = self.get_tags(name)

            if len(tags) > 0:
                image_names.append(name)
                for tag in tags:
                    try: balance_dict[tag].append(name)
                    except KeyError: balance_dict[tag] = [name]
        
        return image_names, balance_dict
    
class SegmentationDataset(ConventionalDataset):
    def __init__(self, root_dir: str, domain: str, dataset: data_utils.Dataset, transform=None, wss=''):
        super().__init__(root_dir, domain, dataset, transform, ['image', 'mask'])
        if len(wss) > 0: self.mask_dir = f'../WSS/{dataset.tag}/{wss}/{domain}/'
        self.image_names, self.balance_dict = self.reject_empty_tags()
    
    def __getitem__(self, i):
        output_dict = super().__getitem__(i)
        return output_dict['image'], output_dict['mask']

class EvalDataset(ConventionalDataset):
    def __init__(self, root_dir: str, domain: str, dataset: data_utils.Dataset, transform=None, without_rejection=False):
        self.without_rejection = without_rejection
        if domain in ['test'] or self.without_rejection: tags = ['id', 'image']
        else: tags = ['id', 'image', 'tags']
        super().__init__(root_dir, domain, dataset, transform, tags)
    
    def __getitem__(self, i):
        output_dict = super().__getitem__(i)
        return output_dict['id'], output_dict['image'], [] if self.domain in ['test'] or self.without_rejection else output_dict['tags']
    
class DHRDataset(ConventionalDataset):
    def __init__(self, root_dir: str, domain: str, dataset: data_utils.Dataset, transform=None, wss='', uss=''):
        super().__init__(root_dir, domain, dataset, transform, ['image', 'mask', 'tags'])
        self.transform = transform

        self.mask_dir = f'../WSS/{dataset.tag}/{wss}/{domain}/'
        self.uss_dir = root_dir + f'{dataset.tag}/{domain}/{uss}/'
    
    def __getitem__(self, i):
        output_dict = super().__getitem__(i)
        f_us = torch.load(self.uss_dir + self.get_id(self.image_names[i]) + '.pt', map_location='cpu').float()
        
        label = get_onehot(output_dict['tags'], self.dataset.num_classes, self.dataset.class_dict)
        if 'background' in self.dataset.class_names: label = label[1:]
        
        return output_dict['image'], output_dict['mask'], label, output_dict['crop_bbox'], f_us
