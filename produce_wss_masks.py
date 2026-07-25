# Copyright (C) 2023 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import ray
import copy
import torch
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed

from PIL import Image
from torch.nn import functional as F

from core import networks, datasets, refinements, ot_utils

from tools import torch_utils, transforms
from tools import io_utils, data_utils, cv_utils

@delayed
def inference(model, test_dataset, indices, device, args, visualize):
    model = model.to(device)
    if visualize: pbar = tqdm(total=len(indices), desc=f'Inference ({device})')

    for index in indices:
        if visualize: pbar.update(1)

        image_id, image, tags = test_dataset[index]

        if len(tags) == 0: continue
        if os.path.isfile(args.temp + image_id + '.pt') or os.path.isfile(args.temp + image_id + '.png'): continue
        
        # preprocessing
        image = torch.from_numpy(image).to(device).unsqueeze(0)
        
        # inference
        with torch.no_grad():
            output_dict = model.apply_ms(image)
        
        pred_mask = output_dict['pred_mask']
        class_mask = torch_utils.get_numpy(torch.max(pred_mask.reshape(test_dataset.dataset.num_classes, -1), dim=1)[0]) > args.threshold

        class_keys = np.nonzero(class_mask)[0]
        pred_mask = torch_utils.get_numpy(pred_mask[class_mask, :, :].half())
        
        # save a torch tensor
        torch.save({'keys': class_keys, 'masks': pred_mask}, args.temp + image_id + '.pt')

@ray.remote
def apply_crf(image_id, image, temp, colors, denorm_fn, crf_fn, args):
    pseudo_path = temp + image_id + '.png'
    if os.path.isfile(pseudo_path):
        return
    
    if os.path.isfile(temp + f'{image_id}.pt'):
        infer_dict = torch.load(temp + f'{image_id}.pt')

        image = denorm_fn(image).copy()

        ih, iw = image.shape[:2]
        sh, sw = infer_dict['masks'].shape[:2]
        
        if ih != sh or iw != sw:
            masks = torch.from_numpy(infer_dict['masks'])
            masks = torch_utils.resize(masks, (ih, iw))
            infer_dict['masks'] = masks.numpy()
        
        pseudo_label = crf_fn(image, infer_dict['masks'])
        pseudo_label = np.argmax(pseudo_label, axis=0)
        pseudo_label = infer_dict['keys'][pseudo_label]

        image = Image.fromarray(pseudo_label.astype(np.uint8)).convert('P')
        image.putpalette(colors)
        image.save(pseudo_path)

        os.remove(temp + f'{image_id}.pt')

def main(args):
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, args.gpus)))

    # read dataset information
    dataset = data_utils.Dataset(args.root + f'{args.data}/{args.data}.json')

    # build model
    if args.decoder == 'deeplabv3+': model = networks.DeepLabv3plus(args.backbone, dataset.num_classes).cpu().eval()

    torch_utils.load_model(model, f'./experiments/models/{args.tag}/{args.checkpoint}.pth', strict=False)
    
    # create datasets
    test_transform = transforms.Compose([transforms.Normalize()])
    test_dataset = datasets.EvalDataset(args.root, args.domain, dataset, test_transform)

    # inference
    args.temp = io_utils.create_dir(f'./experiments/results/{args.data}/{args.tag}@{args.checkpoint}/{args.domain}/')

    length = len(test_dataset)
    indices = np.arange(length)
    length_per_gpu = len(indices) // len(args.gpus)

    params = []
    for gpu_index in range(len(args.gpus)-1):
        param = [copy.deepcopy(model), test_dataset, indices[:length_per_gpu], torch.device('cuda', gpu_index), args, False]
        params.append(param); indices = indices[length_per_gpu:]
    
    param = [copy.deepcopy(model), test_dataset, indices, torch.device('cuda', len(args.gpus)-1), args, True]
    params.append(param)

    Parallel(n_jobs=len(params))([inference(*param) for param in params])
    torch.cuda.empty_cache()

    # CRF
    denorm_fn = transforms.Denormalize()
    crf_fn = refinements.DenseCRF(for_seg=True)

    params = []
    colors = cv_utils.get_colors(dataset.num_classes, data=args.data)

    ray.init(num_cpus=args.cpus)

    for image_id, image, tags in tqdm(test_dataset):
        if len(tags) == 0: continue
        if os.path.isfile(args.temp + image_id + '.png'): continue
        
        params.append(apply_crf.remote(image_id, image, args.temp, colors, denorm_fn, crf_fn, args))
    
    ray.get(params)

if __name__ == '__main__':
    parser = io_utils.Parser()
    parser.add_from_inputs(
        {
            'cpus': io_utils.cpus(), 'gpus': [0], 'data': 'VOC2012', 'domain': 'train_aug', 'root': '../',
            'backbone': 'resnet101', 'decoder': 'deeplabv3+', 'tag': 'ResNet-101@VOC2012@DeepLabv3+@DHR', 'checkpoint': 'last', 'ot': False,
            'threshold': 0.05, # to reduce inference time related to CRF
        }
    )
    main(parser.get())
