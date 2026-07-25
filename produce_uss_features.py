import os
import torch
import numpy as np

from core.uss import inference
from core.uss import DINOv1, DINOv2, STEGO, CAUSE

from tools import io_utils, cv_utils, transforms as T

def inference_for_uss(args, indices, device, visualize):
    if args.uss == 'STEGO': model = STEGO(args.backbone, args.data)
    elif args.uss == 'CAUSE': model = CAUSE(args.backbone, args.data)
    else: raise NotImplementedError(f"Please implement an unsupervised method ({args.uss})")
    
    model.to(device)

    image_transform = T.Compose([T.Normalize(), T.ToTensor()])

    oom_count = 0
    if visualize: pbar = io_utils.Progress(len(indices), f'Inference ({device})')
    
    for index in indices:
        if visualize: pbar.update()
        
        image_path = args.image_paths[index]
        uss_path = args.uss_dir + os.path.basename(image_path).replace('.jpg', '.pt')

        if os.path.isfile(uss_path): 
            continue

        try:
            image = image_transform(cv_utils.imread(image_path, 'pillow').convert('RGB'))
            f_us = inference(model, image)
            torch.save(f_us.half().cpu(), uss_path)
        except RuntimeError:
            oom_count += 1
            io_utils.log(f'{args.uss} | CUDA OOM ({oom_count:05d}) | {image_path} | {image.shape}', f'error_{args.uss}_{args.data}.txt')

if __name__ == '__main__':
    args = io_utils.Parser(
        {
            'gpus': [0], 'root': '../', 'data': 'VOC2012', 'domain': 'train_aug',
            'uss': 'CAUSE', 'backbone': 'ViT-G',
        }
    ).get()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, args.gpus)))
    
    args.uss_dir = io_utils.create_dir(args.root + f'{args.data}/' + args.domain + f'/{args.uss}/')
    args.image_paths = io_utils.listdir(args.root + f'{args.data}/' + args.domain + '/image/' + f'*')

    params = []
    indices = np.arange(len(args.image_paths))
    length_per_gpu = len(indices) // len(args.gpus)
    
    for gpu_index in range(len(args.gpus)-1):
        param = [args, indices[:length_per_gpu], torch.device('cuda', gpu_index), False]
        params.append(param); indices = indices[length_per_gpu:]

    params.append([args, indices, torch.device('cuda', len(args.gpus)-1), True])

    io_utils.parallel(inference_for_uss, params, len(params))
