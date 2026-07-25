import os
import ray
import numpy as np

from tqdm import tqdm
from PIL import Image

from tools import evaluators, io_utils, json_utils, data_utils

@ray.remote
def update_mIoU(obj: evaluators.SemanticSegmentation, pred_mask, gt_mask, image_id):
    meter_dict = obj.set()

    obj_mask = gt_mask != obj.ignore_index
    correct_mask = (pred_mask == gt_mask) * obj_mask
    
    tags = []
    IoUs = []

    for i in range(obj.num_classes):
        meter_dict['P'][i] += np.sum((pred_mask==i)*obj_mask)
        meter_dict['T'][i] += np.sum((gt_mask==i)*obj_mask)
        meter_dict['TP'][i] += np.sum((gt_mask==i)*correct_mask)
        
        union = meter_dict['T'][i] + meter_dict['P'][i] - meter_dict['TP'][i]
        if union == 0:
            continue
        
        tags.append(obj.class_names[i])
        IoUs.append(float(meter_dict['TP'][i] / union))

    meter_dict['image_id'] = image_id
    meter_dict['mIoU'] = float(np.mean(IoUs))
    meter_dict['tags'] = tags
    meter_dict['IoUs'] = IoUs

    return meter_dict

def main(args):
    dataset = data_utils.Dataset(args.root + f'{args.data}/{args.data}.json')
    
    if not args.fix:
        pred_domain = 'train_aug' if args.data == 'VOC2012' and args.domain == 'train' else args.domain
        args.pred += f'{args.data}/{args.tag}/{pred_domain}/'
        args.gt += f'{args.data}/{args.domain}/mask/'
    
    evaluator = evaluators.SemanticSegmentation(dataset.class_names)

    params = []
    sample_dict = {}

    ignore_classes = [0] if 'background' in dataset.class_names else []
    ignore_classes += [255]
    
    ray.init(num_cpus=args.cpus, configure_logging=False)

    for image_name in tqdm(io_utils.listdir(args.gt)):
        image_id = image_name.replace('.png', '')
        
        gt_mask = np.asarray(Image.open(args.gt + image_name))
        gt_indices = sorted(list(np.unique(gt_mask)))
        gt_classes = [gt_index for gt_index in gt_indices if not gt_index in ignore_classes]
        if len(gt_classes) == 0:
            if args.data == 'COCO2014' and args.domain == 'train': pred_mask = np.zeros_like(gt_mask)
            elif args.data == 'COCO2014' and args.domain == 'validation': pass 
            else: continue
        
        if not os.path.isfile(args.pred + image_name):
            print(args.pred, image_name, gt_classes)
            continue
        
        pred_mask = np.asarray(Image.open(args.pred + image_name))
        
        if gt_mask.shape != pred_mask.shape:
            raise Exception(f'Size Error: {image_name} {gt_mask.shape} {pred_mask.shape}')
        
        params.append(update_mIoU.remote(evaluator, pred_mask, gt_mask, image_id))
    
    if len(params) > 0:
        for data in ray.get(params):
            evaluator.add(data)
            sample_dict[data['image_id']] = {'mIoU': data['mIoU'], 'tags': data['tags'], 'IoUs': data['IoUs']}
    
    evaluator.print(args.tag)

if __name__ == '__main__':
    args = io_utils.Parser().add_from_inputs(
        {
            'cpus': os.cpu_count(),
            'data': 'VOC2012', 'domain': 'validation', 'root': '../',
            'pred': './results/', 'gt': '../', 'tag': 'DHR', 'fix': False
        }
    )
    main(args)