import os
import torch
import shutil

import numpy as np

from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from core import networks, datasets, ot_utils, refinements

from tools import io_utils, torch_utils, data_utils, cv_utils
from tools import trainers, optimizers, transforms as T

def collate(batch):
    images = []
    masks = []
    labels = []
    crop_bboxes = []
    f_us_list = []
    
    for image, mask, label, crop_bbox, f_us in batch:
        images.append(torch.from_numpy(image))
        masks.append(torch.from_numpy(mask))
        labels.append(torch.from_numpy(label))
        crop_bboxes.append(crop_bbox)
        f_us_list.append(f_us)
    
    return {
        'images': torch.stack(images),
        'masks': torch.stack(masks),
        'labels': torch.stack(labels),
        'crop_bboxes': np.stack(crop_bboxes),
        'f_us': torch.stack(f_us_list),
    }

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def main(args):
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    gpus = args.gpus.split(',')
    device = torch.device('cuda', 0)

    if RANK in [-1, 0]:
        for i, d in enumerate(gpus):
            p = torch.cuda.get_device_properties(i)
            print(f'[i] CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)')

    if LOCAL_RANK != -1:
        from torch import distributed as dist

        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)

        from datetime import timedelta
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=60))

        assert args.batch_size % WORLD_SIZE == 0, '--batch must be multiple of CUDA device count'
    
    # set directories
    model_dir = f'./experiments/models/{args.tag}/'
    
    txt_path = model_dir + f'{args.tag}.txt'
    if os.path.isfile(txt_path) and RANK in [-1, 0]:
        if input('Found existing logs. yes=remove, no=keyboardinterept') == 'no': raise KeyboardInterrupt
        else:
            if os.path.isdir(model_dir): shutil.rmtree(model_dir)
    
    log_fn = lambda string='': io_utils.log(string, txt_path)
    model_dir = io_utils.create_dir(model_dir) if RANK in [-1, 0] else model_dir

    # read dataset information
    dataset = data_utils.Dataset(args.root + f'{args.data}/{args.data}.json')
    
    # create model
    if args.decoder == 'deeplabv3+':
        model = networks.DeepLabv3plus_with_DHR(
            args.backbone, dataset.num_classes, 
            dataset.num_classes-1 if 'background' in dataset.class_names else dataset.num_classes
        ).to(device)
    
    if RANK in [-1, 0]:
        num_params = torch_utils.calculate_parameters(model.parameters()) 
        log_fn(f'[i] Backbone: {args.backbone} ({num_params:.1f}MB)\n')

    # define loss functions
    cls_loss_fn = nn.MultiLabelSoftMarginLoss().to(device) 
    seg_loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.ignore).to(device)

    ot_fn = ot_utils.OptimalTransport()
    denorm_fn = T.Denormalize()
    crf_fn = refinements.DenseCRF(for_seg=True)
    ignore_key = torch.from_numpy(np.asarray([dataset.ignore])).to(device).long()

    def produce_wss_vectors(f_ws, wss_mask):
        ih, iw = wss_mask.shape
        fh, fw = f_ws.shape[1:]

        while iw % fw != 0: iw += 1
        while ih % fh != 0: ih += 1

        emb_dict = {}
        wss_mask = cv_utils.resize(wss_mask, (iw, ih), mode='nearest')

        pw, ph = iw // fw, ih // fh
        
        for fy in range(fh):
            y = int(fy / fh * ih)
            for fx in range(fw):
                x = int(fx / fw * iw)

                class_indices, class_counts = np.unique(wss_mask[y:y+ph, x:x+pw], return_counts=True)

                class_index = class_indices[np.argmax(class_counts)]
                if class_index in [255]: continue
                
                try: emb_dict[class_index].append(f_ws[:, fy, fx])
                except KeyError: emb_dict[class_index] = [f_ws[:, fy, fx]]

        return {int(class_index): torch.stack(emb_dict[class_index]).mean(dim=0) for class_index in sorted(list(emb_dict.keys()))}
    
    def produce_uss_vectors(f_us, wss_mask, patch_size=14): # patch size is the same as USS configurations
        ih, iw = wss_mask.shape
        fh, fw = f_us.shape[1:]

        while iw % patch_size != 0: iw += 1
        while ih % patch_size != 0: ih += 1

        emb_dict = {}
        wss_mask = cv_utils.resize(wss_mask, (iw, ih), mode='nearest')

        pw, ph = iw // fw, ih // fh
        
        for fy in range(fh):
            y = int(fy / fh * ih)
            for fx in range(fw):
                x = int(fx / fw * iw)

                class_indices, class_counts = np.unique(wss_mask[y:y+ph, x:x+pw], return_counts=True)
                class_index = class_indices[np.argmax(class_counts)]

                if class_index == 255:
                    continue
                
                try: emb_dict[class_index].append(f_us[:, fy, fx])
                except KeyError: emb_dict[class_index] = [f_us[:, fy, fx]]

        return {int(class_index): torch.stack(emb_dict[class_index]).mean(dim=0) for class_index in sorted(list(emb_dict.keys()))}

    def normalize(_heatmaps):
        min_mask, max_mask = torch_utils.get_minmax(_heatmaps)
        _heatmaps = (_heatmaps-min_mask)/(max_mask-min_mask)
        return _heatmaps.clip(min=0, max=1)

    # define trainer 
    class Trainer(trainers.BaseTrainer):
        def __init__(self):
            param = trainers.Parameter(
                args.seed, True, args.ema,
                args.epochs, model_dir, RANK
            )
            super().__init__(model, device, param)

            self.best_mIoU_val = 0
            self.best_mIoU_train = 0
        
        def prepare_dataset(self):
            train_transform = T.get_transform(args.train_transform, args)
            test_transform = T.get_transform(args.test_transform, args)

            if RANK in [-1, 0]:
                log_fn(f'Training augmentation: {train_transform}')
                log_fn(f'Testing augmentation: {test_transform}')
            
            self.train_dataset = datasets.DHRDataset(args.root, args.train, dataset, train_transform, args.wss, args.uss)
            self.valid_dataset = datasets.SegmentationDataset(args.root, args.valid, dataset, test_transform)
        
        def prepare_loader(self, is_print=True):
            if RANK != -1:
                shuffle = False
                train_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            else:
                shuffle = True
                train_sampler = None
            
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch // WORLD_SIZE, num_workers=args.cpus, shuffle=shuffle, drop_last=True, pin_memory=True, sampler=train_sampler, collate_fn=collate)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, num_workers=max(args.cpus // 4, 1), shuffle=False, drop_last=False, pin_memory=True)
            
            if RANK in [-1, 0] and is_print:
                log_fn('The size of training set: {}'.format(len(self.train_dataset)))
                
        def configure_optimizers(self):
            self.optimizer = optimizers.SGD(
                params=[
                    {'params': self.param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[1], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[3], 'lr': 10*args.lr, 'weight_decay': args.wd},
                ],
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov,
                scheduler_option={
                    'scheduler': args.scheduler,
                    'power': 0.9,
                    'max_iterations': self.train_iterations * self.param.max_epochs
                }
            )
        
        def forward(self, data, training: bool=True):
            if training:
                images = data['images'].to(self.device)
                pseudo_masks = data['masks'].to(self.device)
                uss_features = data['f_us'].to(self.device)

                labels = data['labels'].to(self.device)
                crop_bboxes = data['crop_bboxes']
                
                output_dict = self.model(images, with_features=True)
                logits, cams = output_dict['logits'], output_dict['cams']

                ih, iw = images.shape[2:]
                fh, fw = cams.shape[2:]

                # Dual Features-Driven Hierarchical Rebalancing
                with torch.no_grad():
                    refined_cams = F.relu(cams)
                    refined_cams /= F.adaptive_max_pool2d(refined_cams, (1, 1)).clip(min=1e-5)
                    refined_cams *= labels[:, :, None, None]

                    for b in range(labels.shape[0]):
                        # Step 1: OT-based Seed Initialization
                        xmin, ymin, xmax, ymax = crop_bboxes[b]

                        xmin_cam = int(np.floor(xmin / iw * fw))
                        ymin_cam = int(np.floor(ymin / ih * fh))
                        xmax_cam = int(np.floor(xmax / iw * fw))
                        ymax_cam = int(np.floor(ymax / ih * fh))

                        target_cams = refined_cams[b, labels[b] == 1, ymin_cam:ymax_cam, xmin_cam:xmax_cam]

                        if torch.sum(labels[b]).item() > 1:
                            Ts = ot_fn.apply(target_cams.view(target_cams.shape[0], -1).transpose(1, 0))
                            target_cams *= Ts.transpose(1, 0).view(*target_cams.shape)

                        hr_cams = torch_utils.resize(target_cams, (ymax-ymin, xmax-xmin), mode='nearest')
                        hr_cams = crf_fn(denorm_fn(torch_utils.get_numpy(images[b, :, ymin:ymax, xmin:xmax])), torch_utils.get_numpy(hr_cams))
                        hr_cams = torch.from_numpy(hr_cams).to(self.device)

                        class_keys = torch.nonzero(labels[b]==1, as_tuple=True)[0]
                        if 'background' in dataset.class_names: class_keys += 1
                        class_keys = torch.cat([class_keys, ignore_key], dim=0)

                        hr_cams = torch.cat([hr_cams, 1. - torch.max(hr_cams, dim=0, keepdim=True)[0]], dim=0)
                        hr_seed = class_keys[torch.argmax(hr_cams, dim=0)]
                        pseudo_masks[b, ymin:ymax, xmin:xmax][hr_seed != dataset.ignore] = hr_seed[hr_seed != dataset.ignore]

                        # Step 2: USS Rebalancing
                        f_us = uss_features[b]
                        uss_dict = produce_uss_vectors(f_us, pseudo_masks[b])

                        uss_tags = list(uss_dict.keys())
                        uss_embs = torch.stack([uss_dict[tag] for tag in uss_tags])

                        v_us = torch_utils.normalize(torch.stack(uss_embs), dim=1)
                        heatmaps_us = torch_utils.cosine(
                            torch_utils.normalize(f_us, dim=0)[None, ...], # 1 x D_us x H x W
                            v_us[:, :, None, None],                        # C x D_us x 1 x 1
                            dim=1
                        )
                        heatmaps_us = normalize(heatmaps_us)

                        # Step 3: WSS Rebalancing
                        heatmaps_dhr = heatmaps_us.clone()

                        correlation = F.relu(v_us @ v_us.transpose(1, 0))
                        for i in range(correlation.shape[0]):
                            correlation[i, :i] = 0.

                        f_ws_list = [torch_utils.resize(output_dict['features'][i][b], f_us.shape[1:]).cpu() for i in ['C4', 'C5']]
                        
                        if torch.max(torch.sum(correlation > args.tau, dim=1)).item() > 1:
                            heatmaps_ws = []
                            for f_ws in f_ws_list:
                                wss_dict = produce_wss_vectors(f_ws, pseudo_masks[b])
                                wss_dict = {args.dataset[index]: emb for index, emb in wss_dict.items()}
                                
                                v_ws = []
                                for tag in uss_tags:
                                    v_ws.append(wss_dict[tag])
                                
                                v_ws = torch_utils.normalize(torch.stack(v_ws), dim=1)
                                _heatmaps_ws = torch_utils.cosine(
                                    torch_utils.normalize(f_ws, dim=0)[None, ...], # 1 x D_ws x H x W
                                    v_ws[:, :, None, None],                        # C x D_ws x 1 x 1
                                    dim=1
                                )
                                heatmaps_ws.append(normalize(_heatmaps_ws))
                            heatmaps_ws = normalize(torch.stack(heatmaps_ws).mean(dim=0))

                            for i in range(correlation.shape[0]):
                                target_mask = correlation[i] > args.tau
                                if target_mask.sum() > 1:
                                    target_heatmaps_ws = heatmaps_ws[target_mask]
                                    T = ot_fn.apply(target_heatmaps_ws.view(target_heatmaps_ws.shape[0], -1).transpose(1, 0))
                                    T = T.transpose(1, 0).view(*target_heatmaps_ws.shape)
                                    heatmaps_dhr[target_mask] *= T
                        
                        T = ot_fn.apply(heatmaps_dhr.view(heatmaps_dhr.shape[0], -1).transpose(1, 0))
                        heatmaps_dhr *= T.transpose(1, 0).view(*heatmaps_dhr.shape)
                        heatmaps_dhr = normalize(heatmaps_dhr)
                        
                        class_keys = list([args.dataset[tag] for tag in uss_tags])
                        pseudo_label = crf_fn(denorm_fn(torch_utils.get_numpy(images[b, :, ymin:ymax, xmin:xmax])), torch_utils.get_numpy(heatmaps_dhr))
                        pseudo_label = np.argmax(pseudo_label, axis=0)
                        pseudo_label = class_keys[pseudo_label]
                        pseudo_masks[b, ymin:ymax, xmin:xmax] = torch.from_numpy(pseudo_label).cuda()
                
                # for segmentation loss
                loss_seg = seg_loss_fn(logits, torch_utils.resize(pseudo_masks.float(), logits.shape[2:], mode='nearest').long())
                
                # for classification loss
                logits_cls = []
                ih, iw = images.shape[2:]
                fh, fw = cams.shape[2:]

                for b in range(len(crop_bboxes)):
                    xmin, ymin, xmax, ymax = crop_bboxes[b]

                    xmin = int(np.floor(xmin / iw * fw))
                    ymin = int(np.floor(ymin / ih * fh))
                    xmax = int(np.floor(xmax / iw * fw))
                    ymax = int(np.floor(ymax / ih * fh))

                    logits_cls.append(F.adaptive_avg_pool2d(output_dict['cams'][b, :, ymin:ymax, xmin:xmax].unsqueeze(0), (1, 1))[0, :, 0, 0])
                
                logits_cls = torch.stack(logits_cls)
                loss_cls = cls_loss_fn(logits_cls, labels)

                loss = loss_cls + loss_seg

                return loss, {
                    'LR': self.get_learning_rate(), 
                    'L_total': loss.item(),
                    'L_cls': loss_cls.item(),
                    'L_seg': loss_seg.item(),
                }
    
    trainer = Trainer()
    for epoch in range(trainer.epoch, args.epochs+1):
        train_dict = trainer.training_step()
        log_fn('Epoch: {epoch:,}, LR: {LR:.6f}, L_total: {L_total:.3f}, L_cls: {L_cls:.3f}, L_seg: {L_seg:.3f}, {time:.0f}s'.format(**train_dict))
        trainer.save_model(model_dir + 'last.pth')

if __name__ == '__main__':
    parser = io_utils.Parser()
    parser.add_from_inputs(
        {
            'local_rank': -1, 'gpus': '0', 'cpus': 8, 'seed': 1,
            'root': '../', 'data': 'VOC2012', 'train': 'train_aug', 'valid': 'validation',
            'backbone': 'resnet101', 'decoder': 'deeplabv3+', 'wss': 'MARS', 'uss': 'CAUSE', 'tag': 'ResNet-101@', 'tau': 0.8,
            'image': 512, 'batch': 16, 'epochs': 100, 
            'lr': 1e-3, 'wd': 4e-5, 'optimizer': 'SGD', 'momentum': 0.9, 'nesterov': False, 'scheduler': 'PolyLR', 'ema': 0.999, 
            'min_image': 320, 'max_image': 640, 'b_factor': 0.3, 'c_factor': 0.3, 's_factor': 0.3, 'h_factor': 0.1,
            'train_transform': 'RandomResize,RandomHFlip,ColorJitter,Normalize,RandomCrop', 
            'test_transform': 'Normalize',
        }
    )
    main(parser.get())