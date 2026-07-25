import numpy as np

class SemanticSegmentation:
    def __init__(self, class_names, ignore_index=255):
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index
        self.meter_dict = self.set()

    def set(self):
        meter_dict = {}
        for tag in ['P', 'T', 'TP']:
            meter_dict[tag] = np.zeros(self.num_classes, dtype=np.float32)
        return meter_dict
    
    def add(self, pred_mask, gt_mask=None):
        if isinstance(pred_mask, dict):
            for key in ['P', 'T', 'TP']:
                self.meter_dict[key] += pred_mask[key]
        else:
            obj_mask = gt_mask != self.ignore_index
            correct_mask = (pred_mask == gt_mask) * obj_mask

            for i in range(self.num_classes):
                self.meter_dict['P'][i] += np.sum((pred_mask==i)*obj_mask)
                self.meter_dict['T'][i] += np.sum((gt_mask==i)*obj_mask)
                self.meter_dict['TP'][i] += np.sum((gt_mask==i)*correct_mask)
    
    def get(self, st_cls_index=0):
        IoU_list = []
        FPR_list = [] # over activation
        FNR_list = [] # under activation

        TP = self.meter_dict['TP']
        P = self.meter_dict['P']
        T = self.meter_dict['T']
        # print(TP[0], P[0], T[0])

        IoU_dict = {}
        
        for i in range(st_cls_index, self.num_classes):
            union = (T[i] + P[i] - TP[i])
            if union <= 0:
                IoU_list.append(np.nan)
                FPR_list.append(np.nan)
                FNR_list.append(np.nan)
            else:
                IoU = float(TP[i] / union * 100)
                FPR = float((P[i] - TP[i]) / union)
                FNR = float((T[i] - TP[i]) / union)
                
                IoU_list.append(IoU)
                FPR_list.append(FPR)
                FNR_list.append(FNR)
                IoU_dict[self.class_names[i]] = {'IoU': IoU, 'FPR': FPR, 'FNR': FNR}
        
        mIoU = float(np.nanmean(IoU_list))
        mFPR = float(np.nanmean(FPR_list))
        mFNR = float(np.nanmean(FNR_list))
        IoU_dict['Average'] = {'mIoU': mIoU, 'mFPR': mFPR, 'mFNR': mFNR}
        
        self.meter_dict = self.set()
        
        return mIoU, mFPR, mFNR, IoU_dict
    
    def print(self, tag='', end='\n', ignore_bg=False):
        if len(tag) > 0: tag = f'# {tag:>30} | '
        
        mIoU, mFPR, mFNR, IoU_dict = self.get(int(ignore_bg))
        print(f'{tag}mIoU: {mIoU:.1f}%, mFPR: {mFPR:.3f}, mFNR: {mFNR:.3f}', end=end)
        
        return IoU_dict
