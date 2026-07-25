import sys
import tqdm
import torch
import numpy as np

from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter

from . import io_utils, time_utils
from .torch_utils import set_seed, ExponentialMovingAverage, de_parallel, save_model

@dataclass
class Parameter:
    seed: int

    use_ema: bool
    ema_decay: float

    max_epochs: int
    tensorboard_dir: str

    # for ddp
    RANK: int

class BaseTrainer:
    """
    1. prepare_dataset
    2. prepare_loader
    3. configure_optimizers
    4. training step
    5. evaluation step
    """
    def __init__(self, model, device, param: Parameter):
        self.param = param
        self.device = device

        self.model = model
        self.param_groups = de_parallel(self.model).get_parameters()

        set_seed(self.param.seed)

        self.prepare_dataset()
        self.prepare_loader()

        # set variables
        self.epoch = 1
        self.iteration = 1

        self.best_valid_score = 0

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)
        self.max_iterations = len(self.train_loader) * self.param.max_epochs

        self.progress_fmt = 'Epoch=%0{}d'.format(io_utils.get_num_digits(self.param.max_epochs))

        self.configure_optimizers()

        self.evaluator = None 
        self.writer = SummaryWriter(self.param.tensorboard_dir)

        self.ema = None
        if self.param.use_ema:
            self.ema = ExponentialMovingAverage(self.model, self.param.ema_decay)

    # Related to dataset
    def prepare_dataset(self):
        self.train_datset = None
        self.valid_dataset = None
    
    def prepare_loader(self):
        self.train_loader = None
        self.valid_loader = None

    def configure_optimizers(self):
        self.optimizer = None
        self.scheduler = None

    # Related to checkpoints
    def save(self, checkpoint_path: str):
        torch.save(
            {
                'epoch': self.epoch,
                'iteration': self.iteration,
                'model': self.model.state_dict(),
                'ema': self.ema.get_model().state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, checkpoint_path
        )
    
    def load(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path)

        self.epoch = state_dict['epoch']
        self.iteration = state_dict['iteration']

        self.model.load_state_dict(state_dict['model'])
        self.ema.get_model().load_state_dict(state_dict['ema'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
    
    # Related to training and evaluation
    def forward(self, data, training: bool=True):
        raise NotImplementedError
    
    def training_step(self, debug=False):
        st_time = time_utils.tik()

        if self.param.RANK != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        
        data_dict = {}
        if self.param.RANK in [-1, 0]:
            progress = tqdm.tqdm(total=len(self.train_loader), desc=self.progress_fmt%(self.epoch))

        for data in self.train_loader:
            loss, tb_dict = self.forward(data, training=True)
            
            # add data in dictionary
            for key in tb_dict.keys():
                if not key in data_dict: data_dict[key] = []        
                data_dict[key].append(tb_dict[key])
            
            # update weights with gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update ema
            if self.param.RANK in [-1, 0]:
                self.update_ema()
            
            if self.param.RANK in [-1, 0]:
                progress.update(1)
                if self.iteration % 10 == 0:
                    self.update_tensorboard(tb_dict)

            self.iteration += 1

            if debug: break
        
        self.epoch += 1

        for key in data_dict.keys(): data_dict[key] = np.mean(data_dict[key])
        data_dict['epoch'] = self.epoch-1
        data_dict['time'] = time_utils.tok(st_time, ms=False)
        
        return data_dict
    
    def evaluation_step(self, debug=False, valid_loader=None):
        st_time = time_utils.tik()

        if valid_loader is None:
            valid_loader = self.valid_loader

        progress = tqdm.tqdm(total=len(valid_loader), desc=self.progress_fmt%(self.epoch-1))

        for data in valid_loader:
            eval_dict = self.forward(data, training=False)    
            self.evaluator.add(**eval_dict)

            if self.param.RANK in [-1, 0]:
                progress.update(1)

            if debug: break
        
        return time_utils.tok(st_time, ms=False)
    
    def update_ema(self):
        if self.ema is not None:
            self.ema.update(self.model)
    
    def get_learning_rate(self, option='first'):
        lr_list = [group['lr'] for group in self.optimizer.param_groups]
        if option == 'max': return max(lr_list)
        elif option == 'min': return min(lr_list)
        elif option == 'first': return lr_list[0]
        else: return lr_list
    
    def update_tensorboard(self, tb_dict: dict):
        for key in tb_dict.keys():
            self.writer.add_scalar(key, tb_dict[key], self.iteration)
        return tb_dict

    def save_model(self, path):
        if self.param.use_ema: model = self.ema.get()
        else: model = de_parallel(self.model)
        save_model(model, path)