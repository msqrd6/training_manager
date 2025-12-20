from __future__ import annotations 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import islice
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import TYPE_CHECKING

if TYPE_CHECKING:#型チェックに使用
    from training_manager import TrainingManager


class ValidManager():
    def __init__(self,valid_dataloader:DataLoader=None,valid_every_n_epochs:int=None,n_batches_valid:int=None):
        super().__init__()
        self.tm = None
        self.every_n_epochs = valid_every_n_epochs
        self.loss = 0.0

        # バッチ数の決定
        if valid_dataloader is not None:
            if n_batches_valid is None:
                self.n_batches_valid = len(valid_dataloader)
            else:
                self.n_batches_valid = n_batches_valid
        else:
            self.n_batches_valid = 0

        self._dataloader = valid_dataloader

    def set_training_manager(self,tm:TrainingManager):
        self.tm = tm

    @property
    def dataloader(self):
        if self._dataloader is None:
            return []

        return islice(self._dataloader, self.n_batches_valid)
    
    def step_end(self, loss):
        loss = loss.item() if hasattr(loss, 'item') else loss
        self.loss += loss

    def start(self):
        self.tm.eval()
        torch.set_grad_enabled(False)

    def end(self):
        if self.n_batches_valid > 0:
            
            avg_loss = self.loss / self.n_batches_valid
        else:
            avg_loss = 0

        self.tm.log["val_log"].append({'step': self.tm.current_step, 'loss': avg_loss})
        self.loss = 0
        torch.set_grad_enabled(True)
        self.tm.train()