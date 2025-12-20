import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import islice
from tqdm import tqdm
from torch.utils.data import DataLoader


class TrainingManagerBacic():
    def __init__(self, 
                 trainable_modules: list[nn.Module],
                 dataloader: DataLoader,
                 num_epochs: int,
                 save_every_n_epochs: int = None,
                 frozen_modules: list[nn.Module]=[],
                 log_interval: int = None,
                 valid_dataloader: DataLoader = None,
                 valid_every_n_epochs: int = None,
                 n_batches_valid: int = None,
                 info_path: str = None
                 ):
        
        # init
        self.dataloader = dataloader
        self.dataset_len = len(self.dataloader)
        self.num_epochs = num_epochs
        self.current_epoch = 1
        self.total_step = self.dataset_len * self.num_epochs
        self.current_iter = 0
        self.epoch_loss = 0
        self.trainable_modules = trainable_modules
        self.log_interval = log_interval
        self.save_every_n_epochs = save_every_n_epochs

        self.valid_every_n_epochs = valid_every_n_epochs
        self._raw_valid_dataloader = valid_dataloader 

        self.log = {"config":{"num_epochs":0,"total_step":0,"log_interval":0,"save_every_n_epochs":0},
                    "epoch":0,
                    "epoch_loss":[] , 
                    "log":[], 
                    "val_log":[]
                    }
        self.info_path = info_path

        if info_path is None:
            self.write_info()
        else:
            self.load_info()

        # main progressbar
        self.progress_bar = tqdm(
            range(self.total_step), 
            desc=f"Epoch {self.current_epoch}/{self.num_epochs}",
            initial=self.current_iter
            )
            
        # epochs
        self.epochs = range(self.current_epoch, num_epochs + 1)


        # バッチ数の決定
        if valid_dataloader is not None:
            if n_batches_valid is None:
                self.n_batches_valid = len(valid_dataloader)
            else:
                self.n_batches_valid = n_batches_valid
        else:
            self.n_batches_valid = 0

        self.log_loss = 0.0
        self.valid_loss = 0.0

        for module in frozen_modules:
            if hasattr(module, 'eval') and callable(module.eval):
                module.eval()

        # set train mode
        self.train()

    def load_info(self):
        with open(self.info_path, "r") as f:
            info = json.load(f)

            self.log = info

            self.current_epoch = info["epoch"] + 1
            self.log["log"] = info["log"]
            self.log["val_log"] = info["val_log"]

            self.num_epochs = info["config"]["num_epochs"]
            self.total_step = info["config"]["total_step"]
            self.log_interval = info["config"]["log_interval"]
            self.save_every_n_epochs = info["config"]["save_every_n_epochs"]

            #---
            self.current_iter = (self.current_epoch - 1) * self.dataset_len

    def write_info(self):

        self.log["epoch"] = self.current_epoch

        self.log["config"]["num_epochs"] = self.num_epochs
        self.log["config"]["total_step"] = self.total_step
        self.log["config"]["log_interval"] = self.log_interval 
        self.log["config"]["save_every_n_epochs"] = self.save_every_n_epochs




    def save_info(self,output_dir:str):
        os.makedirs(output_dir,exist_ok=True)

        # temp
        avg_epoch_loss = self._get_avg_epoch_loss()
        self.log["epoch_loss"].append({str(self.current_epoch):avg_epoch_loss})

        epoch_info = self.log
        with open(os.path.join(output_dir, "trmn_info.json"), "w") as f:
            json.dump(epoch_info, f, indent=4) # indent=4で見やすく保存



    @property
    def valid_dataloader(self):
        if self._raw_valid_dataloader is None:
            return []

        return islice(self._raw_valid_dataloader, self.n_batches_valid)


    def train(self):
        for module in self.trainable_modules:
            if hasattr(module, 'train') and callable(module.train):
                module.train()

    def eval(self):   
        for module in self.trainable_modules:
            if hasattr(module, 'eval') and callable(module.eval):
                module.eval()

    def get_trainable_params(self) -> list[torch.Tensor]:
        trainable_params = []
        for module in self.trainable_modules:
            for param in module.parameters():
                if param.requires_grad:
                    trainable_params.append(param)
        return trainable_params


    def batch_step(self, loss, **kwargs) -> None:
        loss = loss.item() if hasattr(loss, 'item') else loss

        self.epoch_loss += loss
        self.current_iter += 1

        if self.log_interval is not None:
            self.log_loss += loss
            if self.current_iter % self.log_interval == 0:
                avg_loss = self.log_loss / self.log_interval
                self.log["log"].append({'step': self.current_iter, 'loss': avg_loss})
                self.log_loss = 0.0

        self.progress_bar.update(1)
        self.progress_bar.set_postfix(loss=f"{loss:.4f}", **kwargs)

    def _get_avg_epoch_loss(self):
        avg_epoch_loss = self.epoch_loss / self.dataset_len if self.dataset_len > 0 else 0
        return avg_epoch_loss

    def epoch_step(self, **kwargs) -> None:
        avg_epoch_loss = self._get_avg_epoch_loss()

        msg = f"Epoch {self.current_epoch}/{self.num_epochs} | epoch_loss={avg_epoch_loss:.4f}"

        if kwargs:
            extra_msg = [f"{k}={v}" for k, v in kwargs.items()]
            msg += ", " + ", ".join(extra_msg)

        tqdm.write(msg)  

        self.current_epoch += 1 
        self.epoch_loss = 0

        self.log["epoch"] = self.current_epoch
      
        if self.current_epoch <= self.num_epochs:
            self.progress_bar.set_description(f"Epoch {self.current_epoch}/{self.num_epochs}")

    def valid_step(self, loss):
        loss = loss.item() if hasattr(loss, 'item') else loss
        self.valid_loss += loss

    def valid_start(self):
        self.eval()
        torch.set_grad_enabled(False)

    def valid_end(self):
        if self.n_batches_valid > 0:
            avg_loss = self.valid_loss / self.n_batches_valid
        else:
            avg_loss = 0
        self.log["val_log"].append({'step': self.current_iter, 'loss': avg_loss})
        self.valid_loss = 0
        torch.set_grad_enabled(True)
        self.train()

    def is_savepoint(self) -> bool:
        if self.current_epoch > self.num_epochs: return False # 終了後はFalse
        if self.current_epoch == self.num_epochs:
            return True
        if self.save_every_n_epochs is not None and (self.current_epoch) % self.save_every_n_epochs == 0:
            return True
        return False

    def is_validpoint(self) -> bool:
        if self.current_epoch > self.num_epochs: return False
        if self.valid_every_n_epochs is not None: 
            if self.current_epoch == self.num_epochs:
                return True
            if (self.current_epoch) % self.valid_every_n_epochs == 0:
                return True
        return False
    
    


    def plot(self, name: str = None, output_dir = None) -> None:
        if self.log_interval is not None and len(self.log["log"]) > 0:
            steps = [item['step'] for item in self.log["log"]]
            losses = [item['loss'] for item in self.log["log"]]

            plt.figure(figsize=(10, 5))
            plt.plot(steps, losses, label='Training Loss')

            if len(self.log["val_log"]) > 0:
                v_steps = [item['step'] for item in self.log["val_log"]]
                v_losses = [item['loss'] for item in self.log["val_log"]]
                plt.plot(v_steps, v_losses, label='Validation Loss', marker='o', linestyle='--', color='orange')
            
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)

            name = "training_loss" if name is None else name
            if output_dir is None:
                output_path = f"{name}.png"
            else:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{name}.png")

            plt.savefig(output_path)
            plt.close()


