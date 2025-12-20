import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import islice
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from valid_manager import ValidManager

def get_trainable_params(*trainable_modules:nn.Module) -> list[torch.Tensor]:
        trainable_params = []
        for module in trainable_modules:
            for param in module.parameters():
                if param.requires_grad:
                    trainable_params.append(param)
        return trainable_params



class TrainingManager():
    def __init__(self, 
                 trainable_modules: list[nn.Module],
                 dataloader: DataLoader,
                 num_epochs: int,
                 save_every_n_epochs: int = None,
                 log_interval: int = None,
                 accelerator: Accelerator = None,
                 valid_manager:ValidManager = None,
                 checkpoint_dir: str = None,
                 frozen_modules: list[nn.Module]=[],
                 ):
        
        # init
        self._raw_dataloader = dataloader
        self.epoch_steps = len(self._raw_dataloader)
        self.num_epochs = num_epochs
        self.current_epoch = 1
        self.total_step = self.epoch_steps * self.num_epochs
        self.current_step = 0
        self.epoch_loss = 0
        self.trainable_modules = trainable_modules
        self.log_interval = log_interval
        self.save_every_n_epochs = save_every_n_epochs

        self.accelerator = accelerator

        self._save_info_path = None

        self.valid = valid_manager
        if self.valid is not None:
            self.valid.set_training_manager(self)

        self.log = {"config":{"num_epochs":0,"total_step":0,"log_interval":0,"save_every_n_epochs":0},
                    "epoch":0,
                    "step":0,
                    "epoch_loss":{} , 
                    "log":[], 
                    "val_log":[],
                    "lr_log":[],
                    }
        self.checkpoint_json_name = "trmn_info.json"

        self.checkpoint_dir = checkpoint_dir

        if self.checkpoint_dir is None:
            self.write_info()
        else:
            self.load_info()

        # main progressbar
        self.progress_bar = tqdm(
            range(self.total_step), 
            desc=f"Epoch {self.current_epoch}/{self.num_epochs}",
            initial=self.current_step
            )


        self.log_loss = 0.0

        for module in frozen_modules:
            if hasattr(module, 'eval') and callable(module.eval):
                module.eval()

        # set train mode
        self.train()

    @property
    def epochs(self):
        return range(self.current_epoch, self.num_epochs + 1)

    @property
    def dataloader(self):
        """現在の進捗に合わせて、済んだバッチをスキップしたDataLoaderを返す (Resume対応)"""
        steps_done_in_epoch = self.current_step % self.epoch_steps
        
        # 途中再開なら islice で先頭をスキップ
        if steps_done_in_epoch > 0:
            return islice(self._raw_dataloader, steps_done_in_epoch, None)
        return self._raw_dataloader

    def load_info(self):
        json_path = os.path.join(self.checkpoint_dir,self.checkpoint_json_name)
        with open(json_path, "r") as f:
            info = json.load(f)

            self.log = info

            self.num_epochs = self.log["config"]["num_epochs"]
            self.total_step = self.log["config"]["total_step"]
            self.log_interval = self.log["config"]["log_interval"]
            self.save_every_n_epochs = self.log["config"]["save_every_n_epochs"]
            self.current_step = self.log["step"]

            epoch = self.current_step // (self.epoch_steps) + 1

            self.current_epoch = epoch
        
        if self.accelerator is not None:
            self.accelerator.load_state(self.checkpoint_dir)
            
        

    def write_info(self):
        self.log["epoch"] = self.current_epoch
        self.log["step"] = self.current_step
        self.log["config"]["num_epochs"] = self.num_epochs
        self.log["config"]["total_step"] = self.total_step
        self.log["config"]["log_interval"] = self.log_interval 
        self.log["config"]["save_every_n_epochs"] = self.save_every_n_epochs


    def save_checkpoint(self,output_dir:str=None):
        if self._save_info_path is None:
            if output_dir is None:
                return
            else:
                os.makedirs(output_dir,exist_ok=True)
        
        self._save_info_path = self._save_info_path if self._save_info_path else os.path.join(output_dir, self.checkpoint_json_name)
        
        self.log["epoch_loss"][str(self.current_epoch)]=self._get_avg_epoch_loss()

        self.write_info()

        with open(self._save_info_path, "w") as f:
            json.dump(self.log, f, indent=4) # indent=4で見やすく保存

        if self.accelerator is not None:
            self.accelerator.save_state(output_dir)

    def train(self):
        for module in self.trainable_modules:
            if hasattr(module, 'train') and callable(module.train):
                module.train()

    def eval(self):   
        for module in self.trainable_modules:
            if hasattr(module, 'eval') and callable(module.eval):
                module.eval()



    def step_end(self, loss, **kwargs) -> None:
        loss = loss.item() if hasattr(loss, 'item') else loss

        self.epoch_loss += loss
        self.current_step += 1

        if self.log_interval is not None:
            self.log_loss += loss
            if self.current_step % self.log_interval == 0:
                avg_loss = self.log_loss / self.log_interval
                self.log["log"].append({'step': self.current_step, 'loss': avg_loss})
                self.log_loss = 0.0

        self.progress_bar.update(1)
        self.progress_bar.set_postfix(loss=f"{loss:.4f}", **kwargs)

    def _get_avg_epoch_loss(self):
        current_epoch_steps = self.current_step % self.epoch_steps
        if current_epoch_steps == 0:
            current_epoch_steps = self.epoch_steps
        
        avg_epoch_loss = self.epoch_loss / current_epoch_steps if current_epoch_steps > 0 else 0
        return avg_epoch_loss

    def epoch_end(self, **kwargs) -> None:
        avg_epoch_loss = self._get_avg_epoch_loss()

        msg = f"Epoch {self.current_epoch}/{self.num_epochs} | epoch_loss={avg_epoch_loss:.4f}"

        if kwargs:
            extra_msg = [f"{k}={v}" for k, v in kwargs.items()]
            msg += ", " + ", ".join(extra_msg)

        tqdm.write(msg)  

        self.current_epoch += 1 
        self.epoch_loss = 0
      
        if self.current_epoch <= self.num_epochs:
            self.progress_bar.set_description(f"Epoch {self.current_epoch}/{self.num_epochs}")

        

    def lr_log(self,lr=None):
        if lr is not None:
            self.log["lr_log"].append({"step":self.current_step,"lr":lr})


    def is_savepoint(self) -> bool:
        if self.current_epoch > self.num_epochs: return False # 終了後はFalse
        if self.current_epoch == self.num_epochs:
            return True
        if self.save_every_n_epochs is not None and (self.current_epoch) % self.save_every_n_epochs == 0:
            return True
        return False

    def is_validpoint(self) -> bool:
        if self.valid is None: return False
        if self.current_epoch > self.num_epochs: return False
        if self.valid.every_n_epochs is not None: 
            if self.current_epoch == self.num_epochs:
                return True
            if (self.current_epoch) % self.valid.every_n_epochs == 0:
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

            if len(self.log["lr_log"]) > 0:
                steps = [item['step'] for item in self.log["lr_log"]]
                lr = [item['lr'] for item in self.log["lr_log"]]
                plt.plot(steps, lr, label='lr', linestyle='--', color='yellow')
            
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


