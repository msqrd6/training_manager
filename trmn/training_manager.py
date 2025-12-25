import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import islice
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from trmn.valid_manager import ValidManager
import pandas as pd

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
        self.steps_per_epoch = len(self._raw_dataloader)
        self.num_epochs = num_epochs
        self.current_epoch = 1
        self.total_step = self.steps_per_epoch * self.num_epochs
        self.current_step = 0
        self.trainable_modules = trainable_modules
        self.log_interval = log_interval
        self.save_every_n_epochs = save_every_n_epochs

        self.accelerator = accelerator

        self.valid = valid_manager
        if self.valid is not None:
            self.valid.set_training_manager(self)

        self.log = {"config":{"num_epochs":0,"total_step":0,"log_interval":0,"save_every_n_epochs":0},
                    "epoch":0,
                    "step":0,
                    "epoch_loss":{} , 
                    "loss_log":{}, 
                    "val_log":{},
                    "lr_log":{},
                    }
        
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_json_name = "training_state.json"

        if self.checkpoint_dir is not None:
            self._checkpoint_json_path = os.path.join(self.checkpoint_dir, self.checkpoint_json_name)

            if os.path.exists(self._checkpoint_json_path):
                self._load_state()
                print("load checkpoint dir")
            else:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                self.log["config"]["num_epochs"] = self.num_epochs
                self.log["config"]["total_step"] = self.total_step
                self.log["config"]["log_interval"] = self.log_interval 
                self.log["config"]["save_every_n_epochs"] = self.save_every_n_epochs

                with open(self._checkpoint_json_path, "w") as f:
                    json.dump(self.log, f, indent=4)

                if self.accelerator is not None:
                    self.accelerator.save_state(self.checkpoint_dir)
                print("init checkpoint dir")
            

        self._show_training_conig()
        

        self.log_interval_loss_sum = 0.0
        self.epoch_loss_sum = 0

        for module in frozen_modules:
            if hasattr(module, 'eval') and callable(module.eval):
                module.eval()

        # main progressbar
        self.progress_bar = tqdm(
            range(self.total_step), 
            desc=f"Epoch {self.current_epoch}/{self.num_epochs}",
            initial=self.current_step
            )

        # set train mode
        self.train()

        
    def _show_training_conig(self):
        print("---------------training config-----------------")
        print(f"num_epochs:{self.num_epochs}")
        print(f"total_steps:{self.total_step}")
        print(f"log_interval:{False if self.log_interval is None else self.log_interval}")
        print(f"use_checkpoint:{False if self.checkpoint_dir is None else True}")
        print(f"use_accelerator:{False if self.accelerator is None else True}")
        print("-----------------------------------------------")
        
    @property
    def epochs(self):
        return range(self.current_epoch, self.num_epochs + 1)

    @property
    def dataloader(self):
        """現在の進捗に合わせて、済んだバッチをスキップしたDataLoaderを返す (Resume対応)"""
        steps_done_in_epoch = self.current_step % self.steps_per_epoch
        
        # 途中再開なら islice で先頭をスキップ
        if steps_done_in_epoch > 0:
            return islice(self._raw_dataloader, steps_done_in_epoch, None)
        return self._raw_dataloader

    def _load_state(self):
        json_path = os.path.join(self.checkpoint_dir,self.checkpoint_json_name)
        with open(json_path, "r") as f:
            info = json.load(f)

            self.log = info

            self.num_epochs = self.log["config"]["num_epochs"]
            self.total_step = self.log["config"]["total_step"]
            self.log_interval = self.log["config"]["log_interval"]
            self.save_every_n_epochs = self.log["config"]["save_every_n_epochs"]
            self.current_step = self.log["step"]

            self.current_epoch = self.current_step // (self.steps_per_epoch) + 1
        
        if self.accelerator is not None:
            self.accelerator.load_state(self.checkpoint_dir)
            
    
    def _get_plot_data(self,data:dict[str, float]) -> tuple[list[int], list[float]]:
        xy = [(int(k),v) for k, v in data.items()]
        xy.sort()
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        return x, y
    
    def train(self):
        for module in self.trainable_modules:
            if hasattr(module, 'train') and callable(module.train):
                module.train()

    def eval(self):   
        for module in self.trainable_modules:
            if hasattr(module, 'eval') and callable(module.eval):
                module.eval()


    def save_checkpoint(self):
        if self.checkpoint_dir is None:
            return
        
        self.log["epoch_loss"][str(self.current_epoch)]=self.get_epoch_loss()
        self.log["epoch"] = self.current_epoch
        self.log["step"] = self.current_step

        with open(self._checkpoint_json_path, "w") as f:
            json.dump(self.log, f, indent=4)

        if self.accelerator is not None:
            self.accelerator.save_state(self.checkpoint_dir)


    def step_end(self, loss, **kwargs) -> None:
        loss = loss.item() if hasattr(loss, 'item') else loss

        self.epoch_loss_sum += loss
        self.current_step += 1

        if self.log_interval is not None:
            self.log_interval_loss_sum += loss
            if self.current_step % self.log_interval == 0:
                avg_loss = self.log_interval_loss_sum / self.log_interval
                #self.log["log"].append({'step': self.current_step, 'loss': avg_loss})
                self.log["loss_log"][str(self.current_step)] = avg_loss
                self.log_interval_loss_sum = 0.0

        self.progress_bar.update(1)
        self.progress_bar.set_postfix(loss=f"{loss:.4f}", **kwargs)


    def get_epoch_loss(self):
        current_epoch_steps = self.current_step % self.steps_per_epoch
        if current_epoch_steps == 0:
            current_epoch_steps = self.steps_per_epoch
        
        avg_epoch_loss = self.epoch_loss_sum / current_epoch_steps if current_epoch_steps > 0 else 0
        return avg_epoch_loss

    def epoch_end(self, **kwargs) -> None:
        msg = f"Epoch {self.current_epoch}/{self.num_epochs}"

        if kwargs:
            extra_msg = [f"{k}={v}" for k, v in kwargs.items()]
            msg += ": " + " | ".join(extra_msg)

        tqdm.write(msg)  

        self.current_epoch += 1 
        self.epoch_loss_sum = 0
      
        if self.current_epoch <= self.num_epochs:
            self.progress_bar.set_description(f"Epoch {self.current_epoch}/{self.num_epochs}")

        
    def lr_log(self,lr=None):
        if lr is not None:
            self.log["lr_log"][str(self.current_step)] = lr


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


    def plot(self, output_dir = None, file_name: str = None) -> None:
        if self.log_interval is not None and len(self.log["loss_log"]) > 0:

            steps, losses = self._get_plot_data(self.log["loss_log"])
            smooth_losses = pd.Series(losses).ewm(alpha=0.1).mean()

            fig, ax1 = plt.subplots(figsize=(10, 5)) # ax1を作成

            # Lossの描画 (左軸)
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss', color='tab:blue')

            ax1.plot(steps, losses, color='tab:blue', alpha=0.3, label='Training Loss')
            ax1.plot(steps, smooth_losses, color='tab:blue', linewidth=2, label='Training Loss')

            #ax1.plot(steps, losses, label='Training Loss', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.grid(True)

            # Validation Lossも左軸でOK
            if len(self.log["val_log"]) > 0:
                v_steps, v_losses = self._get_plot_data(self.log["val_log"])
                ax1.plot(v_steps, v_losses, label='Validation Loss', marker='o', linestyle='--', color='orange')

            # LRの描画 (右軸: twinx)
            if len(self.log["lr_log"]) > 0:
                ax2 = ax1.twinx()  # 右軸を作成
                ax2.set_ylabel('Learning Rate', color='tab:red')
                
                lr_steps,lr_values = self._get_plot_data(self.log["lr_log"])
                
                ax2.plot(lr_steps, lr_values, label='Learning Rate', linestyle='--', color='tab:red', alpha=0.6)
                ax2.tick_params(axis='y', labelcolor='tab:red')
                
                # 凡例をまとめて表示するための工夫
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax1.legend(loc='upper right')

            plt.title('Training Metrics')

            file_name = "training_loss" if file_name is None else file_name
            if output_dir is None:
                output_path = f"{file_name}.png"
            else:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{file_name}.png")

            plt.savefig(output_path)
            plt.close()


