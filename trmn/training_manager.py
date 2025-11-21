import os
import time
import random
import matplotlib.pyplot as plt
from itertools import islice
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class TrainingManager():
    def __init__(self, dataloader: DataLoader,
                 num_epochs: int,
                 save_every_n_epochs: int = None,
                 log_interval: int = None,
                 valid_every_n_epochs: int = None,
                 valid_dataloader: DataLoader = None,
                 n_batches_valid: int = None,
                 training_models: list = None,
                 ):
        
        self.dataloader = dataloader
        try:
            self.dataset_len = len(self.dataloader)
        except:
            self.dataset_len = 0

        self.num_epochs = num_epochs
        self.save_every_n_epochs = save_every_n_epochs

        self.training_models = training_models

        self.total_step = self.dataset_len * self.num_epochs
        self.current_iter = 0
        self.current_epoch = 1
        self.epoch_loss = 0

        # main progressbar
        self.progress_bar = tqdm(range(self.total_step), desc=f"Epoch {1}/{num_epochs}")

        # epoch iter
        self.epochs = range(1, num_epochs + 1)

        # log
        self.log = []
        self.log_interval = log_interval
        self.log_loss = 0.0

        # valid
        self.valid_every_n_epochs = valid_every_n_epochs
        self._raw_valid_dataloader = valid_dataloader 
        
        # バッチ数の決定
        if valid_dataloader is not None:
            if n_batches_valid is None:
                self.n_batches_valid = len(valid_dataloader)
            else:
                self.n_batches_valid = n_batches_valid
        else:
            self.n_batches_valid = 0
            
        self.valid_loss = 0.0
        self.val_log = []

    def training_mode(self):
        for model in self.training_models:
            if hasattr(model, 'train') and callable(model.train):
                model.train()

    def eval_mode(self):
        for model in self.training_models:
            if hasattr(model, 'eval') and callable(model.eval):
                model.eval()

    def get_epochs(self):
        return self.epochs
    
    def get_dataloader(self):
        return self.dataloader
    
    def get_valid_dataloader(self):
        if self._raw_valid_dataloader is None:
            return []

        return islice(self._raw_valid_dataloader, self.n_batches_valid)


    def batch_step(self, loss, **kwargs) -> None:
        loss = loss.item() if hasattr(loss, 'item') else loss

        self.epoch_loss += loss
        self.current_iter += 1

        if self.log_interval is not None:
            self.log_loss += loss
            if self.current_iter % self.log_interval == 0:
                avg_loss = self.log_loss / self.log_interval
                self.log.append({'step': self.current_iter, 'loss': avg_loss})
                self.log_loss = 0.0

        self.progress_bar.update(1)
        self.progress_bar.set_postfix(loss=f"{loss:.4f}", **kwargs)

    def epoch_step(self, **kwargs) -> None:
        avg_epoch_loss = self.epoch_loss / self.dataset_len if self.dataset_len > 0 else 0

        msg = f"Epoch {self.current_epoch}/{self.num_epochs} | epoch_loss={avg_epoch_loss:.4f}"

        if kwargs:
            extra_msg = [f"{k}={v}" for k, v in kwargs.items()]
            msg += ", " + ", ".join(extra_msg)

        tqdm.write(msg) 

        self.current_epoch += 1 
        self.epoch_loss = 0
      
        if self.current_epoch <= self.num_epochs:
            self.progress_bar.set_description(f"Epoch {self.current_epoch}/{self.num_epochs}")

    def valid_step(self, loss):
        loss = loss.item() if hasattr(loss, 'item') else loss
        self.valid_loss += loss

    def valid_end(self):
        if self.n_batches_valid > 0:
            avg_loss = self.valid_loss / self.n_batches_valid
        else:
            avg_loss = 0
        self.val_log.append({'step': self.current_iter, 'loss': avg_loss})
        self.valid_loss = 0

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
        if self.log_interval is not None and len(self.log) > 0:
            steps = [item['step'] for item in self.log]
            losses = [item['loss'] for item in self.log]

            plt.figure(figsize=(10, 5))
            plt.plot(steps, losses, label='Training Loss')

            if len(self.val_log) > 0:
                v_steps = [item['step'] for item in self.val_log]
                v_losses = [item['loss'] for item in self.val_log]
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

class MyDataset(Dataset):
    def __init__(self, repeat):
        self.dataset = [i for i in range(10)]
        self.repeat = repeat

    def __len__(self):
        return len(self.dataset) * self.repeat
    
    def __getitem__(self, idx):
        true_idx = idx % len(self.dataset)
        return self.dataset[true_idx]

def main():
    num_epochs = 5
    save_every_n_epochs = 2
    
    batch_size = 2
    repeat = 5
    
    dataset = MyDataset(repeat=repeat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    tm = TrainingManager(
        dataloader,
        num_epochs,
        save_every_n_epochs,
        log_interval=5,
        valid_every_n_epochs=1,
        valid_dataloader=valid_dataloader,
        n_batches_valid=3 
    )

    tm.eval_mode()
    for data in tm.get_valid_dataloader():
        val_loss = random.random() * 10
        tm.valid_step(val_loss)
    tm.valid_end()
    
    # --- Training Loop ---
    for epoch in tm.get_epochs():
        tm.training_mode() 
        for data in tm.get_dataloader():
            time.sleep(0.01) 
            loss = random.random() * 10
            
            # ここに実際の model(data) や optimizer.step() が入る
            
            tm.batch_step(loss)

        if tm.is_validpoint():
            tm.eval_mode()
            for data in tm.get_valid_dataloader():
                val_loss = random.random() * 10
                tm.valid_step(val_loss)
            tm.valid_end() 

        if tm.is_savepoint():
            pass
        
        tm.plot(tm.current_epoch)
        tm.epoch_step()

    

if __name__ == "__main__":
    main()
