from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import time
import random
import matplotlib.pyplot as plt
import os

class TrainingManager():
    def __init__(self,dataloader:DataLoader,
                 num_epochs:int,
                 save_every_n_epochs:int = None,
                 log_interval:int = None,
                 valid_every_n_epochs:int = None
                 ):
        
        
        self.dataloader = dataloader
        self.dataset_len = len(self.dataloader)

        self.num_epochs = num_epochs
        self.save_every_n_epochs = save_every_n_epochs

        self.valid_every_n_epochs = valid_every_n_epochs


        self.total_step = self.dataset_len*self.num_epochs
        self.current_iter = 0
        self.current_epoch = 1
        self.epoch_loss = 0

        #main progressbar
        self.progress_bar = tqdm(range(self.total_step),desc=f"Epoch {1}/{num_epochs}")

        # 
        self.epochs = range(1, num_epochs + 1)

        #log
        self.log = []
        self.log_interval = log_interval
        self.log_loss = 0.0

        self.val_log = []
        

    def batch_step(self,loss,**kwargs)->None:
        loss = loss.item() if hasattr(loss, 'item') else loss

        self.epoch_loss += loss
        self.current_iter += 1

        if self.log_interval is not None:
            self.log_loss += loss
            if self.current_iter % self.log_interval==0:
                avg_loss = self.log_loss/self.log_interval
                self.log.append({'step':self.current_iter,'loss':avg_loss})
                self.log_loss = 0.0


        self.progress_bar.update(1)
        self.progress_bar.set_postfix(loss=f"{loss:.4f}",**kwargs)

        
    def epoch_step(self,**kwargs)->None:
        avg_epoch_loss = self.epoch_loss/self.dataset_len

        msg = f"Epoch {self.current_epoch}/{self.num_epochs} | epoch_loss={avg_epoch_loss:.4f}"

        if kwargs:
            extra_msg = [f"{k}={v}" for k, v in kwargs.items()]
            msg += ", " + ", ".join(extra_msg)

        tqdm.write(msg)

        self.current_epoch += 1 
        self.epoch_loss = 0
      
        if self.current_epoch <= self.num_epochs:
            self.progress_bar.set_description(f"Epoch {self.current_epoch}/{self.num_epochs}")

    
    
    def is_savepoint(self)->bool:
        if self.current_epoch == self.num_epochs:
            return True
        
        if self.save_every_n_epochs is not None and (self.current_epoch) % self.save_every_n_epochs == 0:
            return True
            
        return False
    

    def is_validpoint(self)->bool:
        if self.valid_every_n_epochs is not None: 
            if self.current_epoch == self.num_epochs:
                return True
            
            if (self.current_epoch) % self.valid_every_n_epochs == 0:
                return True
            
        return False
    

    def record_valid_loss(self,val_loss)->None:
        self.val_log.append({'step':self.current_iter,'loss':val_loss})
        

    

    def plot_loss_curve(self,name:str=None,output_dir=None)->None:
        if self.log_interval is not None:
            steps = [item['step'] for item in self.log]
            losses = [item['loss'] for item in self.log]

            # プロット
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

            # save
            name = "training_loss" if name is None else name
            if output_dir is None:
                output_path = f"{name}.png"
            else:
                os.makedirs(output_dir,exist_ok=True)
                output_path = os.path.join(output_dir,f"{name}.png")

            plt.savefig(output_path)
            plt.close()

    
    
class Dataset(Dataset):
        def  __init__(self,repeat):
            self.dataset = [i for i in range(10)]
            self.repeat=repeat

        def __len__(self):
            return len(self.dataset)*self.repeat
        
        def __getitem__(self,idx):
            true_idx = idx%len(self.dataset)
            return self.dataset[true_idx]
    


def main():
    num_epochs = 20
    save_every_n_epochs = 10

    batch_size = 2
    repeat = 10
    dataset = Dataset(repeat=repeat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    tm = TrainingManager(dataloader,num_epochs,save_every_n_epochs,log_interval=10,valid_every_n_epochs=1)

    val_loss = random.randint(0,10)
    tm.record_valid_loss(val_loss)
    
    for epoch in tm.epochs:
        for data in tm.dataloader:
            time.sleep(0.001)
            loss = random.randint(0,10)
            tm.batch_step(loss)
            
        if tm.is_savepoint():
            pass

        if tm.is_validpoint():
            val_loss = random.randint(0,10)
            tm.record_valid_loss(val_loss)

        tm.epoch_step()

    tm.plot_loss_curve()


if __name__=="__main__":
    main()
        
