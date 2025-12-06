import time
import random
from torch.utils.data import DataLoader, Dataset
from trmn.training_manager import TrainingManager


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

    """
    optimizer = optimizer(
        params = tm.get_trainable_params(),
    )
    """
    
    tm.valid_start()
    for data in tm.valid_dataloader:
        val_loss = random.random() * 10
        tm.valid_step(val_loss)
    tm.valid_end()

    def forward_process(data):
        time.sleep(0.01) 
        loss = random.random() * 10
        return loss
    
    def save_model():
        return
    

    # training roop
    for epoch in tm.epochs:
        for data in tm.dataloader:
            loss = forward_process(data)

            # loss.backword()
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()
            
            tm.batch_step(loss)

        if tm.is_validpoint():
            tm.valid_start()
            for data in tm.valid_dataloader:
                val_loss = random.random() * 10
                tm.valid_step(val_loss)
            tm.valid_end()

        if tm.is_savepoint():
            save_model()
        
        tm.plot(tm.current_epoch)
        tm.epoch_step()

    

if __name__ == "__main__":
    main()