import time
import random
from torch.utils.data import DataLoader, Dataset
from training_manager import TrainingManager
from valid_manager import ValidManager
from accelerate import Accelerator
import torch
import os

class MyDataset(Dataset):
    def __init__(self, repeat):
        self.dataset = [i for i in range(100)]
        self.repeat = repeat

    def __len__(self):
        return len(self.dataset) * self.repeat
    
    def __getitem__(self, idx):
        true_idx = idx % len(self.dataset)
        return self.dataset[true_idx]

def main():
    num_epochs = 5
    save_every_n_epochs = 1
    output_dir = "output"
    os.makedirs(output_dir,exist_ok=True)

    accelerator = Accelerator()
    
    batch_size = 1
    repeat = 1
    lr = 1e-1

    model = torch.nn.Sequential(torch.nn.Linear(10,10))
    
    dataset = MyDataset(repeat=repeat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    """
    optimizer = optimizer(
        params = get_trainable_params(model),
    )
    """

    model, dataloader,valid_dataloader = accelerator.prepare(
        model, dataloader, valid_dataloader
    )

    vm = ValidManager(
        valid_every_n_epochs=1,
        valid_dataloader=valid_dataloader,
        n_batches_valid=None,
    )

    tm = TrainingManager(
        trainable_modules=[model],
        dataloader=dataloader,
        num_epochs=num_epochs,
        save_every_n_epochs=save_every_n_epochs,
        log_interval=5,
        accelerator=accelerator,
        valid_manager=vm,
        checkpoint_dir=os.path.join(output_dir,"checkpoint"),
    )

    

    def forward_process(data):
        time.sleep(0.05) 
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
            tm.step_end(loss)


        if tm.is_validpoint():
            tm.valid.start()
            for data in tm.valid.dataloader:
                val_loss = random.random() * 10
                tm.valid.step_end(val_loss)
            tm.valid.end()


        if tm.is_savepoint():
            save_model()

        
        tm.lr_log(lr)
        tm.save_checkpoint()
        tm.plot(output_dir)
        tm.epoch_end(epoch_loss = f"{tm.get_epoch_loss():.4f}")


if __name__ == "__main__":
    main()