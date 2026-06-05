import time
import random
import os
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

# 作成した最新のクラス群をインポート
from trmn.training_manager import TrainingManager, MetricsPlotter

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
    output_dir = "trmn/output"
    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator()
    
    batch_size = 1
    repeat = 1
    lr = 1e-1

    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    
    dataset = MyDataset(repeat=repeat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Validationが不要になったため、modelとdataloaderのみをprepare
    model, dataloader = accelerator.prepare(model, dataloader)

    # 💡 新しいManagerの初期化
    tm = TrainingManager(
        trainable_modules=[model],
        dataloader=dataloader,
        num_epochs=num_epochs,
        accelerator=accelerator,
        output_dir=output_dir,
        save_every_n_epochs=1,
    )

    def forward_process(data, current_epoch):
        time.sleep(0.01) # テスト実行が早くなるよう少し短縮しています
        # エポックが進むごとにLossが下がるようにシミュレート
        loss = (random.random() * 10) / current_epoch
        return loss
    
    # =========================================================
    # 💡 大幅にシンプルになったトレーニングループ
    # =========================================================
    for epoch in tm.epochs:
        
        # 途中再開時は済んだバッチを自動スキップしてくれる tm.dataloader を使用
        for data in tm.dataloader:
            loss = forward_process(data, epoch)
            
            # --- 実際の学習処理 ---
            # accelerator.backward(loss)
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()
            
            # 💡 動的ロギング（lossやlrなど記録したいものを何でも渡すだけ）
            # プログレスバーには小数点以下3桁(decimals=3)で表示されます
            tm.step_end(decimals=3, loss=loss, learning_rate=lr)

        # 💡 エポック終了時の処理（自動で平均化・1行ログ出力が行われます）
        tm.epoch_end()

        if tm.is_savepoint():
            # save_model(model)
            pass
        
        # 💡 チェックポイントの保存
        tm.checkpoint()
        tm.plot()


if __name__ == "__main__":
    main()