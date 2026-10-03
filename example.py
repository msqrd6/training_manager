import time
import random
import os
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

# 作成した最新のクラス群をインポート
from trmn.training_manager import TrainingManager
from trmn.config_mixin import ConfigMixin, register_config

class MyDataset(Dataset):
    def __init__(self, repeat):
        self.dataset = [i for i in range(100)]
        self.repeat = repeat

    def __len__(self):
        return len(self.dataset) * self.repeat
    
    def __getitem__(self, idx):
        true_idx = idx % len(self.dataset)
        return torch.randn(10)  # ダミー入力 (10次元)

# =========================================================
# 🌟 ConfigMixinと@register_configを活用したモデル定義
# =========================================================
class Model(torch.nn.Module, ConfigMixin):
    @register_config
    def __init__(self, hidden_dim=32):
        super().__init__()
        # use_ema=True(デフォルト)により、初期化完了後に自動で self.ema が作られます
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layer(x)

def main():
    num_epochs = 5
    output_dir = "trmn/output"
    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator()
    
    batch_size = 1
    repeat = 10
    lr = 1e-1

    # =========================================================
    # 🌟 モデルの初期化 (学習時なので use_ema=True を指定)
    # =========================================================
    model = Model(hidden_dim=32, use_ema=True)
    print(f"✨ 自動生成されたEMAモデル: {getattr(model, 'ema', None)}")
    
    dataset = MyDataset(repeat=repeat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
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
        time.sleep(0.001)
        
        # 実際の学習を模した順伝播とロス計算
        out = model(data)
        
        # エポックが進むごとにLossが下がるようにシミュレート
        fake_loss_val = (random.random() * 10) / current_epoch
        loss = out.mean() * 0.0 + fake_loss_val  # 勾配グラフを維持しつつダミーロス
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
            # =========================================================
            # 🌟 学習ステップの終わりにEMAを更新
            # =========================================================
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "ema"):
                # TrainingManager が管理している正確なステップ数を渡す
                unwrapped_model.ema.step(tm.step)
            
            # 💡 動的ロギング（lossやlrなど記録したいものを何でも渡すだけ）
            tm.step_end(decimals=3, loss=loss, learning_rate=lr)

        # 💡 エポック終了時の処理（自動で平均化・1行ログ出力）
        tm.epoch_end()

        # =========================================================
        # 🌟 保存処理
        # =========================================================
        if tm.is_savepoint():
            save_path = os.path.join(output_dir, f"epoch_{epoch}")
            unwrapped_model = accelerator.unwrap_model(model)
            
            # 1. モデルの重みと設定(config.json)を保存
            unwrapped_model.save_pretrained(save_path)
            
            # 2. EMAモデルの重みも保存
            if hasattr(unwrapped_model, "ema"):
                unwrapped_model.ema.save_pretrained(save_path, save_name="ema_model")
            
            print(f"✅ 保存完了: {save_path}")
        
        tm.plot()

if __name__ == "__main__":
    main()