import time
import random
import os
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from trmn import TrainingManager, ConfigMixin, register_config

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
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layer(x)

def main():
    num_epochs = 3
    output_dir = "trmn/_output"
    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator()
    
    batch_size = 4
    repeat = 5

    # 1. モデルとデータローダーの準備
    model = Model(hidden_dim=32)
    dataset = MyDataset(repeat=repeat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model, dataloader = accelerator.prepare(model, dataloader)

    # =========================================================
    # 💡 TrainingManagerの初期化
    # - ema=True: 自動でEMAを作成し、GPUへ配置
    # - 単一モデルは自動で 'model' フォルダに保存 ({"model": model} の辞書指定も可能)
    # - save_every_n_epochs=None: 常に最新1件を上書き保存
    #   (数値を指定すると checkpoints/epoch_001/model/... のように履歴保存)
    # =========================================================
    tm = TrainingManager(
        trainable_modules={"model":model},
        dataloader=dataloader,
        num_epochs=num_epochs,
        accelerator=accelerator,
        output_dir=output_dir,
        save_every_n_epochs=None,  # 常に最新の1件（checkpoints/model/）を上書き
        ema=True,                  # EMAモデルの自動管理を有効化
    )

    def forward_process(data, current_epoch):
        time.sleep(0.001)
        out = model(data)
        fake_loss_val = (random.random() * 10) / current_epoch
        loss = out.mean() * 0.0 + fake_loss_val
        return loss

    # =========================================================
    # 💡 トレーニングループ
    # =========================================================
    for epoch in tm.epochs:
        
        # 途中再開時は完了済みバッチを自動スキップ
        for data in tm.dataloader:
            # 💡 tm.accumulate() で勾配蓄積と autocast を自動適用
            with tm.accumulate():
                loss = forward_process(data, epoch)

            # --- 実際の学習処理 ---
            # tm.backward(loss)
            # if tm.sync_gradients: 
            #     optimizer.step()
            #     optimizer.zero_grad()
            
            # 💡 メトリクス記録 + プログレスバー更新 + EMAの自動ステップ更新
            tm.step_end(decimals=3, loss=loss)

            # 💡 (任意) 巨大データセット等でエポック途中にクラッシュ対策したい場合
            # if tm.step % 500 == 0:
            #     tm.save_training_state()

        # 💡 エポック終了処理
        # - ログ出力
        # - _training_state/ への再開用自動上書き保存（クラッシュ対策）
        # - checkpoints/ へのモデル重み自動保存（is_savepointを内部で自動判定）
        tm.epoch_end()

        # 💡 グラフ描画
        tm.plot()

    # =========================================================
    # 💡 学習後の推論テスト（通常モデルとEMAモデルの簡単ロード）
    # =========================================================
    ckpt_dir = tm.checkpoint_dir / "model"
    print("\n[INFO] 学習完了モデルのロードテスト:")
    
    # 通常モデルのロード
    loaded_model = Model.from_pretrained(str(ckpt_dir))
    print("[SUCCESS] 通常モデルのロード成功:", type(loaded_model))

    # EMAモデルのロード（同じフォルダから model_ema.safetensors を指定するだけ）
    loaded_ema = Model.from_pretrained(str(ckpt_dir), load_model_name="model_ema.safetensors")
    print("[SUCCESS] EMAモデルのロード成功:", type(loaded_ema))

if __name__ == "__main__":
    main()
