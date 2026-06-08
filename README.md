# TrainingManager
PyTorchの学習ループ（Accelerator対応）を簡潔に記述し、進行管理やチェックポイントの自動保存・復元、メトリクスのプロットを行うためのユーティリティクラスです。

## インストール

```bash
pip install git+https://github.com/msqrd6/training_manager.git
```

## 基本的な使い方

`TrainingManager` は、Hugging Face の `accelerate` に対応しており、以下のように学習ループを非常にシンプルに実装できます。

```python
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from trmn import TrainingManager, get_trainable_params

accelerator = Accelerator()

# モデル、データローダー、オプティマイザの初期化
model = MyModel()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(get_trainable_params(model), lr=1e-4)

# Acceleratorによる準備
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# TrainingManagerの初期化
tm = TrainingManager(
    trainable_modules=[model],
    dataloader=dataloader,
    num_epochs=10,
    accelerator=accelerator,
    output_dir='./output',       # チェックポイントやプロットの保存先
    save_every_n_epochs=2,       # チェックポイントの保存間隔 (省略時は最終エポックのみ)
    logs_per_epoch=10,        # 損失ログの記録ステップ間隔
)

# 学習ループ
for epoch in tm.epochs:
    # 途中再開時は、すでに処理されたバッチを自動スキップしてくれる tm.dataloader を使用します
    for data in tm.dataloader:
        loss = model(data)
        
        # 実際の学習処理
        # accelerator.backward(loss)
        # optimizer.step()
        # optimizer.zero_grad()
        
        # 動的なロギング (任意のキーで記録でき、プログレスバーに decimals 桁で表示されます)
        tm.step_end(decimals=3, loss=loss, lr=1e-4)
    
    # エポック終了時の処理 (平均値の算出とログ出力)
    tm.epoch_end()
    
    # 保存タイミングかを判定し、チェックポイントを保存
    if tm.is_savepoint():
        tm.checkpoint()
        
    # 学習曲線のプロット (output_dir/plots/all.png などに保存されます)
    tm.plot()
```

## チェックポイントからの自動再開

`output_dir` 内にチェックポイントが存在する場合、`TrainingManager` の初期化時に自動的に状態（エポック、ステップ、学習履歴、および Accelerator の状態）が復元され、途中から学習を再開できます。

```python
# output_dir を指定するだけで、既存のチェックポイントがあれば自動で読み込まれます
tm = TrainingManager(
    trainable_modules=[model],
    dataloader=dataloader,
    num_epochs=10,
    accelerator=accelerator,
    output_dir='./output',
)
```

## API リファレンス

### `TrainingManager`

#### 初期化パラメータ

- `trainable_modules` (`list[nn.Module]`): 学習対象のモジュール（パラメータ抽出や学習モードの切り替えに使用します）
- `dataloader` (`DataLoader`): 学習用データローダー
- `num_epochs` (`int`): 総学習エポック数
- `accelerator` (`Accelerator`): `accelerate.Accelerator` のインスタンス
- `output_dir` (`str`): チェックポイントやグラフの出力先ディレクトリ
- `save_every_n_epochs` (`int`, optional): チェックポイントを保存するエポック間隔
- `step_log_interval` (`int`, optional): 損失等のステップログを記録するステップ間隔（デフォルト: 10）

#### 主要メソッド

- `step_end(decimals=3, **kwargs)`: バッチ終了時に呼び出し、任意のメトリクス（例: `loss=loss`）を記録しプログレスバーに表示します。
- `epoch_end()`: エポック終了時に呼び出し、各メトリクスのエポック平均の算出、ログ出力、および進行状況の更新を行います。
- `checkpoint()`: チェックポイントを `output_dir/checkpoints/` ディレクトリに保存します（モデルやオプティマイザの重み、および学習状態を保存）。
- `is_savepoint() -> bool`: 現在のエポックが保存タイミング（`save_every_n_epochs` の倍数、または最終エポック）であるかを判定します。
- `plot(output_name="all.png")`: 記録された全メトリクスの推移を平滑化してグラフとして描画し、`output_dir/plots/` に保存します。

#### プロパティ

- `epoch` (`int`): 現在のエポック数
- `step` (`int`): これまでに実行された総ステップ数
- `epochs` (`range`): 途中再開を考慮した、残りのエポックのレンジ
- `dataloader` (`iterator`): 途中再開時に、そのエポック内ですでに処理されたバッチを自動的にスキップしたデータローダーのイテレータ

### `get_trainable_params(*trainable_modules)`

指定したモジュール群から `requires_grad=True` のパラメータを抽出するヘルパー関数です。オプティマイザの初期化に使用します。

```python
optimizer = torch.optim.Adam(get_trainable_params(model), lr=1e-4)
```

## ライセンス

MIT License