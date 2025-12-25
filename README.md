# TrainingManager
PyTorchの学習ループを簡潔に記述するためのユーティリティクラスです。

## インストール

```bash
pip install git+https://github.com/msqrd6/training_manager.git
```

## 基本的な使い方
```python
from trmn import TrainingManager, get_trainable_params

# TrainingManagerの初期化
tm = TrainingManager(
    trainable_modules=[model],
    dataloader=train_dataloader,
    num_epochs=10,
    save_every_n_epochs=2,
    log_interval=100,
    checkpoint_dir='./checkpoints',  # チェックポイント保存先
)

# オプティマイザの初期化
optimizer = torch.optim.Adam(get_trainable_params(model), lr=1e-4)

# 学習ループ
for epoch in tm.epochs:
    for data in tm.dataloader:
        loss = model(data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tm.step_end(loss)  # 損失を記録
    
    if tm.is_savepoint():
        tm.save_checkpoint()  # チェックポイント保存
    
    tm.epoch_end()  # エポック終了処理

# 学習曲線をプロット
tm.plot(output_dir='./results')
```

## バリデーション付きの例

```python
from trmn import TrainingManager, ValidManager

# ValidManagerの初期化
valid_manager = ValidManager(
    valid_dataloader=valid_dataloader,
    valid_every_n_epochs=1,
    n_batches_valid=10  # 使用するバッチ数（省略時は全バッチ）
)

# TrainingManagerの初期化
tm = TrainingManager(
    trainable_modules=[model],
    dataloader=train_dataloader,
    num_epochs=10,
    valid_manager=valid_manager,
)

# 学習ループ
for epoch in tm.epochs:
    for data in tm.dataloader:
        loss = model(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tm.step_end(loss)
    
    # バリデーション
    if tm.is_validpoint():
        tm.valid.start()  # 評価モードに切り替え
        for data in tm.valid.dataloader:
            val_loss = model(data)
            tm.valid.step_end(val_loss)
        tm.valid.end()  # 学習モードに戻す
    
    tm.epoch_end()
```

## Accelerator対応

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

tm = TrainingManager(
    trainable_modules=[model],
    dataloader=train_dataloader,
    num_epochs=10,
    accelerator=accelerator,
    checkpoint_dir='./checkpoints',
)

# 学習ループは同じ
```

## チェックポイントからの再開

`checkpoint_dir`を指定すると、自動的にチェックポイントから再開されます：

```python
tm = TrainingManager(
    trainable_modules=[model],
    dataloader=train_dataloader,
    num_epochs=10,
    checkpoint_dir='./checkpoints',  # 既存のチェックポイントがあれば自動で読み込み
)
```
## API リファレンス

### TrainingManager

#### 初期化パラメータ

- `trainable_modules` (list[nn.Module]): 学習対象のモジュール
- `dataloader` (DataLoader): 学習用データローダー
- `num_epochs` (int): 学習エポック数
- `save_every_n_epochs` (int, optional): チェックポイント保存間隔
- `log_interval` (int, optional): 損失ログの記録間隔（バッチ単位）
- `accelerator` (Accelerator, optional): Acceleratorインスタンス
- `valid_manager` (ValidManager, optional): ValidManagerインスタンス
- `checkpoint_dir` (str, optional): チェックポイント保存先ディレクトリ
- `frozen_modules` (list[nn.Module], optional): 凍結するモジュール

#### 主要メソッド

- `step_end(loss, **kwargs)`: バッチ終了時に呼び出し、損失を記録
- `epoch_end(**kwargs)`: エポック終了時に呼び出し
- `save_checkpoint()`: チェックポイントを保存
- `is_savepoint()`: 保存タイミングかを判定
- `is_validpoint()`: バリデーションタイミングかを判定
- `plot(output_dir, file_name)`: 学習曲線をプロット
- `lr_log(lr)`: 学習率を記録
- `train()`: 学習モードに切り替え
- `eval()`: 評価モードに切り替え

#### プロパティ

- `epochs`: エポックのイテレータ
- `dataloader`: 学習用データローダー（再開時は途中から）

### ValidManager

#### 初期化パラメータ

- `valid_dataloader` (DataLoader, optional): バリデーション用データローダー
- `valid_every_n_epochs` (int, optional): バリデーション実行間隔
- `n_batches_valid` (int, optional): 使用するバッチ数

#### 主要メソッド

- `start()`: バリデーション開始（評価モードに切り替え）
- `step_end(loss)`: バリデーション損失を記録
- `end()`: バリデーション終了（学習モードに戻す）

#### プロパティ

- `dataloader`: バリデーション用データローダー

### get_trainable_params(*trainable_modules)

学習対象のパラメータを取得する関数。オプティマイザの初期化に使用します。

```python
optimizer = torch.optim.Adam(get_trainable_params(model), lr=1e-4)
```

## ライセンス

MIT License