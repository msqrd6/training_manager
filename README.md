# TrainingManager

PyTorchの学習ループを簡潔に記述するためのユーティリティクラスです。プログレスバー表示、損失のロギング、バリデーション、定期的なチェックポイント保存などの機能を提供します。

## 特徴

- 学習の進捗をプログレスバーで視覚化
- 学習・検証損失の自動ロギングとプロット
- エポックごとのチェックポイント保存タイミングの管理
- 検証データローダーの部分的な使用に対応
- 複数モデルの学習・評価モード切り替えに対応

## インストール

### 方法1: ファイルをコピーして使う場合
リポジトリ内の `training_manager.py` をあなたのプロジェクトフォルダにコピーしてください。
その上で、必要なライブラリをインストールします:
```bash
pip install torch matplotlib tqdm
```

### 方法2: pipでインストールする場合
Git経由でライブラリとしてインストールすることも可能です:
```bash
pip install git+https://github.com/msqrd6/training_manager.git
```

## 使い方

### 基本的な使用例

```python
from torch.utils.data import DataLoader
from trmn import TrainingManager

# データローダーの準備
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# TrainingManagerの初期化
tm = TrainingManager(
    train_dataloader,
    num_epochs=10,
    save_every_n_epochs=2,
    log_interval=100,
    valid_every_n_epochs=1,
    valid_dataloader=valid_dataloader,
    training_models=[model]  # 学習するモデルをリストで渡す
)

# 学習ループ
for epoch in tm.epochs:
    tm.train_mode()
    
    for data in tm.dataloader:
        # 順伝播
        output = model(data)
        loss = criterion(output, target)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 損失を記録
        tm.batch_step(loss)
    
    # バリデーション
    if tm.is_validpoint():
        for data in tm.valid_dataloader:
            with torch.no_grad():
                output = model(data)
                val_loss = criterion(output, target)
                tm.valid_step(val_loss)
        tm.valid_end()
    
    # チェックポイント保存
    if tm.is_savepoint():
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
    
    # エポック終了処理
    tm.epoch_step()

# 学習曲線をプロット
tm.plot(name='training_curve', output_dir='./results')
```

## API リファレンス

### TrainingManager

#### 初期化パラメータ

- `dataloader` (DataLoader): 学習用データローダー
- `num_epochs` (int): 学習エポック数
- `save_every_n_epochs` (int, optional): チェックポイント保存間隔
- `log_interval` (int, optional): 損失ログの記録間隔（バッチ単位）
- `valid_every_n_epochs` (int, optional): バリデーション実行間隔
- `valid_dataloader` (DataLoader, optional): バリデーション用データローダー
- `n_batches_valid` (int, optional): バリデーションで使用するバッチ数（省略時は全バッチ）
- `training_models` (list, optional): 学習対象のモデルのリスト（複数指定可能）

#### 主要メソッド・プロパティ

##### `train_mode()`
登録された全モデルを学習モードに切り替えます。

##### `eval_mode()`
登録された全モデルを評価モードに切り替えます。

##### `epochs` (プロパティ)
エポック数のイテレータを返します。

##### `dataloader` (プロパティ)
学習用データローダーを返します。

##### `valid_dataloader` (プロパティ)
バリデーション用データローダー（必要に応じて制限付き）を返します。

##### `batch_step(loss, **kwargs)`
バッチごとの処理を行います。損失の記録とプログレスバーの更新を実行します。

- `loss`: 損失値（torch.Tensorまたはfloat）
- `**kwargs`: プログレスバーに追加表示する情報

##### `epoch_step(**kwargs)`
エポック終了時の処理を行います。エポック平均損失を表示し、内部カウンタを更新します。

- `**kwargs`: 追加で表示する情報

##### `valid_step(loss)`
バリデーションの各バッチで呼び出し、損失を記録します。

##### `valid_end()`
バリデーション終了時に呼び出し、平均損失を計算してログに記録します。

##### `is_savepoint()`
現在のエポックがチェックポイント保存のタイミングかを判定します。

##### `is_validpoint()`
現在のエポックがバリデーション実行のタイミングかを判定します。

##### `plot(name=None, output_dir=None)`
学習曲線とバリデーション曲線をプロットして保存します。

- `name` (str, optional): 保存ファイル名（デフォルト: "training_loss"）
- `output_dir` (str, optional): 保存先ディレクトリ

## サンプルコード

リポジトリにはダミーデータを使用したサンプルコード（`main()`関数）が含まれています。実行方法：

```bash
python training_manager.py
```

## ライセンス

[MIT License](LICENSE.md)

## 貢献

バグ報告や機能リクエストは、GitHubのIssueでお願いします。プルリクエストも歓迎します。

## 注意事項

- このクラスはPyTorchの学習ループを補助するものであり、学習ロジック自体は実装する必要があります
- `log_interval`を設定しない場合、詳細な損失ログは記録されません
- バリデーションデータローダーを指定しない場合、バリデーション機能は無効になります