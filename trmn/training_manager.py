import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import islice
from torch.utils.data import DataLoader
from accelerate import Accelerator
from pathlib import Path

from collections import defaultdict
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

def get_trainable_params(*trainable_modules:nn.Module) -> list[torch.Tensor]:
        trainable_params = []
        for module in trainable_modules:
            for param in module.parameters():
                if param.requires_grad:
                    trainable_params.append(param)
        return trainable_params

class MetricsPlotter:
    """TrainingStateの履歴(step_history)から汎用的でクリアなグラフを描画・保存するクラス"""
    
    def __init__(self, state, output_dir: str = "./plots"):
        self.state = state
        self.output_dir = output_dir
        
        # 出力先ディレクトリが存在しない場合は作成
        os.makedirs(self.output_dir, exist_ok=True)

        # 汎用的なカラーパレット (Matplotlib標準のtab10に準拠)
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    def _get_step_plot_data(self, data_dict: dict) -> tuple[list[int], list[float]]:
        """ステップ単位の辞書データから横軸(X)と縦軸(Y)を生成する専用メソッド"""
        xy = [(int(k), float(v)) for k, v in data_dict.items()]
        xy.sort(key=lambda x: x[0])
        return [p[0] for p in xy], [p[1] for p in xy]

    def _apply_k_formatting(self, ax, xlabel: str):
        """X軸がStepsのときだけ、数値を 'k' 表記(1000 -> 1k)にする内部メソッド"""
        if xlabel == "Steps":
            formatter = ticker.FuncFormatter(lambda x, pos: f"{x/1000:g}k" if x >= 1000 else f"{x:g}")
            ax.xaxis.set_major_formatter(formatter)

    def _apply_standard_styling(self, ax, title: str, xlabel: str):
        """グラフ全体に汎用的でクリアなスタイリングを適用する内部メソッド"""
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_title(title, fontsize=13, pad=15)

        ax.legend(loc='upper right', framealpha=0.9)
        self._apply_k_formatting(ax, xlabel)

    # =========================================================
    # プロット関数群
    # =========================================================
    def plot(self, filename: str = "all_metrics.png"):
        """全メトリクスの平滑化されたデータのみを1つのグラフにまとめて描画する"""
        if not self.state.step_history: 
            return
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for i, (metric_name, data_dict) in enumerate(self.state.step_history.items()):
            if not data_dict: 
                continue
            
            x_steps, y_values = self._get_step_plot_data(data_dict)
            color = self.colors[i % len(self.colors)]
            
            # 平滑化データのみを描画
            smoothed_values = pd.Series(y_values).ewm(alpha=0.05).mean()
            ax.plot(x_steps, smoothed_values, color=color, linewidth=2.0, label=f"{metric_name} (Smoothed)")
            
        self._apply_standard_styling(ax, "Training Metrics (Smoothed)", "Steps")
        
        filepath = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)


    def plot_individual(self, metric_name: str, filename: str = None):
        """指定した指標の生データと平滑化データを重ねて描画する"""
        if metric_name not in self.state.step_history or not self.state.step_history[metric_name]: 
            return
        
        x_steps, y_values = self._get_step_plot_data(self.state.step_history[metric_name])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 1. 生データ（薄い青、細い線）
        ax.plot(x_steps, y_values, color='tab:blue', alpha=0.3, linewidth=1.0, label=f'{metric_name} (Raw)')
        
        # 2. 平滑化データ（濃い青、太い線）
        smoothed_values = pd.Series(y_values).ewm(alpha=0.05).mean()
        ax.plot(x_steps, smoothed_values, color='tab:blue', linewidth=2.0, label=f'{metric_name} (Smoothed)')
        
        self._apply_standard_styling(ax, f"{metric_name.capitalize()}", "Steps")
        
        if filename is None: 
            filename = f"{metric_name}.png"
            
        filepath = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)


    def plot_all(self,filename="all.png"):
        """記録されているすべてのメトリクスについて、plot_individualを一括実行する"""
        if not self.state.step_history: 
            return
        
        self.plot(filename=filename)
        for metric_name in self.state.step_history.keys():
            self.plot_individual(metric_name)

class TrainingState:
    def __init__(self, **config_kwargs):
        self.config = config_kwargs
        self.step_log_interval = config_kwargs.get("step_log_interval", 10)
        
        self.epoch = 1
        self.step = 0
        self.history = defaultdict(dict)
        self.step_history = defaultdict(dict)
        self._running_state = defaultdict(lambda: {"sum": 0.0, "count": 0})

    def step_end(self, **kwargs):
        self.step += 1
        for key, value in kwargs.items():
            val = value.item() if hasattr(value, 'item') else value

            self._running_state[key]["sum"] += val
            self._running_state[key]["count"] += 1

            if self.step % self.step_log_interval == 0:
                self.step_history[key][self.step] = val

    def epoch_end(self):
        for key, data in self._running_state.items():
            avg_value = data["sum"] / data["count"] if data["count"] > 0 else 0.0
            self.history[f"epoch_{key}"][self.epoch] = avg_value
        
        self._running_state.clear()
        self.epoch += 1

    def get_current_state(self) -> Dict[str, float]:
        current_state = {}
        for key, data in self._running_state.items():
            if isinstance(data, dict):
                avg = data["sum"] / data["count"] if data["count"] > 0 else 0.0
                current_state[key] = avg
        return current_state

    def state_dict(self) -> Dict[str, Any]:
        running_state_dict = {
            "epoch": self.epoch,
            "step": self.step
        }
        for k, v in self._running_state.items():
            running_state_dict[k] = dict(v)

        return {
            "config": self.config,
            "running_state": running_state_dict,
            "history": {k: dict(v) for k, v in self.history.items()},
            "step_history": {k: dict(v) for k, v in self.step_history.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.config = state.get("config", {})
        
        self.history.clear()
        for k, v in state.get("history", {}).items():
            self.history[k].update(v)

        # 💡 修正箇所: JSONからstep_historyを復元する
        self.step_history.clear()
        for k, v in state.get("step_history", {}).items():
            self.step_history[k].update(v)

        running_state = state.get("running_state", {})
        self.epoch = running_state.get("epoch", 1)
        self.step = running_state.get("step", 0)
        
        self._running_state.clear()
        for k, v in running_state.items():
            if k not in ["epoch", "step"]:
                self._running_state[k].update(v)

class TrainingManager:
    """
    Acceleratorを用いた学習ループの進行と、
    チェックポイントのセーブ/ロードのみを担当する最小構成クラス。
    """
    def __init__(self,
                 trainable_modules: list[nn.Module],
                 dataloader: DataLoader,
                 num_epochs: int,
                 accelerator: Accelerator,
                 output_dir: str,
                 save_every_n_epochs: int = None,
                 step_log_interval: int = 10,
                 ):
        
        self.trainable_modules = trainable_modules
        self._raw_dataloader = dataloader
        self.num_epochs = num_epochs
        self.accelerator = accelerator
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plot_dir = self.output_dir / "plots"

        self.steps_per_epoch = len(self._raw_dataloader)
        self.total_step = self.steps_per_epoch * self.num_epochs

        self.training_state = TrainingState(
            num_epochs=self.num_epochs, 
            total_steps=self.total_step,
            steps_per_epoch=self.steps_per_epoch,
            step_log_interval = step_log_interval
        )

        self.checkpoint_json_name = "training_state.json"

        # チェックポイントの読み込み（あれば復元、なければ初期化）
        self._load_state()

        # モデルを学習モードへ切り替え
        self.train()

        # 進捗バーの初期化
        self.progress_bar = tqdm(
            range(self.total_step),
            desc=f"Epoch {self.training_state.epoch}/{self.num_epochs}",
            initial=self.training_state.step
        )

    @property
    def epoch(self):
        return self.training_state.epoch

    @property
    def step(self):
        return self.training_state.step

    @property
    def epochs(self):
        """途中再開を考慮したエポックのイテレータを返す"""
        return range(self.training_state.epoch, self.num_epochs + 1)

    @property
    def dataloader(self):
        """途中再開（Resume）時に、そのエポックですでに処理済みのバッチをスキップして返す"""
        steps_done_in_epoch = self.training_state.step % self.steps_per_epoch
        if steps_done_in_epoch > 0:
            return islice(self._raw_dataloader, steps_done_in_epoch, None)
        return self._raw_dataloader
    

    def _load_state(self):
        """チェックポイントが存在すれば読み込み、状態を復元する"""
        if self.checkpoint_dir is None:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        json_path = os.path.join(self.checkpoint_dir, self.checkpoint_json_name)

        if os.path.exists(json_path):
            # 1. 独自の進行状況を美しいJSON構造から復元
            with open(json_path, "r") as f:
                state = json.load(f)
                self.training_state.load_state_dict(state)

            # 2. Acceleratorが管理する重みやオプティマイザの状態を復元
            self.accelerator.load_state(self.checkpoint_dir)
            print(f"Loaded checkpoint: Epoch {self.training_state.epoch}, Step {self.training_state.step}")
        else:
            print("Initialized new checkpoint directory.")

    def is_savepoint(self) -> bool:
        if self.training_state.epoch > self.num_epochs: return False # 終了後はFalse
        if self.training_state.epoch == self.num_epochs:
            return True
        if self.save_every_n_epochs is not None and (self.training_state.epoch) % self.save_every_n_epochs == 0:
            return True
        return False


    def checkpoint(self):
        """現在のステップとエポック、およびモデルの状態を保存する"""
        if self.checkpoint_dir is None:
            return

        # 1. TrainingState が生成する完璧な辞書をそのまま保存
        json_path = os.path.join(self.checkpoint_dir, self.checkpoint_json_name)
        with open(json_path, "w") as f:
            json.dump(self.training_state.state_dict(), f, indent=4)

        # 2. Acceleratorが管理する重みやオプティマイザの状態を保存
        self.accelerator.save_state(self.checkpoint_dir)


    def train(self):
        for module in self.trainable_modules:
            if hasattr(module, 'train') and callable(module.train):
                module.train()

    def step_end(self, decimals: int = 3, **kwargs):
        """1ステップ（1バッチ）終了時の処理。kwargsで動的に受け取る"""
        
        # TrainingStateにはメトリクス(kwargs)だけを投げる
        # (引数 decimals は kwargs の中に含まれないので、ノイズとして記録されません)
        self.training_state.step_end(**kwargs)
        
        self.progress_bar.update(1)
        
        # プログレスバーに表示するための文字列辞書を作成
        postfix_data = {}
        for key, value in kwargs.items():
            # Tensor型の場合は数値に変換
            val = value.item() if hasattr(value, 'item') else value
            
            # 数値型であれば指定の桁数でフォーマット、それ以外の型(文字列など)はそのまま
            if isinstance(val, (float, int)):
                postfix_data[key] = f"{val:.{decimals}f}"
            else:
                postfix_data[key] = val
                
        # 辞書を展開(unpack)して tqdm の set_postfix に渡す
        self.progress_bar.set_postfix(**postfix_data)

    def epoch_end(self):
        """1エポック終了時の処理"""
        metrics = self.training_state.get_current_state()
        
        metrics_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        
        tqdm.write(f"Epoch {self.training_state.epoch}/{self.num_epochs} : {metrics_str}")
        
        self.training_state.epoch_end()
        
        # 5. 次のエポックへプログレスバーの表示を更新
        if self.training_state.epoch <= self.num_epochs:
            self.progress_bar.set_description(f"Epoch {self.training_state.epoch}/{self.num_epochs}")

    def plot(self, output_name: str = "all.png"):
        plotter = MetricsPlotter(self.training_state, self.plot_dir)
        plotter.plot_all(filename=output_name)

