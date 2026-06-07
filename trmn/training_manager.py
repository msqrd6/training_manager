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



class TrainingState:
    def __init__(self, **config_kwargs):
        self.config = config_kwargs
        self.step_log_interval = config_kwargs.get("step_log_interval", 10)
        
        self.epoch = 1
        self.step = 0
        self.epoch_history = defaultdict(dict)
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
            self.epoch_history[f"epoch_{key}"][self.epoch] = avg_value
        
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
            "epoch_history": {k: dict(v) for k, v in self.epoch_history.items()},
            "step_history": {k: dict(v) for k, v in self.step_history.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.config = state.get("config", {})
        
        self.epoch_history.clear()
        for k, v in state.get("history", {}).items():
            self.epoch_history[k].update(v)

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

class Plotter:
    """TrainingStateの履歴(step_history)から汎用的でクリアなグラフを描画・保存するクラス"""
    def __init__(self, state, output_dir: str = "./plots", **custom_styles):
        self.state = state
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # =========================================================
        # 💡 ここでデザインを一元管理！ (外から上書き可能)
        # =========================================================
        self.style = {
            "title" : "Training Loss",
            "xlabel": "steps",

            # 色
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'],
            
            # 線の設定
            'raw_linewidth': 1.0,      # 生データの線の太さ
            'raw_alpha': 0.3,          # 生データの透明度
            'smooth_linewidth': 1.2,   # 平滑化データの線の太さ
            'smooth_factor': 0.1,      # 平滑化の強さ (0.1にすると少しトレンドに敏感になる)
            
            # 全体の設定
            'grid_style': '--',        # グリッドの線種
            'grid_alpha': 0.8,         # グリッドの透明度
            'title_size': 12,          # タイトルの文字サイズ
            'label_size': 10,          # 軸ラベル(X軸/Y軸の名前)の文字サイズ
            
            #軸の目盛り(Tick)の設定
            'tick_label_size': 9,      # 目盛りの数字のサイズ
            'tick_color': "#181818",   # 目盛りの色
            
            # 枠線（Spine）の設定
            'spine_width': 0.4,        # 枠線の太さ
            'spine_color': "#313131"
        }
        
        self.style.update(custom_styles)

    # =========================================================
    # スタイル適用メソッド
    # =========================================================
    def _plot_raw_data(self, ax, x, y, label, color='tab:blue'):
        ax.plot(x, y, color=color, 
                alpha=self.style['raw_alpha'], 
                linewidth=self.style['raw_linewidth'])

    def _plot_smoothed_data(self, ax, x, y, label, color='tab:blue'):
        smoothed_values = pd.Series(y).ewm(alpha=self.style['smooth_factor']).mean()
        ax.plot(x, smoothed_values, color=color, 
                linewidth=self.style['smooth_linewidth'], 
                label=label)

    def _apply_k_formatting(self, ax, xlabel: str):
        formatter = ticker.FuncFormatter(lambda x, pos: f"{x/1000:g}k" if x >= 1000 else f"{x:g}")
        ax.xaxis.set_major_formatter(formatter)

    def _apply_standard_styling(self, ax, title: str, xlabel: str):
        ax.grid(True, linestyle=self.style['grid_style'], alpha=self.style['grid_alpha'])
        ax.set_axisbelow(True)
        
        ax.set_xlabel(xlabel, fontsize=self.style['label_size'])
        ax.set_ylabel("loss", fontsize=self.style['label_size'])
        ax.set_title(title, fontsize=self.style['title_size'])
        ax.legend(loc='upper right', framealpha=0.9)
        
        for spine in ax.spines.values():
            spine.set_linewidth(self.style['spine_width'])
            spine.set_color(self.style['spine_color'])

        # 💡 追加: 目盛り(Tick)のスタイルを一括適用
        ax.tick_params(
            axis='both', 
            labelsize=self.style['tick_label_size'], 
            colors=self.style['tick_color']
        )

        self._apply_k_formatting(ax, xlabel)

    # =========================================================
    # データ取得ヘルパー
    # =========================================================
    def _get_step_plot_data(self, data_dict: dict) -> tuple[list[int], list[float]]:
        xy = [(int(k), float(v)) for k, v in data_dict.items()]
        xy.sort(key=lambda x: x[0])
        return [p[0] for p in xy], [p[1] for p in xy]

    # =========================================================
    # プロット実行関数群
    # =========================================================
    def plot(self, filename: str = "all_metrics.png"):
        if not self.state.step_history: return
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for i, (metric_name, data_dict) in enumerate(self.state.step_history.items()):
            if not data_dict: continue
            x_steps, y_values = self._get_step_plot_data(data_dict)
            color = self.style['colors'][i % len(self.style['colors'])]
            
            self._plot_smoothed_data(ax, x_steps, y_values, label=metric_name, color=color)
            
        self._apply_standard_styling(ax, self.style['title'], self.style['xlabel'])
        
        filepath = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_individual(self, metric_name: str, filename: str = None):
        if metric_name not in self.state.step_history or not self.state.step_history[metric_name]: return
        x_steps, y_values = self._get_step_plot_data(self.state.step_history[metric_name])
        fig, ax = plt.subplots(figsize=(10, 5))
        
        self._plot_raw_data(ax, x_steps, y_values, label=metric_name, color='tab:blue')
        self._plot_smoothed_data(ax, x_steps, y_values, label=metric_name, color='tab:blue')
        
        self._apply_standard_styling(ax,self.style['title'], self.style['xlabel'])
        ax.set_ylabel(metric_name.capitalize(), fontsize=self.style['label_size'])
        
        if filename is None: filename = f"{metric_name}.png"
        filepath = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_all(self, filename="all.png"):
        if not self.state.step_history: return
        self.plot(filename=filename)
        for metric_name in self.state.step_history.keys():
            self.plot_individual(metric_name)


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
                 logs_per_epoch: int = 10,
                 checkpoint = True,
                 ):
        
        self.trainable_modules = trainable_modules
        self._raw_dataloader = dataloader
        self.num_epochs = num_epochs
        self.accelerator = accelerator
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints" if checkpoint else None
        self.plot_dir = self.output_dir / "plots"

        self.steps_per_epoch = len(self._raw_dataloader)
        self.total_step = self.steps_per_epoch * self.num_epochs

        MAX_EPOCH_PER_LOG = 500
        logs_per_epoch = min(logs_per_epoch, MAX_EPOCH_PER_LOG)
        
        if self.steps_per_epoch <= logs_per_epoch:
            # エポック当たりのlogを取る回数よりもエポックのステップの方が少ない場合、各ステップでlogを取る
            step_log_interval = 1
        else:
            step_log_interval = max(1,self.steps_per_epoch // logs_per_epoch)

        print(step_log_interval)

        self.training_state = TrainingState(
            num_epochs=self.num_epochs, 
            total_steps=self.total_step,
            steps_per_epoch=self.steps_per_epoch,
            step_log_interval = step_log_interval
        )

        self.checkpoint_json_name = "training_state.json"
        self.is_finished = False

        self._load_state()

        if not self.is_finished:
            self.train()
            self.progress_bar = tqdm(
                range(self.total_step),
                desc=f"Epoch {self.training_state.epoch}/{self.num_epochs}",
                initial=self.training_state.step
            )
        else:
            self.progress_bar = None

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

        if self.training_state.epoch > self.num_epochs:
            self.is_finished = True
            print("Training is already finished.")
            

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
        plotter = Plotter(self.training_state, self.plot_dir)
        plotter.plot_all(filename=output_name)