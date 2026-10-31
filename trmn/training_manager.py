import os
import json
import contextlib
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from safetensors.torch import save_model, load_model
from accelerate import Accelerator

from collections import defaultdict
from typing import Any, Dict, Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from .ema import EMAModule


def _resolve_trainable_modules(modules) -> dict[str, nn.Module]:
    """モジュールを辞書形式で解決して返す"""
    if isinstance(modules, dict):
        return dict(modules)

    if isinstance(modules, nn.Module):
        return {"model": modules}

    if isinstance(modules, (list, tuple)):
        module_list = list(modules)
        if len(module_list) == 1:
            return {"model": module_list[0]}
        return {f"model_{i + 1}": mod for i, mod in enumerate(module_list)}

    raise TypeError(f"trainable_modules must be nn.Module, list/tuple of nn.Module, or dict, got {type(modules)}")


def get_trainable_params(*trainable_modules) -> list[torch.Tensor]:
    if len(trainable_modules) == 1:
        arg = trainable_modules[0]
        if isinstance(arg, dict):
            trainable_modules = list(arg.values())
        elif isinstance(arg, (list, tuple)):
            trainable_modules = list(arg)
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
        for k, v in state.get("epoch_history", {}).items():
            self.epoch_history[k].update(v)

        # JSONからstep_historyを復元する
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
        
        self.style = {
            "title" : "Training Loss",
            "xlabel": "steps",

            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'],
            
            'raw_linewidth': 1.0,      # 生データの線の太さ
            'raw_alpha': 0.3,          # 生データの透明度
            'smooth_linewidth': 1.2,   # 平滑化データの線の太さ
            'smooth_factor': 0.1,      # 平滑化の強さ
            
            'grid_style': '--',
            'grid_alpha': 0.8,
            'title_size': 12,
            'label_size': 10,
            
            'tick_label_size': 9,
            'tick_color': "#181818",
            
            'spine_width': 0.4,
            'spine_color': "#313131"
        }
        
        self.style.update(custom_styles)

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

        ax.tick_params(
            axis='both', 
            labelsize=self.style['tick_label_size'], 
            colors=self.style['tick_color']
        )

        self._apply_k_formatting(ax, xlabel)

    def _get_step_plot_data(self, data_dict: dict) -> tuple[list[int], list[float]]:
        xy = [(int(k), float(v)) for k, v in data_dict.items()]
        xy.sort(key=lambda x: x[0])
        return [p[0] for p in xy], [p[1] for p in xy]

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
        
        self._apply_standard_styling(ax, self.style['title'], self.style['xlabel'])
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
    チェックポイント（重みログ）および再開用ステート（_training_state/）の管理を行うクラス。
    """
    def __init__(self,
                 trainable_modules: list[nn.Module] | dict[str, nn.Module] | nn.Module,
                 dataloader: DataLoader,
                 num_epochs: int,
                 accelerator: Accelerator,
                 output_dir: str,
                 save_every_n_epochs: int = None,
                 logs_per_epoch: int = 10,
                 save_training_state: bool = True,
                 ema: bool = False,
                 ema_decay: float | Callable[[int], float] = 0.999,
                 ):
        self.named_modules: dict[str, nn.Module] = _resolve_trainable_modules(trainable_modules)
        self.trainable_modules: list[nn.Module] = list(self.named_modules.values())

        self._raw_dataloader = dataloader
        self.num_epochs = num_epochs
        self.accelerator = accelerator
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = Path(output_dir)
        
        # 💡 再開用ステート（最新上書き用）と、推論用チェックポイント（エポックごとの重みログ）を分離
        self.training_state_dir = self.output_dir / "_training_state" if save_training_state else None
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plot_dir = self.output_dir / "plots"

        self.use_ema = ema
        self.ema_decay = ema_decay
        self.ema_modules: dict[str, EMAModule] = {}
        if self.use_ema:
            base_decay = self.ema_decay if not callable(self.ema_decay) else 0.999
            for name, module in self.named_modules.items():
                raw_module = self.accelerator.unwrap_model(module)
                ema_mod = EMAModule(raw_module, decay=base_decay)
                ema_mod.to(self.accelerator.device)
                self.ema_modules[name] = ema_mod

        self.steps_per_epoch = len(self._raw_dataloader)
        self.total_step = self.steps_per_epoch * self.num_epochs

        MAX_EPOCH_PER_LOG = 500
        logs_per_epoch = min(logs_per_epoch, MAX_EPOCH_PER_LOG)
        
        if self.steps_per_epoch <= logs_per_epoch:
            step_log_interval = 1
        else:
            step_log_interval = max(1, self.steps_per_epoch // logs_per_epoch)

        self.training_state = TrainingState(
            num_epochs=self.num_epochs, 
            total_steps=self.total_step,
            steps_per_epoch=self.steps_per_epoch,
            step_log_interval=step_log_interval
        )

        self.checkpoint_json_name = "training_state.json"
        self.is_finished = False
        self._last_completed_epoch = None

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
    def ema(self):
        """EMAモジュールが1つの場合のショートカット、または先頭のEMAモジュールを返す"""
        return next(iter(self.ema_modules.values())) if self.ema_modules else None

    @property
    def epochs(self):
        """途中再開を考慮したエポックのイテレータを返す"""
        return range(self.training_state.epoch, self.num_epochs + 1)

    @property
    def sync_gradients(self) -> bool:
        """accelerator.sync_gradients のプロパティ転送"""
        return self.accelerator.sync_gradients

    @contextlib.contextmanager
    def accumulate(self, *models, autocast: bool = True):
        """
        勾配蓄積(accelerator.accumulate)と自動混合精度(accelerator.autocast)を
        一括で適用するコンテキストマネージャー。
        モデル引数を省略した場合は、登録されている全 trainable_modules を自動対象にします。
        例:
            with tm.accumulate():
                pred = model(x)
                loss = loss_fn(pred, y)
        """
        target_models = models if models else self.trainable_modules
        with contextlib.ExitStack() as stack:
            for m in target_models:
                stack.enter_context(self.accelerator.accumulate(m))
            if autocast:
                stack.enter_context(self.accelerator.autocast())
            yield

    def backward(self, loss, **kwargs):
        """accelerator.backward のプロキシメソッド"""
        self.accelerator.backward(loss, **kwargs)

    # タイポ対応エイリアス
    backword = backward

    @contextlib.contextmanager
    def unwrapped(self, *models):
        """
        withブロック内だけで一時的にアンラップされた生のモデル（リスト）を返すコンテキストマネージャー。
        例:
            with tm.unwrapped(model1, model2) as (raw1, raw2):
                raw1.save_pretrained(...)
        """
        if not models:
            models = self.trainable_modules
        unwrapped_models = tuple(self.accelerator.unwrap_model(m) for m in models)
        if len(unwrapped_models) == 1:
            yield unwrapped_models[0]
        else:
            yield unwrapped_models

    @property
    def dataloader(self):
        """途中再開（Resume）時に、そのエポックですでに処理済みのバッチをスキップして返す"""
        steps_done_in_epoch = self.training_state.step - (self.training_state.epoch - 1) * self.steps_per_epoch
        if steps_done_in_epoch > 0:
            return self.accelerator.skip_first_batches(self._raw_dataloader, num_batches=steps_done_in_epoch)
        return self._raw_dataloader

    def _load_state(self):
        """_training_state/ （またはレガシーな training_state/, checkpoints/）が存在すれば読み込み、状態を復元する"""
        if self.training_state_dir is None:
            return

        json_path = os.path.join(self.training_state_dir, self.checkpoint_json_name)
        legacy_dirs = [
            self.output_dir / "training_state",
            self.output_dir / "checkpoints",
        ]

        load_dir = None
        if os.path.exists(json_path):
            load_dir = self.training_state_dir
        else:
            for legacy_dir in legacy_dirs:
                legacy_json = os.path.join(legacy_dir, self.checkpoint_json_name)
                if os.path.exists(legacy_json):
                    load_dir = legacy_dir
                    json_path = legacy_json
                    break

        if load_dir is not None and os.path.exists(json_path):
            # 1. 独自の進行状況をJSON構造から復元
            with open(json_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                self.training_state.load_state_dict(state)

            # 2. Acceleratorが管理する重みやオプティマイザの状態を復元
            self.accelerator.load_state(load_dir)

            # 3. EMAモデルの重みを復元
            if self.use_ema:
                for i, (name, ema_mod) in enumerate(self.ema_modules.items()):
                    ema_path = os.path.join(load_dir, f"{name}_ema.safetensors")
                    if not os.path.exists(ema_path):
                        # レガシーな ema_000.safetensors 形式のフォールバック
                        legacy_path = os.path.join(load_dir, f"ema_{i:03d}.safetensors")
                        if os.path.exists(legacy_path):
                            ema_path = legacy_path
                    if os.path.exists(ema_path):
                        load_model(ema_mod.ema_model, ema_path)

            print(f"Loaded training state: Epoch {self.training_state.epoch}, Step {self.training_state.step}")
        else:
            print("Initialized new training state directory.")

        if self.training_state.epoch > self.num_epochs:
            self.is_finished = True
            print("Training is already finished.")

    def is_savepoint(self) -> bool:
        """チェックポイント（重みログ）の保存タイミングであるかを判定する"""
        target_epoch = self._last_completed_epoch if self._last_completed_epoch is not None else self.training_state.epoch
        if target_epoch > self.num_epochs:
            return False

        # save_every_n_epochs が None の場合: 常に最新1件を上書き保存するため毎エポック True
        if self.save_every_n_epochs is None:
            return True

        if target_epoch == self.num_epochs:
            return True
        if self.save_every_n_epochs > 0 and target_epoch % self.save_every_n_epochs == 0:
            return True
        return False

    def save_training_state(self):
        """学習再開用の最新状態（Optimizer, Scheduler, 乱数状態, EMA等）を _training_state/ に上書き保存する"""
        if self.training_state_dir is None:
            return

        self.accelerator.wait_for_everyone()
        os.makedirs(self.training_state_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            # 1. TrainingState のJSONを保存
            json_path = os.path.join(self.training_state_dir, self.checkpoint_json_name)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.training_state.state_dict(), f, indent=4)

            # 2. EMAモデルの状態を保存（再開用）
            if self.use_ema:
                for name, ema_mod in self.ema_modules.items():
                    save_model(ema_mod.ema_model, str(self.training_state_dir / f"{name}_ema.safetensors"))

        # 3. Acceleratorが管理する重みやオプティマイザの状態を全プロセスで保存
        self.accelerator.save_state(self.training_state_dir)
        self.accelerator.wait_for_everyone()

    def checkpoint(self, save_dir_name: str = None):
        """
        推論・評価用のチェックポイント（純粋なモデル重みとEMAの.safetensors）を checkpoints/ に保存する。
        - save_every_n_epochs が数値（例: 2）の場合:
            checkpoints/epoch_001/transformer/model.safetensors, model_ema.safetensors, config.json
        - save_every_n_epochs が None の場合:
            checkpoints/transformer/model.safetensors, model_ema.safetensors, config.json (常に最新で上書き)
        """
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if save_dir_name is None:
                if self.save_every_n_epochs is None:
                    base_save_path = self.checkpoint_dir
                else:
                    target_epoch = self._last_completed_epoch if self._last_completed_epoch is not None else self.epoch
                    base_save_path = self.checkpoint_dir / f"epoch_{target_epoch:03d}"
            else:
                base_save_path = self.checkpoint_dir / save_dir_name

            base_save_path.mkdir(parents=True, exist_ok=True)

            # 各モデルごとにサブフォルダを作成して保存
            for name, module in self.named_modules.items():
                model_dir = base_save_path / name
                model_dir.mkdir(parents=True, exist_ok=True)

                raw_model = self.accelerator.unwrap_model(module)
                if hasattr(raw_model, "save_pretrained") and callable(raw_model.save_pretrained):
                    raw_model.save_pretrained(str(model_dir), file_name="model.safetensors")
                else:
                    save_model(raw_model, str(model_dir / "model.safetensors"))

                # EMAモデルの保存
                if self.use_ema and name in self.ema_modules:
                    ema_mod = self.ema_modules[name]
                    save_model(ema_mod.ema_model, str(model_dir / "model_ema.safetensors"))

        self.accelerator.wait_for_everyone()

    # 別名エイリアス
    save_checkpoint = checkpoint

    def save_models(self, save_dir_name="save_model"):
        """指定フォルダにモデルとEMAのsafetensorsを保存する（互換用メソッド）"""
        self.checkpoint(save_dir_name=save_dir_name)

    def train(self):
        for module in self.trainable_modules:
            if hasattr(module, 'train') and callable(module.train):
                module.train()

    def step_end(self, decimals: int = 3, **kwargs):
        """1ステップ（1バッチ）終了時の処理。kwargsで動的に受け取る"""
        if self.use_ema and self.sync_gradients:
            if callable(self.ema_decay):
                current_decay = self.ema_decay(self.training_state.step)
            else:
                current_decay = self.ema_decay

            for ema_mod in self.ema_modules.values():
                ema_mod.step(decay=current_decay)
        
        self.training_state.step_end(**kwargs)
        self.progress_bar.update(1)
        
        postfix_data = {}
        for key, value in kwargs.items():
            val = value.item() if hasattr(value, 'item') else value
            if isinstance(val, (float, int)):
                postfix_data[key] = f"{val:.{decimals}f}"
            else:
                postfix_data[key] = val
                
        self.progress_bar.set_postfix(**postfix_data)

    def epoch_end(self, auto_checkpoint: bool = True, auto_save_training_state: bool = True):
        """1エポック終了時の処理。再開用ステート（_training_state/）更新およびチェックポイント保存を一括実行する"""
        self._last_completed_epoch = self.training_state.epoch
        metrics = self.training_state.get_current_state()
        metrics_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        tqdm.write(f"Epoch {self.training_state.epoch}/{self.num_epochs} : {metrics_str}")
        
        self.training_state.epoch_end()

        # ① 再開用ステートを毎エポック自動更新（クラッシュ対策）
        if auto_save_training_state:
            self.save_training_state()

        # ② チェックポイントタイミングであれば自動で保存
        if auto_checkpoint and self.is_savepoint():
            self.checkpoint()
        
        if self.training_state.epoch <= self.num_epochs:
            self.progress_bar.set_description(f"Epoch {self.training_state.epoch}/{self.num_epochs}")

    def plot(self, output_name: str = "all.png"):
        plotter = Plotter(self.training_state, self.plot_dir)
        plotter.plot_all(filename=output_name)
