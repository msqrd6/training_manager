import copy
import torch
import torch.nn as nn
from safetensors.torch import save_file
import weakref
from typing import Callable


def default_decay_scheduler(current_step: int, max_decay: float = 0.999) -> float:
    """ステップ数に応じて徐々に max_decay へ漸近するデフォルトのスケジューラ関数"""
    return min(max_decay, (1.0 + current_step) / (10.0 + current_step))


def get_decay_scheduler(max_decay: float = 0.999) -> Callable[[int], float]:
    """指定した max_decay に向かってウォームアップするスケジューラ関数を返すビルダー"""
    return lambda step: min(max_decay, (1.0 + step) / (10.0 + step))


class EMAModule(nn.Module):
    def __init__(self, model, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self._source_model_ref = weakref.ref(model)
        
        # 1. 元のモデルと全く同じ構造・キー名を持つ完全なクローンを作成
        self.ema_model = copy.deepcopy(model)
        
        # 💡 元モデルが FP16/BF16 であっても、アンダーフロー防止のため EMAモデルは float32 で保持する
        self.ema_model.float()
        
        # 2. EMAモデルは学習（逆伝播）させないので、勾配計算をオフにしてメモリを節約
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        # 3. Dropoutなどが誤作動しないよう、常に評価(推論)モードにしておく
        self.ema_model.eval()

    def step(self, decay: float = None):
        """学習用モデルからEMA値を更新"""
        model = self._source_model_ref()
        if model is None:
            raise RuntimeError("元のモデルが既に破棄されているため、step()を実行できません。")
                
        current_decay = decay if decay is not None else self.decay

        with torch.no_grad():
            # 元のモデルとEMAモデルのパラメータを辞書形式で取得し、名前でマッチングする
            model_params = dict(model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())

            for name, param in model_params.items():
                if param.requires_grad:
                    if name in ema_params:
                        ema_param = ema_params[name]
                        # 元モデルのパラメータを detach し、EMAモデルと同じデバイス・型 (float32) に変換
                        src_param = param.detach().to(device=ema_param.device, dtype=ema_param.dtype)
                        # lerp_ (線形補間: ema = ema + weight * (src - ema)) でインプレース高速更新
                        ema_param.lerp_(src_param, 1.0 - current_decay)

            # バッファ（BatchNormのrunning_mean, running_varなど）も同期
            model_buffers = dict(model.named_buffers())
            ema_buffers = dict(self.ema_model.named_buffers())
            for name, buf in model_buffers.items():
                if name in ema_buffers:
                    ema_buf = ema_buffers[name]
                    src_buf = buf.detach().to(device=ema_buf.device)
                    if ema_buf.is_floating_point():
                        src_buf = src_buf.to(dtype=ema_buf.dtype)
                        ema_buf.lerp_(src_buf, 1.0 - current_decay)
                    else:
                        ema_buf.copy_(src_buf)

    def state_dict(self, *args, **kwargs):
        """
        保存時に、親のEMAModuleではなく、中の ema_model の状態を直接返すように上書き
        """
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """
        読み込み時も同様に、中の ema_model に直接流し込む
        """
        return self.ema_model.load_state_dict(state_dict, strict=strict)
    
    def save_pretrained(self, save_directory, sub_dir=None, save_name="ema_model"):
        import os
        if sub_dir is not None:
            save_directory = os.path.join(save_directory, sub_dir)
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, f"{save_name}.safetensors")
        save_file(self.ema_model.state_dict(), path)

    def load_pretrained(self, load_directory, sub_dir=None, load_name="ema_model"):
        import os
        from safetensors.torch import load_file
        if sub_dir is not None:
            load_directory = os.path.join(load_directory, sub_dir)
        path = os.path.join(load_directory, f"{load_name}.safetensors")
        if os.path.exists(path):
            state_dict = load_file(path)
            self.load_state_dict(state_dict)
        else:
            print(f"Warning: EMA checkpoint not found at {path}")
