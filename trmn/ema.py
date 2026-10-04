import copy
import torch
import torch.nn as nn
from safetensors.torch import save_file
import weakref

def decay_scheduler(current_step, max_decay=0.999):
    return min(max_decay, (1.0 + current_step) / (10.0 + current_step))

class EMAModule(nn.Module):
    def __init__(self, model, decay=0.999, use_scheduler=True, scheduler_fn=None):
        super().__init__()
        self.decay = decay
        self.use_scheduler = use_scheduler
        
        # デフォルトのスケジューラ、またはカスタムのスケジューラ関数をセット
        if scheduler_fn is not None:
            self.scheduler_fn = scheduler_fn
        else:
            self.scheduler_fn = decay_scheduler
            
        # PyTorchのサブモジュールとして登録されないよう、弱参照(weakref)で元のモデルを保持する
        # これにより循環参照（model -> ema -> model）によるエラーを防ぎます
        self._source_model_ref = weakref.ref(model)
        
        # 1. 元のモデルと全く同じ構造・キー名を持つ完全なクローンを作成
        self.ema_model = copy.deepcopy(model)
        
        # 2. EMAモデルは学習（逆伝播）させないので、勾配計算をオフにしてメモリを節約
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        # 3. Dropoutなどが誤作動しないよう、常に評価(推論)モードにしておく
        self.ema_model.eval()

    def step(self, current_step=None, decay=None):
        """学習用モデルからEMA値を更新"""
        model = self._source_model_ref()
        if model is None:
            raise RuntimeError("元のモデルが既に破棄されているため、step()を実行できません。")
                
        # 今回のステップで使用するdecayを決定
        if decay is not None:
            current_decay = decay
        elif self.use_scheduler:
            if current_step is None:
                raise ValueError("use_scheduler=Trueの場合、step(current_step=...) で現在のステップ数を渡す必要があります。")
            current_decay = self.scheduler_fn(current_step, self.decay)
        else:
            current_decay = self.decay

        with torch.no_grad():
            # 元のモデルとEMAモデルのパラメータを辞書形式で取得し、名前でマッチングする
            model_params = dict(model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())

            for name, param in model_params.items():
                if param.requires_grad:
                    if name in ema_params:
                        ema_param = ema_params[name]
                        # 指数移動平均の計算
                        ema_param.copy_(
                            current_decay * ema_param + (1.0 - current_decay) * param.data
                        )

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

