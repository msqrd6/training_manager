import copy
import torch
import torch.nn as nn

def decay_scheduler(current_step, max_decay=0.999):
    return min(max_decay, (1.0 + current_step) / (10.0 + current_step))

class EMAModule(nn.Module):
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.decay = decay
        
        # 1. 元のモデルと全く同じ構造・キー名を持つ完全なクローンを作成
        self.ema_model = copy.deepcopy(model)
        
        # 2. EMAモデルは学習（逆伝播）させないので、勾配計算をオフにしてメモリを節約
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        # 3. Dropoutなどが誤作動しないよう、常に評価(推論)モードにしておく
        self.ema_model.eval()

    def step(self, model, decay=None):
        """学習用モデルからEMA値を更新"""
        decay = decay if decay is not None else self.decay

        with torch.no_grad():
            # 元のモデルとEMAモデルのパラメータを辞書形式で取得し、名前でマッチングする
            model_params = dict(model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())

            for name, param in model_params.items():
                if param.requires_grad:
                    ema_param = ema_params[name]
                    # 指数移動平均の計算
                    ema_param.copy_(
                        decay * ema_param + (1.0 - decay) * param.data
                    )

    def state_dict(self, *args, **kwargs):
        """
        保存時に、親のEMAModuleではなく、
        中の ema_model の状態を直接返すように上書きする
        """
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """
        読み込み時も同様に、中の ema_model に直接流し込む
        """
        return self.ema_model.load_state_dict(state_dict, strict=strict)