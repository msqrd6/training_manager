import os
import json
from safetensors.torch import save_model, load_model
import functools
import inspect

# ==========================================
# 🌟 新兵器：自動で引数を保存するデコレータ
# ==========================================
def register_config(init_func):
    """__init__に渡された引数を自動的に self._init_kwargs に保存する魔法のデコレータ"""
    @functools.wraps(init_func)
    def wrapper(self, *args, **kwargs):
        # 渡された引数を解析
        sig = inspect.signature(init_func)
        
        # __init__ の引数に 'use_ema' や '**kwargs' があるかを判定
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        has_use_ema = 'use_ema' in sig.parameters
        
        # クラスが use_ema 等を受け取れない仕様の場合、ここで kwargs から抜き取る（TypeError防止）
        extracted_ema_kwargs = {}
        for key in ['use_ema', 'ema_decay', 'ema_use_scheduler']:
            if key not in sig.parameters and not has_kwargs and key in kwargs:
                extracted_ema_kwargs[key] = kwargs.pop(key)
            
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # self と プライベート変数を除外して保存
        self._init_kwargs = {}
        for k, v in bound_args.arguments.items():
            if k == 'self' or k.startswith('_'):
                continue
            
            # **kwargs の場合は、入れ子にならずフラットに保存する（ロード時に整合させるため）
            param_kind = sig.parameters[k].kind if k in sig.parameters else None
            if param_kind == inspect.Parameter.VAR_KEYWORD:
                self._init_kwargs.update(v)
            else:
                self._init_kwargs[k] = v

        # 抜き取った EMA関連の引数 があれば復元用に記録
        for key, value in extracted_ema_kwargs.items():
            self._init_kwargs[key] = value
            
        # EMAを作るかどうか判定（デフォルトはFalse）
        use_ema_val = self._init_kwargs.get('use_ema', False)

        if 'metadata' not in self._init_kwargs:
            self._init_kwargs['metadata'] = {}
        # self.__class__.__name__ で「MyModel」などのクラス名文字列を取得
        self._init_kwargs['metadata']['class_name'] = self.__class__.__name__
        
        # 元の __init__ を実行（ここでモデルの各レイヤーが作成される）
        result = init_func(self, *args, **kwargs)
        
        # ======== 🌟 EMAの自動初期化 ========
        if use_ema_val:
            from .ema import EMAModule  # 循環参照を防ぐための遅延インポート
            ema_decay = self._init_kwargs.get('ema_decay', 0.999)
            ema_use_scheduler = self._init_kwargs.get('ema_use_scheduler', False)
            self.ema = EMAModule(self, decay=ema_decay, use_scheduler=ema_use_scheduler)
            
        return result
    return wrapper

class ConfigMixin:

    def add_metadata(self, **kwargs):
        """モデルの付加情報を config.json の 'metadata' キー配下にまとめて保存する"""
        if not hasattr(self, '_init_kwargs'):
            self._init_kwargs = {}
        
        if 'metadata' not in self._init_kwargs:
            self._init_kwargs['metadata'] = {}
        
        self._init_kwargs['metadata'].update(kwargs)

    def save_config(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        # ① 設定(config.json)の保存
        if hasattr(self, '_init_kwargs'):
            config_path = os.path.join(save_directory, "config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._init_kwargs, f, indent=4)
        else:
            print("警告: _init_kwargs が見つかりません。Configは保存されません。")

    def save_model(self, save_directory: str, file_name="model.safetensors"):
        os.makedirs(save_directory, exist_ok=True)
        weight_path = os.path.join(save_directory, file_name)
        save_model(self, weight_path)
    
    def save_pretrained(self, save_directory: str):
        """設定と重みの両方を指定ディレクトリに保存する"""
        self.save_config(save_directory)
        self.save_model(save_directory)
        

    @classmethod
    def from_pretrained(cls, load_directory: str, load_model_name="model.safetensors",device="cpu"):
        """ディレクトリから設定と重みを読み込み、モデルを復元する"""
        
        # ① 設定(config.json)のロードとモデルの初期化
        config_path = os.path.join(load_directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                kwargs = json.load(f)
                kwargs.pop("metadata", None)
            model = cls(**kwargs)  # 保存された引数でまっさらなモデルを作成
        else:
            raise FileNotFoundError(f"config.json が {load_directory} に見つかりません。")
            
        # ② 重み(model.safetensors)のロード
        weight_path = os.path.join(load_directory, load_model_name)
        if os.path.exists(weight_path):
            load_model(model, weight_path)
        else:
            raise FileNotFoundError(f"model.safetensors が {load_directory} に見つかりません。")
        
        # 指定されたデバイスへ転送
        model.to(device)
        return model