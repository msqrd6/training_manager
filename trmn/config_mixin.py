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

        if 'metadata' not in self._init_kwargs:
            self._init_kwargs['metadata'] = {}
        # self.__class__.__name__ で「MyModel」などのクラス名文字列を取得
        self._init_kwargs['metadata']['model_name'] = self.__class__.__name__  # 最初から存在するデフォルトの model_name
        
        # 元の __init__ を実行（ここでモデルの各レイヤーが作成される）
        result = init_func(self, *args, **kwargs)
        
        return result
    return wrapper

class ConfigMixin:

    def set_model_name(self, name: str):
        """モデルの名前（メタデータの model_name）を変更・設定する特別な関数"""
        if not hasattr(self, '_init_kwargs'):
            self._init_kwargs = {}
        if 'metadata' not in self._init_kwargs:
            self._init_kwargs['metadata'] = {}
        
        self._init_kwargs['metadata']['model_name'] = name

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
    
    def save_pretrained(self, save_directory: str, sub_dir=None, file_name="model.safetensors"):
        """設定と重みの両方を指定ディレクトリに保存する"""
        if sub_dir is not None:
            save_directory = os.path.join(save_directory, sub_dir)
        self.save_config(save_directory)
        self.save_model(save_directory, file_name=file_name)
        

    @classmethod
    def from_pretrained(cls, load_directory: str, sub_dir=None, load_model_name="model.safetensors", device="cpu"):
        """ディレクトリから設定と重みを読み込み、モデルを復元する"""
        if sub_dir is not None:
            load_directory = os.path.join(load_directory, sub_dir)
            
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