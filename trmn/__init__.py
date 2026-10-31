from .training_manager import TrainingManager, get_trainable_params
from .ema import EMAModule, get_decay_scheduler, default_decay_scheduler
from .config_mixin import ConfigMixin, register_config

# 後方互換性エイリアス
decay_scheduler = default_decay_scheduler
