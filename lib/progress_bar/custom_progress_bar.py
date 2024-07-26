from typing import Any

from lightning.pytorch.callbacks import TQDMProgressBar


class CustomProgressBar(TQDMProgressBar):
    def __init__(self, dataset_name, model_name, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.model_name = model_name
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any):
        super().on_train_epoch_start(trainer)
        self.train_progress_bar.set_description(f'{self.model_name}-{self.dataset_name} Epoch {trainer.current_epoch}')
