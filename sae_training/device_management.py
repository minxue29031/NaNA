from typing import Callable, Iterable

import torch
from torch import nn


def get_device(module):
    return next(module.parameters()).device


def unload_competing_modules_on_use(
        func: Callable,
        competing_modules: Iterable[nn.Module],
):
    def wrapper(self, *args, **kwargs):
        if get_device(self) == torch.device(self.cfg.device):
            return func(*args, **kwargs)

        for module in competing_modules:
            if get_device(module) == torch.device(self.cfg.device):
                nn.Module.to(self, "cpu")

        torch.cuda.empty_cache()

        # This enables multi-gpu support for Transformers
        if hasattr(self, "move_model_modules_to_device"):
            self.move_model_modules_to_device()
        else:
            nn.Module.to(self, self.cfg.device)

        return func(*args, **kwargs)
    return wrapper