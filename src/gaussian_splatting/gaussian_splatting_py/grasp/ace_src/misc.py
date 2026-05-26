import yaml
import os
import sys
import random
import torch
import numpy as np
from typing import Any, List, Tuple, Union


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def to(self, device):
        '''note that this function only convert torch.tensor, other type will be ignored
        '''
        for key in self.keys():
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].to(device)
            elif isinstance(self[key], EasyDict):
                self[key].to(device)
        return self

    def detach(self):
        for key in self.keys():
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].detach()
            elif isinstance(self[key], EasyDict):
                self[key].detach()
        return self

    def append(self, d):
        '''合并两个 EasyDict 中可以合并的 value
        '''
        for key in self.keys():
            if isinstance(self[key], torch.Tensor):
                self[key] = torch.cat([self[key], d[key]], dim=0)
            elif isinstance(self[key], np.ndarray):
                self[key] = np.concatenate([self[key], d[key]], axis=0)
            else:
                print(f"Got unsupported type {type(self[key])} in EasyDict.")  # or raise error


def blockPrint():
    # Disable
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    # Restore
    sys.stdout = sys.__stdout__


def load_config(config_path="GIGA.yaml"):
    """
    Load config from /configs/*.yaml
    change path format to pathlib.Path
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/", config_path)
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    cfg["learning_rate"] = float(cfg["learning_rate"])
    return cfg


def set_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # load_config("ACE.yaml")
    a = EasyDict()
    s = "sss"
    a.s = 1
    print("done")
