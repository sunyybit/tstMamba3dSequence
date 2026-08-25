import sys
import torch.nn as nn
from collections import OrderedDict
from models.utils.structure import Point


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                else:
                    input = module(input)
        return input
