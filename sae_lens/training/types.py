from collections.abc import Iterator

import torch

DataProvider = Iterator[torch.Tensor]
MultiHookDataProvider = Iterator[dict[str, torch.Tensor]]
