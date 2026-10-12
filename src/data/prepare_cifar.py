"""Download CIFAR-100 and store as npy arrays for fast GPU loading."""
import os

import numpy as np
import torchvision

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "data_vision")
os.makedirs(ROOT, exist_ok=True)

tr = torchvision.datasets.CIFAR100(ROOT, train=True, download=True)
te = torchvision.datasets.CIFAR100(ROOT, train=False, download=True)
np.save(os.path.join(ROOT, "train_x.npy"), tr.data)  # (50000,32,32,3) uint8
np.save(os.path.join(ROOT, "train_y.npy"), np.array(tr.targets, np.int64))
np.save(os.path.join(ROOT, "test_x.npy"), te.data)
np.save(os.path.join(ROOT, "test_y.npy"), np.array(te.targets, np.int64))
print("CIFAR_PREP_DONE", flush=True)
