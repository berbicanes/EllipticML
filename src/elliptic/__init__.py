"""Detecting illicit Bitcoin transactions on the Elliptic dataset."""

# On macOS, xgboost (Homebrew libomp) and torch (its own bundled libomp) load
# two OpenMP runtimes into one process. Tested combinations:
#   torch first                      -> xgboost crashes on fit
#   xgboost first, torch multi-core  -> torch crashes ("pthread_mutex_init failed")
#   xgboost first, torch one thread  -> both work
# So xgboost is imported first and torch is limited to one CPU thread. The
# neural models here are small, so the cost is a few minutes in total.
import xgboost  # noqa: F401, I001  (must stay first)
import torch

torch.set_num_threads(1)
