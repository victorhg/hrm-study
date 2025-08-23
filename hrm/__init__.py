import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import pydantic 

from typing import Optional, Tuple

# Base module for lower and higher layers


from .hierarchical_reasoning_models import HierarchicalReasoningModel, HRMConfig
from .models import RecurrentModule, ModelConfig
 