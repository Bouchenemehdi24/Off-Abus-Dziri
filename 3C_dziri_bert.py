# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
#!pip install autogluon

import numpy as np
import re
import sys, os, io

import warnings
import autogluon as ag
warnings.filterwarnings("ignore")
np.random.seed(123)
# %matplotlib inline

from autogluon.core import TabularDataset
from autogluon.multimodal import MultiModalPredictor

train_data_3c = TabularDataset("keras_train_pre_3c_trans_.txt")
test_data_3c = TabularDataset("keras_test_pre_3c_trans_.txt")

print(train_data_3c.head(10))


predictor = MultiModalPredictor(label='label')
predictor.fit(train_data_3c,
              hyperparameters={ 'model.hf_text.checkpoint_name': 'alger-ia/dziribert',"optimization.optim_type": "adamw",'optimization.max_epochs': 4})
                  
print("[INFO] Test Metrics are...")


predictor.evaluate(test_data_3c, metrics=["f1_macro", "accuracy","f1_micro"])