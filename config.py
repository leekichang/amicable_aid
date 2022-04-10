import os
import torch
import warnings
import sys
warnings.filterwarnings(action='ignore')

BATCH_SIZE      = 256
LEARNING_RATE   = 0.001
EPOCH           = 50
input_size      = 256

USE_CUDA        = torch.cuda.is_available()
DEVICE          = torch.device("cuda" if USE_CUDA else "cpu")
print(f"WORKING WITH {DEVICE}", file = sys.stderr)

keti_config       = {'sampling_rate' : 0.1  ,
                     'n_train'       : 11176,
                     'n_test'        : 5745 ,
                     'n_channel'     : 4    ,
                     'n_label'       : 2    ,
                     'n_axis'        : 1    }

motion_config     = {'sampling_rate' : 30   ,
                     'n_train'       : 9599 ,
                     'n_test'        : 1474 ,
                     'n_channel'     : 45   ,
                     'n_label'       : 4    ,
                     'n_axis'        : 3    }

seizure_config    = {'sampling_rate' : 256  ,
                     'n_train'       : 19292,
                     'n_test'        : 2144 ,
                     'n_channel'     : 18   ,
                     'n_label'       : 2    ,
                     'n_axis'        : 1    }

wifi_config       = {'sampling_rate' : 1000 ,
                     'n_train'       : 7289 ,
                     'n_test'        : 810  ,
                     'n_channel'     : 180  ,
                     'n_label'       : 7    ,
                     'n_axis'        : 3    }

                     
pamap2_config     = {'sampling_rate' : 100  ,
                     'n_train'       : 22794,
                     'n_test'        : 2000 ,
                     'n_channel'     : 9    ,
                     'n_label'       : 7    ,
                     'n_axis'        : 3    }

dataset_config = {'keti'   : keti_config,
                  'motion' : motion_config,
                  'seizure': seizure_config,
                  'wifi'   : wifi_config,
                  'PAMAP2' : pamap2_config,}