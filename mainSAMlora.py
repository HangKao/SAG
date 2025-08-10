import argparse
import pytorch_lightning as pl
from SAMraw import SamPredictor, sam_model_registry,build_sam_vit_b_ckpt

from sam_lora import LoRA_Sam

import cfg

from sdataset_geobody import geoDataset

from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


import numpy as np
import torch
from torch.utils import data

import warnings
warnings.filterwarnings("ignore")
import random
import os



RANDOM_SEED = 42 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 
    torch.backends.cudnn.deterministic = True # 
    torch.backends.cudnn.benchmark = False # 

set_seed(RANDOM_SEED)


class PrintAccuracyAndLossCallback(pl.Callback):
    def __init__(self,logFile):
        super().__init__()
        self.logFile = logFile
        
    # def on_train_batch_end(self, trainer, pl_module):
    def on_train_batch_end(self, trainer, pl_module,outputs,batch,batch_idx):
        # get the loss value of current epoch
        train_loss = trainer.callback_metrics['train_loss']

        # get the metric of current epoch
        val_eiou = trainer.callback_metrics['train_eiou']
        val_edice = trainer.callback_metrics['train_edice']

        # print the loss and metrics of current epoch
        # print(f"Epoch {trainer.current_epoch}: Train Loss {train_loss:.4f}, Train IOU {val_eiou:.4f}, Train DICE {val_edice:.4f}")
        logText = f"Epoch {trainer.current_epoch} | Step {batch_idx}: Train Loss {train_loss:.4f}, Train IOU {val_eiou:.4f}, Train DICE {val_edice:.4f}.\n"
        self.logginFile(logText)


    def on_validation_epoch_end(self, trainer, pl_module):
        # get the loss value of current epoch
        val_loss = trainer.callback_metrics['valid_loss']

        # get the metric of current epoch
        val_eiou = trainer.callback_metrics['valid_eiou']
        val_edice = trainer.callback_metrics['valid_edice']

        # print the loss and metrics of current epoch
        # print(f"Epoch {trainer.current_epoch}: Valid Loss {val_loss:.4f}, Valid IOU {val_eiou:.4f}, Valid DICE {val_edice:.4f}")
        logText = f"Epoch {trainer.current_epoch} : Valid Loss {val_loss:.4f}, Valid IOU {val_eiou:.4f}, Valid DICE {val_edice:.4f}.\n"
        self.logginFile(logText)

    def logginFile(self,text):
        with open(self.logFile,'a+') as f:
            f.write(text)



def main(args):
    cudaNum = args.gpu_device.split(",")
    cudaNum = [int(x) for x in cudaNum]
    print(cudaNum)

    logFile = "log.txt"
    
    model = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
    # model = build_sam_vit_b_ckpt(args,args.sam_ckpt)

    rankLora = 8

    print("rank = %d"%rankLora)
    # model = LoKA_Sam(model,32)
    model = LoRA_Sam(model,rankLora,args)

    nworkers = 2

    prefix = "/Your_Path/newSamDz/"
    # metaprefix = prefix+"metaFolder/Romney/"
    
    metaprefix = prefix+"metaFolder/GeobodyV1/"
    seis_train_dataset = geoDataset(prefix,metaprefix+"dzTrain.txt")
    seis_valid_dataset = geoDataset(prefix,metaprefix+"dzValid.txt")
    

    # metaprefix = prefix+"metaFolder/Strata/Romney/"
    # seis_train_dataset = geoDataset(prefix,metaprefix+"datasetTrain.txt")
    # seis_valid_dataset = geoDataset(prefix,metaprefix+"datasetValid.txt")


    train_loader = DataLoader(seis_train_dataset, batch_size=args.b, shuffle=True, num_workers=nworkers, pin_memory=True)
    valid_loader = DataLoader(seis_valid_dataset, batch_size=args.b, shuffle=False, num_workers=nworkers, pin_memory=True)

    lr_mon = LearningRateMonitor(logging_interval='epoch')
    log_mon = PrintAccuracyAndLossCallback(logFile)

    # numPrec = 32
    numPrec = 16
    # numPrec = 'bf16-mixed'

    # logger = TensorBoardLogger("tb_logs",name=args.exp_name,version="GAT")
    logger = TensorBoardLogger("tb_logs",name=args.exp_name,version="SAG")


    timesave_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=50,save_top_k=-1)
    bestckpt_callback = pl.callbacks.ModelCheckpoint(mode='min',filename='best_model',save_top_k=-1)
    

    # trainer = pl.Trainer(callbacks=[PrintAccuracyAndLossCallback()])
    trainer = pl.Trainer(devices=cudaNum, 
                        log_every_n_steps=5,
                        logger=logger,
                        max_epochs=args.epoch,
                        # default_root_dir=rgs.pthPath,
                        accelerator='gpu',
                        precision=numPrec,
                        callbacks=[lr_mon,log_mon,timesave_callback,bestckpt_callback],
                        strategy='ddp_find_unused_parameters_true',
                        # strategy='ddp',
                        # weights_save_path=rgs.pthPath
                        )
    
    trainer.fit(model,train_loader,valid_loader)




if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    
    args = cfg.parse_args()
    main(args)
