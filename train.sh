#!/bin/bash


# 2020-11-13
#python trainer.py -m /home/nas/user/kbh/DCUNet/chkpt -c ./config/default.yaml

# 2020-11-13
#python trainer.py --chkpt ./model_ckpt/bestmodel.pth -m /home/nas/user/kbh/DCUNet/chkpt -c ./config/v2.yaml -v v2 

python trainer.py --chkpt ./model_ckpt/bestmodel.pth -m /home/nas/user/kbh/DCUNet/chkpt -c ./config/v3.yaml -v v3
