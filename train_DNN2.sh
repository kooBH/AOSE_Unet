#!/bin/bash


# 2020-11-13
#python trainer.py -m /home/nas/user/kbh/DCUNet/chkpt -c ./config/default.yaml

# 2020-11-13
#python trainer.py --chkpt ./model_ckpt/bestmodel.pth -m /home/nas/user/kbh/DCUNet/chkpt -c ./config/v2.yaml -v v2 

# 2020-11019 DNN2
#python trainer.py  -m /home/nas/user/kbh/DCUNet/chkpt -c ./config/DNN2_v1.yaml -v DNN2_v1

#python trainer.py --chkpt /home/nas/user/kbh/DCUNet/chkpt/DNN2_v1/bestmodel.pt -m /home/nas/user/kbh/DCUNet/chkpt -c ./config/DNN2_v1.yaml -v DNN2_v1_t1

## 2020-11-20 DNN v2
#python trainer.py --chkpt /home/nas/user/kbh/DCUNet/chkpt/CGMM_best/best_v1_201120.pt -m /home/nas/user/kbh/DCUNet/chkpt -c ./config/DNN2_v2.yaml -v DNN2_v2

## 2020-11-26 DNN v3 masking by complex operation
python trainer.py -m /home/nas/user/kbh/DCUNet/chkpt -c ./config/DNN2_v3.yaml -v DNN2_v3
