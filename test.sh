#!/bin/bash

#python inference.py -c ./config/test_cuda_1.yaml -m /home/nas/user/kbh/DCUNet/chkpt/v1/bestmodel.pt -i /home/nas/user/kbh/Chime_MLDR/full/ -o ./output_v1


# python inference.py -c ./config/test_cuda_1.yaml -m ./model_ckpt/bestmodel.pth -i /home/nas/user/kbh/Chime_MLDR/full/ -o ./output_best

#python inference.py -c ./config/default.yaml -m /home/nas/user/kbh/DCUNet/chkpt/v1/bestmodel.pt -i ./sample_16 -o ./output_v1

# python inference.py -c ./config/test_cuda_1.yaml -m /home/nas/user/kbh/CHiME4_CGMM -i /home/nas/user/kbh/Chime_MLDR/full/ -o ./output_best

# python inference.py -c ./config/test_cuda_1.yaml -m /home/nas/user/kbh/DCUNet/chkpt/temp/best_11191450.pt -i /home/nas/user/albert/2020_IITP_share/data/CGMM-RLS/trial_01/winL1024_gamma0.99_Ln5_MVDRon0/test/ -o /home/nas/user/kbh/DCUNet/output_best/CGMM_on0/

#python inference.py -c ./config/test_cuda_1.yaml -m /home/nas/user/kbh/DCUNet/chkpt/DNN2_v1/bestmodel.pt -i ./sample_16 -o ./output_sample


### 2020-11-20

python inference.py -c ./config/test_cuda_1.yaml -m /home/nas/user/kbh/DCUNet/chkpt/CGMM_best/best_201120l.pt -i /home/nas/user/albert/2020_IITP_share/data/CGMM-RLS/trial_01/winL1024_gamma0.99_Ln5_MVDRon0/test/ -o /home/nas/user/kbh/DCUNet/output_best/CGMM_on0/
python inference.py -c ./config/test_cuda_1.yaml -m /home/nas/user/kbh/DCUNet/chkpt/CGMM_best/best_201120l.pt -i /home/nas/user/albert/2020_IITP_share/data/CGMM-RLS/trial_01/winL1024_gamma0.99_Ln5_MVDRon1/test/ -o /home/nas/user/kbh/DCUNet/output_best/CGMM_on1
