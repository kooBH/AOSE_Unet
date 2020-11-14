#!/bin/bash

python inference.py -c ./config/default.yaml -m /home/nas/user/kbh/DCUNet/chkpt/bestmodel.pt -i ./sample_16 -o ./output_16
