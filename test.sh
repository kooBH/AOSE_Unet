#!/bin/bash


fs='16' ##16 32 48 44.100, 22.050,..... Other than 16,32,48, sr is automatically changed to nearest fs.
test_model='model_ckpt/bestmodel.pth'
test_data_root_folder='sample_16/'
#test_data_root_folder='.'
test_data_output_path='./output_16/'

python test_DCUnet_jsdr_demand.py \
--fs $fs --test_model $test_model \
--test_data_root_folder $test_data_root_folder \
--test_data_output_path $test_data_output_path
