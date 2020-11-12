import os
import numpy as np
import sys
sys.path.append('..')
import glob
import argparse

import pickle
import torch
import torchaudio

from tqdm import tqdm
from utils.hparams import HParam
from sklearn.model_selection import train_test_split


# MLDR output path to clean path
def pathTarget2Clean(path):
    splited_path = path.split('/')
    target = splited_path[-1]
    root_path = splited_path[:-3]
    root_path = '/'.join(root_path)
    clean_path= root_path + '/clean/'
    file_id = target.split('_')
    clean_file =  '_'.join(file_id[:2]) + '_BTH.CH0.wav'
    clean_path = clean_path + clean_file

    file_id = target.split('.')[0]

    return clean_path,file_id

# https://tutorials.pytorch.kr/beginner/audio_preprocessing_tutorial.html 
def normalize(tensor):
    tensor_minus_mean = tensor - tensor.mean()
    return tensor_minus_mean / tensor_minus_mean.abs().max()


def genPickle(target_path,save_path,win_len):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    clean_path ,file_id = pathTarget2Clean(target_path)

    window=torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)

    tgt_wav,_ = torchaudio.load(clean_path)
    noi_wav,_ = torchaudio.load(target_path)

    # normalize
    tgt_wav = normalize(tgt_wav)
    noi_wav = normalize(noi_wav)

    tgt_wav_len = tgt_wav.shape[1]

    spec_tgt = torchaudio.functional.spectrogram(waveform=tgt_wav, pad=0, window=window, n_fft=win_len, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)
    spec_noi = torchaudio.functional.spectrogram(waveform=noi_wav, pad=0, window=window, n_fft=win_len, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)
    tgt_wav_real = spec_tgt[0,:,:,0]
    tgt_wav_imag = spec_tgt[0,:,:,1]
    input_wav_real = spec_noi[0,:,:,0]
    input_wav_imag = spec_noi[0,:,:,1]

    batch_dict = {"id": file_id,"tgt_wav_len":tgt_wav_len, "audio_wav" : [noi_wav, tgt_wav],"audio_data_Real":[input_wav_real,tgt_wav_real], "audio_data_Imagine":[input_wav_imag,tgt_wav_imag]}

    with open(save_path+'/'+file_id +'.pkl', 'wb') as f:
        pickle.dump(batch_dict, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',required=True, type=str,help='configuration')
    args = parser.parse_args()

    hp = HParam(args.config)
    fs = hp.train.fs
    wav_path = hp.data.wav

    win_len = int(1024*(fs/16))

    mldr_list = [x for x in glob.glob(os.path.join(wav_path,'mldr','**'), recursive=True) if not os.path.isdir(x)]
    
    train_save_path = hp.data.pkl + 'train/'
    test_save_path  = hp.data.pkl + 'test/' 

    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    mldr_list = [x for x in glob.glob(os.path.join(wav_path, 'mldr','**'), recursive=True) if not os.path.isdir(x)]

    train_list, test_list = train_test_split(mldr_list,test_size=hp.data.test_train_ratio) 

    for target_train in tqdm(train_list,ascii=True,desc="train"):
        genPickle(target_train, train_save_path, win_len)
    for target_test in tqdm(test_list,ascii=True,desc="test"):
        genPickle(target_test, test_save_path, win_len)

        

    
    

