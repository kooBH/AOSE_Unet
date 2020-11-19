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
from sklearn.preprocessing import normalize
from SyncWav import sync_wav

from multiprocessing import Pool, cpu_count

clean_root = '/home/nas/DB/CHiME4/data/audio/16kHz/isolated_ext/'
clean_root = '/home/nas/DB/CHiME4/data/audio/16kHz/isolated_ext/'
noisy_root_1 = '/home/nas/user/albert/2020_IITP_share/data/CGMM-RLS/trial_01/winL1024_gamma0.99_Ln5_MVDRon0/'
noisy_root_2 = '/home/nas/user/albert/2020_IITP_share/data/CGMM-RLS/trial_01/winL1024_gamma0.99_Ln5_MVDRon1/'

clean_dict = {}

def getIDfromNoisy(t):
    x = t.split('/')
    x = x[-1]
    x = x.split('_')
    if len(x) > 2 :
        y = x[2].split('.')
        x = x[0]+'_'+x[1]
        return x, x +'_'+y[0]
    else :
        x = x[0] + '_'+ x[1].split('.')[0]
        return x, x


def genPickle(noisy_path,save_path,window,win_len):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    targetID,file_id = getIDfromNoisy(noisy_path)
    clean_path = clean_dict[targetID]

    clean_wav,_ = torchaudio.load(clean_path)
    noisy_wav,_ = torchaudio.load(noisy_path)

    clean_numpy = clean_wav.numpy()
    noisy_numpy = noisy_wav.numpy()

    clean_numpy = normalize(clean_numpy)
    noisy_numpy = normalize(noisy_numpy)

    clean_numpy,noisy_numpy = sync_wav(clean_numpy,noisy_numpy)

    clean_wav = torch.from_numpy(clean_numpy)
    noisy_wav = torch.from_numpy(noisy_numpy)


    clean_wav = clean_wav.unsqueeze(0)
    noisy_wav = noisy_wav.unsqueeze(0)

    clean_wav_len = clean_wav.shape[1]

    spec_clean = torchaudio.functional.spectrogram(waveform=clean_wav, pad=0, window=window, n_fft=win_len, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)
    spec_noisy = torchaudio.functional.spectrogram(waveform=noisy_wav, pad=0, window=window, n_fft=win_len, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)

    clean_wav_real = spec_clean[0,:,:,0]
    clean_wav_imag = spec_clean[0,:,:,1]
    noisy_wav_real = spec_noisy[0,:,:,0]
    noisy_wav_imag = spec_noisy[0,:,:,1]

    batch_dict = {"id": file_id,"clean_wav_len":clean_wav_len, "audio_wav" : [noisy_wav, clean_wav],"audio_data_Real":[noisy_wav_real,clean_wav_real], "audio_data_Imagine":[noisy_wav_imag,clean_wav_imag]}

    with open(save_path+'/'+file_id +'.pkl', 'wb') as f:
        pickle.dump(batch_dict, f)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c','--config',required=True, type=str,help='configuration')
    # args = parser.parse_args()

    hp = HParam('../config/CGMM_v1.yaml')
    fs = hp.train.fs

    win_len = int(1024*(fs/16))

    train_list_1 = [x for x in glob.glob(os.path.join(noisy_root_1,'tr*simu','*.wav')) if not os.path.isdir(x)]
    test_list_1= [x for x in glob.glob(os.path.join(noisy_root_1,'dt*simu','*.wav')) if not os.path.isdir(x)] + [x for x in glob.glob(os.path.join(noisy_root_1,'et*simu','*.wav')) if not os.path.isdir(x)]


    train_list_2 = [x for x in glob.glob(os.path.join(noisy_root_2,'tr*simu','*.wav')) if not os.path.isdir(x)]
    test_list_2 = [x for x in glob.glob(os.path.join(noisy_root_2,'dt*simu','*.wav')) if not os.path.isdir(x)] + [x for x in glob.glob(os.path.join(noisy_root_2,'et*simu','*.wav')) if not os.path.isdir(x)]

    clean_list =   [x for  x in glob.glob(os.path.join(clean_root,'*','*.CH1.Clean.wav')) if not os.path.isdir(x)]

    for t in clean_list : 
        x = t.split('/')
        x = x[-1]
        x = x.split('_')
        x = x[0]+'_'+x[1]
        clean_dict[x] = t

    window=torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    
    train_save_path_1 = hp.data.pkl + 'train_1/'
    test_save_path_1  = hp.data.pkl + 'test_1/' 

    train_save_path_2 = hp.data.pkl + 'train_2/'
    test_save_path_2  = hp.data.pkl + 'test_2/' 

    if not os.path.exists(train_save_path_1):
        os.makedirs(train_save_path_1)
    if not os.path.exists(test_save_path_1):
        os.makedirs(test_save_path_1)
    if not os.path.exists(train_save_path_2):
        os.makedirs(train_save_path_2)
    if not os.path.exists(test_save_path_2):
        os.makedirs(test_save_path_2)


    cpu_num = cpu_count()

    def train_wrapper_1(num):
        target = train_list_1[num]
        genPickle(target,train_save_path_1,window,win_len)

    def test_wrapper_1(num):
        target = test_list_1[num]
        genPickle(target,test_save_path_1,window,win_len)

    def train_wrapper_2(num):
        target = train_list_2[num]
        genPickle(target,train_save_path_2,window,win_len)

    def test_wrapper_2(num):
        target = test_list_2[num]
        genPickle(target,test_save_path_2,window,win_len)


    arr = list(range(len(train_list_1)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(train_wrapper_1, arr), total=len(arr),ascii=True,desc='TRAIN_1'))

    arr = list(range(len(test_list_1)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(test_wrapper_1, arr), total=len(arr),ascii=True,desc='TEST_1'))


    arr = list(range(len(train_list_2)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(train_wrapper_2, arr), total=len(arr),ascii=True,desc='TRAIN_2'))

    arr = list(range(len(test_list_2)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(test_wrapper_2, arr), total=len(arr),ascii=True,desc='TEST_2'))

