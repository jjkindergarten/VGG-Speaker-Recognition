# Third Party
import librosa
import numpy as np
import time as timelib
import scipy
import soundfile as sf
import scipy.signal as sps
from scipy import interpolate
# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav_fast(vid_path, sr, mode='train'):
    """load_wav() is really slow on this version of librosa.
    load_wav_fast() is faster but we are not ensuring a consistent sampling rate"""
    wav, sr_ret = sf.read(vid_path)

    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav

def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr

    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def lin_spectogram_from_path(path, sr, hop_length, win_length, n_fft, mode):
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    return linear_spect

def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train', data_fromat='wav', argumentation=False):
    if data_fromat == 'wav':
        wav = load_wav(path, sr=sr, mode=mode)
        if argumentation:
            speedup_ratio = np.random.random()/2+0.75
            wav = librosa.effects.time_stretch(wav, speedup_ratio)
        linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    elif data_fromat == 'npy':
        path = path.replace('.wav', '.npy')
        linear_spect = np.load(path)
    else:
        raise IOError('cannot load the data format {}'.format(data_fromat))
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    # print('time: {}, and spec_len is 250'.format(time))
    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time-spec_len)
            spec_mag = mag_T[:, randtime:randtime+spec_len]
        else:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag*(10**5), 0, keepdims=True)/(10**5)
    return (spec_mag - mu) / (std + 1e-3)




