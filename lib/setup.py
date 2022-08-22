import IPython.display as ipd
import sys
from network.branched_network_class import branched_network
import tensorflow as tf
from scipy import signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt 
from pycochleagram import cochleagram as cgram 
from PIL import Image
from pydub import AudioSegment
import os
import random
from sphfile import SPHFile
import string
import shutil
from heapq import nlargest
import copy
import syllables
import prosodic as p
import numpy as np




#retrieves the model, returns it and its associated word key
def load_model():
    np.seterr(divide = 'ignore')
    tf.reset_default_graph()
    net_object = branched_network()
    word_key = np.load('./demo_stim/logits_to_word_key.npy')
    w1 = word_key[:242]
    w2 = word_key[243:588]
    word_key = np.concatenate((w1, w2))
    new_key = []
    trans = str.maketrans('', '', string.punctuation)
    for word in word_key:
        new_key.append(word.decode('UTF-8').lower().translate(trans).strip())
    return net_object, new_key

#returns a lower-case set of the word bank
def build_word_bank():
    bank = set()
    trans = str.maketrans('', '', string.punctuation)
    with open('wordbank.txt','r') as f:
        for line in f:
            for word in line.split():
                bank.add(word.lower().translate(trans).strip())
    return bank

def resample(example, new_size):
    im = Image.fromarray(example)
    resized_image = im.resize(new_size, resample=Image.ANTIALIAS)
    return np.array(resized_image)

def plot_cochleagram(cochleagram, title): 
    plt.figure(figsize=(6,3))
    plt.matshow(cochleagram.reshape(256,256), origin='lower',cmap=plt.cm.Blues, fignum=False, aspect='auto')
    plt.yticks([]); plt.xticks([]); plt.title(title); 
    
def play_wav(wav_f, sr, title):   
    print (title+':')
    ipd.display(ipd.Audio(wav_f, rate=sr))

def generate_cochleagram(wav_f, sr):
    # define parameters
    n, sampling_rate = 50, 16000
    low_lim, hi_lim = 20, 8000
    sample_factor, pad_factor, downsample = 4, 2, 200
    nonlinearity, fft_mode, ret_mode = 'power', 'auto', 'envs'
    strict = True

    # create cochleagram
    #print(type(wav_f))
    #print(wav_f.shape)
    c_gram = cgram.cochleagram(wav_f, sr, n, low_lim, hi_lim, 
                               sample_factor, pad_factor, downsample,
                               nonlinearity, fft_mode, ret_mode, strict)
    #frequencies, times, c_gram = signal.spectrogram(wav_f, sr)
    
    # rescale to [0,255]
    c_gram_rescaled =  255*(1-((np.max(c_gram)-c_gram)/np.ptp(c_gram)))
    #print(type(c_gram_rescaled))
    
    # reshape to (256,256)
    c_gram_reshape_1 = np.reshape(c_gram_rescaled, (211,400))
    c_gram_reshape_2 = resample(c_gram_reshape_1,(256,256))
    
    #plot_cochleagram(c_gram_reshape_2, title)

    # prepare to run through network -- i.e., flatten it
    c_gram_flatten = np.reshape(c_gram_reshape_2, (1, 256*256)) 
    
    return c_gram_flatten

