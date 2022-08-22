from lib.setup import *

#manual offset as everything is 16kHz
def process_wav(sr, wav_f, offset):
    min_len = 2 * sr
    if offset < 0:
        buf = (-1) * offset
        silence = np.zeros((buf,), dtype = int)
        wav_f = np.concatenate((silence, wav_f))
        if len(wav_f[:2 * sr]) < min_len:
            buf = min_len - len(wav_f)
            silence = np.zeros((buf,), dtype = int)
            wav_f = np.concatenate((wav_f, silence))
        return wav_f[:2 * sr]
    elif len(wav_f) < min_len + offset:
        buf = min_len + offset - len(wav_f)
        silence = np.zeros((buf,), dtype = int)
        wav_f = np.concatenate((wav_f, silence))
    return wav_f[offset : offset + 2 * sr]

