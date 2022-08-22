from lib.setup import *

#takes in audio file, sampling rate, return prob dist
def run_file(sr, wav_f, title, model):
    c_gram = generate_cochleagram(wav_f, sr, title)
    logits = model.session.run(model.word_logits, feed_dict={model.x: c_gram})
    l1 = logits[0][:242]
    l2 = logits[0][243:588]
    return np.concatenate((l1, l2))

#return prob dist for pregenerated cgram
def run_cgram(cgram, model):
    logits = model.session.run(model.word_logits, feed_dict={model.x: cgram})
    l1 = logits[0][:242]
    l2 = logits[0][243:588]
    return np.concatenate((l1, l2))

#changes dist to top n outcomes
def distrib(logits, n): #gets top n outcomes
    inds = np.argpartition(logits, -1 * n)[-1 * n:]
    return inds[np.argsort(logits[inds])][::-1]

def merge_inds(results, target, inds):
    if target in results:
        dictA = results[target]
    else:
        dictA = dict()
    l = len(inds)
    for i in range(l):
        place = l - i
        val = inds[i]
        if val in dictA:
            dictA[val] += place
        else:
            dictA[val] = place
    results[target] = dictA
    return results

