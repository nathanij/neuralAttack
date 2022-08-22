from lib.setup import *
from lib.fileinfo import *
from lib.running import *

#checks if a word is in the top n of a results dictionary
def check_top_n(word, dic, n):
    arr = sorted(dic, key = dic.get, reverse = True)[:min(len(dic), n)]
    if word in arr:
        return 1
    else:
        return 0

#finds max results in a results dictionary
def find_max_word(dic):
    return sorted(dic, key = dic.get, reverse = True)[0]

#generates word bank of n words
def random_sel(word_bank, n): #1 <= n <= 587
    l = random.sample(word_bank, n)
    bank = dict()
    for word in l:
        bank[word] = word_bank.index(word)
    return bank

#uses full bank (or random selection of n words) to check every cgram
#returns an array of result dictionaries (by word) for each dr
def full_word_test(drs, p_range, n = 587):
    net_object, word_key = load_model()
    word_bank = random_sel(word_key, n)
    ds = []

    for dr in drs:
        bp = os.path.join("inputs", f"dr{dr}")
        results = dict()
        input_set = []

        for subdir in os.listdir(bp):
            d_path = os.path.join(bp, subdir)
            if not os.path.isdir(d_path):
                continue
            w_dir = os.path.join(d_path, "wrds")
            c_dir = os.path.join(d_path, "cgrams")

            for f in os.listdir(w_dir):
                fpath = os.path.join(w_dir, f)
                arr = read_wrd(fpath)
                target = overlap_wrd(arr, 16000)
                if target and check_bank(target, word_bank):
                    title = get_title(f)
                    wpath = os.path.join(c_dir, title + ".npy")
                    input_set.append((wpath, word_bank[target]))
            
        for i in input_set:
            c_path, target = i[0], i[1]
            cgram = np.load(c_path)
            logits = run_cgram(cgram, net_object)
            inds = distrib(logits, p_range)
            results = merge_inds(results, target, inds)

        ds.append(results)
    return ds, word_key

#uses same full bank (or random sel) to check every cgram
#returns array by dr of arrays containing all results in the form (target, result)
def full_input_test(drs, p_range, n = 587):
    net_object, word_key = load_model()
    word_bank = random_sel(word_key, n)
    ds = []

    for dr in drs:
        bp = os.path.join("inputs", f"dr{dr}")
        results = []
        input_set = []

        for subdir in os.listdir(bp):
            d_path = os.path.join(bp, subdir)
            if not os.path.isdir(d_path):
                continue
            w_dir = os.path.join(d_path, "wrds")
            c_dir = os.path.join(d_path, "cgrams")

            for f in os.listdir(w_dir):
                fpath = os.path.join(w_dir, f)
                arr = read_wrd(fpath)
                target = overlap_wrd(arr, 16000)
                if target and check_bank(target, word_bank):
                    title = get_title(f)
                    wpath = os.path.join(c_dir, title + ".npy")
                    input_set.append((wpath, word_bank[target]))
            
        for i in input_set:
            c_path, target = i[0], i[1]
            cgram = np.load(c_path)
            logits = run_cgram(cgram, net_object)
            inds = distrib(logits, p_range)
            results.append((target, inds))

        ds.append(results)
    return ds, word_key

#checks every gram, but includes adjacent words within start and end timestamps as targets
def adj_input_test(drs, p_range, start, end, n = 587):
    net_object, word_key = load_model()
    word_bank = random_sel(word_key, n)
    ds = []

    for dr in drs:
        bp = os.path.join("inputs", f"dr{dr}")
        results = []
        input_set = []

        for subdir in os.listdir(bp):
            d_path = os.path.join(bp, subdir)
            if not os.path.isdir(d_path):
                continue
            w_dir = os.path.join(d_path, "wrds")
            c_dir = os.path.join(d_path, "cgrams")

            for f in os.listdir(w_dir):
                fpath = os.path.join(w_dir, f)
                arr = read_wrd(fpath)
                target = wrd_range(arr, start, end, word_bank)
                mid = overlap_wrd(arr, 16000)
                if mid and check_bank(mid, word_bank):
                    #print(len(target))
                    if len(target) == 0:
                        print(mid)
                    title = get_title(f)
                    wpath = os.path.join(c_dir, title + ".npy")
                    tbank = []
                    for t in target:
                        tbank.append(word_bank[t])
                    input_set.append((wpath, tbank))
            
        for i in input_set:
            c_path, target = i[0], i[1]
            cgram = np.load(c_path)
            logits = run_cgram(cgram, net_object)
            inds = distrib(logits, p_range)
            results.append((target, inds))

        ds.append(results)
    return ds, word_key


#checks through all cuts, saves in word form
def cut_input_test(drs, p_range):
    net_object, word_key = load_model()
    word_bank = random_sel(word_key, 587)
    ds = []
    for dr in drs:
        bp = os.path.join("cuts", f"dr{dr}")
        results = []
        for subdir in os.listdir(bp):
            d_path = os.path.join(bp, subdir)
            if not os.path.isdir(d_path):
                continue
            for f in os.listdir(d_path):
                c_path = os.path.join(d_path, f)
                target = cut_target(f)
                cgram = np.load(c_path)
                logits = run_cgram(cgram, net_object)
                inds = distrib(logits, p_range)
                results.append((word_bank[target], inds))
        ds.append(results)
    return ds, word_key

def cut_test(drs, p_range):
    net_object, word_key = load_model()
    word_bank = random_sel(word_key, 587)
    inputs = []
    words = []
    for dr in drs:
        bp = os.path.join("cuts", f"dr{dr}")
        sub_in = []
        sub_words = dict()
        for subdir in os.listdir(bp):
            d_path = os.path.join(bp, subdir)
            if not os.path.isdir(d_path):
                continue
            for f in os.listdir(d_path):
                if f ==  '.DS_Store':
                    continue
                c_path = os.path.join(d_path, f)
                target = cut_target(f)
                cgram = np.load(c_path)
                logits = run_cgram(cgram, net_object)
                inds = distrib(logits, p_range)
                sub_words = merge_inds(sub_words, word_bank[target], inds)
                sub_in.append((word_bank[target], inds))
        words.append(sub_words)
        inputs.append(sub_in)
    return inputs, words, word_key

#checks if words is in the top n of a results array
def arr_top_n(word, arr, n):
    #print(word, arr)
    bound = min(n, len(arr))
    val = word in arr[:bound]
    #print(val)
    return val

def adj_top_n(words, arr, n):
    bound = min(n, len(arr))
    for w in words:
        if w in arr[:bound]:
            return True
    return False

#finds max result in results array
def arr_max(arr):
    return arr[0]

