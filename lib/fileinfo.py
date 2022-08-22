from lib.setup import *

#checks if a word is in the given word bank
def check_bank(word, bank):
    trans = str.maketrans('', '', string.punctuation)
    return word.lower().translate(trans).strip() in bank

#extracts full string, speaker detail string, title, and offset from reformatted
def extract_details(f):
    info = f[:f.index(".")]
    sp_des = info[:5]
    f = info[5:]
    title = f[:f.index("-")]
    offset = f[f.index("-") + 1 :]
    return sp_des, title, int(offset)

def timit_details(f):
    return f[:f.index(".")]

#return array of tuples with word and time range
def read_wrd(file):
    arr = []
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        segs = line.split(' ')
        t1, t2 = int(segs[0]), int(segs[1])
        wrd = segs[2].strip()
        arr.append((wrd, t1, t2))
    return arr

#return true if banked word in arr
def screen_wrd(arr, wb):
    for group in arr:
        word = group[0]
        if word in wb:
            return True
    return False

#returns word overlapping a given time in a clip, False if no word overlaps
def overlap_wrd(arr, time):
    for group in arr:
        t1 = group[1]
        if t1 <= time:
            t2 = group[2]
            if t2 >= time:
                trans = str.maketrans('', '', string.punctuation)
                return group[0].lower().translate(trans).strip()
        else:
            return False
    return False

def get_title(f):
    return f.split('.')[0]

def wrd_range(arr, start, end, wb):
    res = []
    trans = str.maketrans('', '', string.punctuation)
    for group in arr:
        #print(group)
        t1 = group[1]
        t2 = group[2]
        if t1 > end:
            return res
        if t2 >= start:
            word = group[0].lower().translate(trans).strip()
            if check_bank(word, wb):
                #print(True)
                res.append(word)
    return res

#returns a list of all banked words and their center point (for cutting)
def wrd_centers(arr, wb):
    trans = str.maketrans('', '', string.punctuation)
    centers = []
    for group in arr:
        word = group[0].lower().translate(trans).strip()
        if check_bank(word, wb):
            t1 = group[1]
            t2 = group[2]
            mid = (t1 + t2) // 2
            centers.append((word, mid))
    return centers

def cut_target(f):
    half = f.split("-")[1]
    return half.split(".")[0]