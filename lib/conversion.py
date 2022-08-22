from lib.setup import *
from lib.editing import *
from lib.fileinfo import *

#Converts SPH to WAV in TEST and TRAIN (shouldn't need again)
def dr_conv(dr):
    #Used to convert SPH files to wavs in each subdirectory in training dataset
    path = "./TIMIT/TEST/" + f"DR{dr}"
    for subdir in os.listdir(path):
        new_path = os.path.join(path, subdir)
        if os.path.isdir(new_path):
            wavpath = new_path + '/wavs'
            os.mkdir(wavpath)
            for filename in os.listdir(new_path):
                if filename.lower().endswith('.wav'):
                    audiopath = os.path.join(new_path, filename)
                    sph = SPHFile(audiopath)
                    sph.write_wav(os.path.join(new_path + '/wavs', filename))

#takes all inputs in a certain dr and generates cgrams with the desired offset #REWRITE FOR WRD DATA
def reform_data(s, dr, offset):
    print(f"Reforming dr{dr} with offset {offset}.")
    np.seterr(divide = 'ignore')
    dest = os.path.join("graph_data", f"dr{dr}")
    if not os.path.isdir(dest):
        os.mkdir(dest)
    a_dir = os.path.join(dest, "wavs")
    t_dir = os.path.join(dest, "txts")
    c_dir = os.path.join(dest, "cgrams")
    if not os.path.isdir(a_dir):
        os.mkdir(a_dir)
    if not os.path.isdir(t_dir):
        os.mkdir(t_dir)
    if not os.path.isdir(c_dir):
        os.mkdir(c_dir)
    o_dir = os.path.join(c_dir, f"{offset}")
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)
    dr_path = os.path.join(f"./TIMIT/{s}", f"DR{dr}")
    word_bank = build_word_bank()

    sdc = 0

    for sd in os.listdir(dr_path):
        base_path = os.path.join(dr_path, sd)
        if os.path.isdir(base_path):
            sdc += 1
            print(f"Entering subdirectory number {sdc}; {sd}.")
            full_path = os.path.join(base_path, 'wavs')
            for f in os.listdir(full_path):
                frag = f[:f.index(".")]
                title = sd + frag

                audiopath = os.path.join(full_path, f)
                textpath = os.path.join(base_path, frag + '.txt')

                #check validity
                with open(textpath, encoding = 'us-ascii') as t:
                    tscript = t.readlines()[0]
                tscript = tscript.split(' ')[2:]
                trans = str.maketrans('', '', string.punctuation)
                word_set = set()

                for i in range(len(tscript)):
                    x = tscript[i]
                    basic = x.lower().translate(trans).strip()
                    if check_bank(basic, word_bank):
                        word_set.add(basic)
                        
                if len(word_set) == 0:
                            #print(f'No banked words for {title}.')
                            continue
                

                #print(f"{word_set} banked for {title}")

                sr, wav_f = wav.read(audiopath)
                wav_f = process_wav(sr, wav_f, offset)
                c_gram = generate_cochleagram(wav_f, sr)
                fname = os.path.join(a_dir, f"{title}.wav")
                if not os.path.isfile(fname):
                    shutil.copy(audiopath, fname)
                tname = os.path.join(t_dir, f"{title}.txt")
                if not os.path.isfile(tname):
                    shutil.copy(textpath, tname)
                cname = os.path.join(o_dir, f"{title}-{offset}.npy")
                np.save(cname, c_gram)

#checks if a directory exists and creates it if not
def check_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

#converts all preformatted cgrams to new accepted format
def txt_to_wrd():
    b_dest = 'inputs'
    src = './graph_data'
    for dr in os.listdir(src):
        dr_dest = os.path.join(b_dest, dr)
        check_mkdir(dr_dest)
        dr_path = os.path.join(src, dr)
        if os.path.isdir(dr_path):
            c_dir = os.path.join(os.path.join(dr_path, 'cgrams'), '0')
            for cgram in os.listdir(c_dir):
                cfile = os.path.join(c_dir, cgram)
                sp_des, title, o = extract_details(cgram)
                p1 = os.path.join(f"./TIMIT/TRAIN", f"{dr}")
                p2 = os.path.join(f"./TIMIT/TEST", f"{dr}")
                p1 = os.path.join(p1, sp_des)
                p2 = os.path.join(p2, sp_des)
                if os.path.isdir(p1):
                    wrd = os.path.join(p1, title) + '.WRD'
                else:
                    wrd = os.path.join(p2, title) + '.WRD'
                #now add as desired
                dest_dir = os.path.join(dr_dest, sp_des)
                check_mkdir(dest_dir)
                c_dest = os.path.join(dest_dir, 'cgrams')
                w_dest = os.path.join(dest_dir, 'wrds')
                check_mkdir(c_dest)
                check_mkdir(w_dest)
                c_path = os.path.join(c_dest, f'{title}.npy')
                w_path = os.path.join(w_dest, f'{title}.wrd')
                shutil.copy(cfile, c_path)
                shutil.copy(wrd, w_path)

#generates all possible inputs for the given dr
def cut_to_center(dr):
    wb = build_word_bank()
    b_dest = 'cuts'
    test_src = f"./TIMIT/TEST/DR{dr}"
    train_src = f"./TIMIT/TRAIN/DR{dr}"
    for src in [test_src, train_src]:
        print(f"In source {src}")
        for subset in os.listdir(src):
            s_path = os.path.join(src, subset)
            if not os.path.isdir(s_path):
                continue
            print(f"In subset {subset}")
            count = 0
            w_path = os.path.join(s_path, "wavs")
            for f in os.listdir(w_path):
                #print(wav)
                title = timit_details(f)
                #print(title)
                wav_path = os.path.join(w_path, f)
                wrd_path = os.path.join(s_path, title + ".wrd")
                arr = read_wrd(wrd_path)
                centers = wrd_centers(arr, wb)

                sr, wav_f = wav.read(wav_path)
                for c in centers:
                    target = c[0]
                    mid = c[1]
                    offset = mid - 16000
                    wf = process_wav(sr, wav_f, offset)
                    c_gram = generate_cochleagram(wf, sr)
                    c_dr = os.path.join(b_dest, f"dr{dr}")
                    check_mkdir(c_dr)
                    c_sub = os.path.join(c_dr, subset)
                    check_mkdir(c_sub)
                    cname = os.path.join(c_sub, f"{title}-{target}.npy")
                    np.save(cname, c_gram)
                    count += 1
            print(f"Generated {count} cgrams.")
