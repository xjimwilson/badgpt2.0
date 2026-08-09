import os
import numpy as np
import shared, train, jsontotext


def readfile(file):
    return jsontotext.convjson(file)

def savefile(file):
    if file == "": file = "fulltrained" #defaults to fulltrained.npy when empty
    
    else:
        file = file.replace(".txt",'')

        shared.filepath = f"memory/{file}"

        tfidfarr = np.array(train.tfidfvectors, dtype=object) # converts to 1d object array to avoid 0d error

        np.savez_compressed(str(shared.filepath),
            tokens = np.array(train.shared.tokens, dtype=object),
            suffixarray = np.array(train.shared.suffixarray, dtype=np.int32),
            vocab = np.array(train.vocab, dtype=object),
            idf = train.idf,
            tfidfvectors = tfidfarr,
            userquestions = np.array(shared.userquestions, dtype=object),
            userresponses = np.array(getattr(shared, "userresponses", []), dtype=object))
        
        #saves as .npz (finding out compression was like discovering fire)

def loadfile(file):
    global result

    if ".npz" not in file:
        file = file + ".npz" # just in case user does not input .npz

    if file == "":
        file = "fulltrained"

    try:
        data = np.load(f"memory/{file}", allow_pickle=True) # pickle fing

        # load into shared variables while converting to lists to avoid numpy type issues

        shared.suffixarray = list(data["suffixarray"])
        print("Loaded suffix array")
        shared.tokens = list(data["tokens"])
        print("Loaded", len(shared.tokens), "tokens")

        shared.suffixn = len(shared.tokens)

        #load everything else associated with tf-idf
        print("loading tf-idf variables and parameters...")

        shared.userquestions = data["userquestions"]
        shared.userresponses = data["userresponses"]

        shared.vocab = data["vocab"]
        shared.idf = data["idf"]

        arr = data["tfidfvectors"]
        # handle 0-D object array (single object) and 1-D object array
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            if arr.ndim == 0:
                # single object stored inside a 0-D array (older/accidental saves)
                tflist = arr.item()
            else:
                # normal 1-D object array
                tflist = arr.tolist()
        else:
            # fallback: try to iterate (covers numeric arrays or other shapes)
            try:
                tflist = [np.array(row) for row in arr]
            except Exception:
                tflist = []

        # assign into shared; keep original type (list of sparse matrices or arrays)
        shared.tfidfvectors = tflist

        data.close() #closes afterwards to free up memory

        print("Sucessfully loaded", file)
        return True #signals it went alright

    except Exception as e:
        print(f"Could not load {file}, {e}")
        return #returns nothing so it stays as None


def getfilesize():
    return os.path.getsize(f"{shared.filepath}.npz")