import re, multiprocessing, os

import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="cupy") #keeps showing cuda warning but its working just fine so idk just ignore

import cupy as cp # genuinely awful abbreviation
from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
from cupyx.scipy.sparse import vstack as gpu_vstack
#got a beefier gpu last week, so i'm switching from numpy to cupy
#switching over was not as easy as they make it sound, especially when you're trying to get fast training

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack # i imported scipy's sparse matrix helper tho, i was getting desperate
import shared

# i am fully aware i can use scikit-learn's tfidfvectorizer import, but i want to make it myself for a bit of pain

workervocab = None # using globals to avoid pickle passing overhead
workeridf = None

def initworker(vocab, idf):
    warnings.filterwarnings("ignore", category=UserWarning, module="cupy") # gotta also ignore when every worker is created fml
    global workervocab, workeridf
    workervocab = vocab
    workeridf = idf

def tokenize(text): #cleanse text and split into tuples
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def buildvocab(tokenlists): #create list of unique words
    return list(set().union(*tokenlists)) # avoids re-tokenizing by passing tokenlists

def gettermfrequency(tokens, generate):
    global workervocab
    
    counts = Counter(tokens)

    if generate:
        workervocab = shared.vocab

    return cp.array([counts.get(word, 0) for word in workervocab]) #i dont know what this does anymore but its faster ig

def getidf(alltokens, vocab): #inverse document frequency, not israel defense force
    n = len(alltokens)
    dfarray = cp.zeros(len(vocab))
    dfdict = Counter()
    
    for tokens in alltokens:
        for word in set(tokens):
            dfdict[word] += 1
    
    dfarray = cp.array([dfdict.get(word, 0) for word in vocab])
    return cp.log(n / (1 + dfarray))

def combinetfidf(indexandtokens):
    index, tokens = indexandtokens
    
    tf = gettermfrequency(tokens, False)

    vec = tf * workeridf
    sparsevec = gpu_csr_matrix(vec)
    
    return index, sparsevec

def cosinesimilarity(a, b):
    dot = cp.dot(a, b) # i <3 numpy dot products it's just so clean and attractive
    norm = cp.linalg.norm(a) * cp.linalg.norm(b)
    if norm == 0:
        return 0
    return dot / norm

def buildtfidf(dataset):
    global vocab, idf

    print("Tokenizing data for vectors...")

    with multiprocessing.Pool(processes=shared.workercount) as pool:
        tokenlists = list(pool.imap_unordered(tokenize, dataset, chunksize=200))#using imap_unordered rather than map, quicker tokenisation

    print("Building vocab...")
    vocab = buildvocab(tokenlists)

    print("Generating inverse document frequencies...")    
    idf = getidf(tokenlists, vocab)
    
    print("Building TF-IDF vectors (takes forever)...")

    args = [(i, tokenlists[i]) for i in range(len(tokenlists))]
    results = [None] * len(tokenlists) # collect results

    with multiprocessing.Pool(processes=shared.workercount, initializer=initworker, initargs=(vocab, idf)) as pool:
        for index, vec in pool.imap_unordered(combinetfidf, args, chunksize=200):
            results[index] = vec

    print("Stacking TF-IDF matrix...")
    tfidfmatrix = gpu_vstack(results)

    return vocab, idf, tfidfmatrix

def findclosest(prompt, vocab, idf, tfidfmatrix, responserange):
    promptvec = cp.array(gettermfrequency(tokenize(prompt), True)) * cp.asarray(idf)
    promptsparse = gpu_csr_matrix(promptvec.reshape(1, -1)) #reshape because cp just cooked the matrix shape

    prod = promptsparse.dot(tfidfmatrix.T)
    similarities = cp.asarray(prod.toarray()).ravel()

    # norms: sqrt of sum of squares per row of the sparse tfidf matrix
    norms = cp.sqrt(cp.asarray(tfidfmatrix.multiply(tfidfmatrix).sum(axis=1)).ravel())
    
    similarities = cp.asnumpy(similarities / (cp.linalg.norm(promptvec) * norms + 1e-10))
    closestresponses = cp.argsort(similarities)[-(responserange):][::-1]
    
    return closestresponses.astype(int), similarities
