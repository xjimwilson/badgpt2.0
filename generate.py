import random, string
from collections import Counter
import shared, tfidf

maxwords = 100
matchesthreshold = 0.25
weightmult = 5 #boost on-topic words, 1 for no influence
p_boost = 0.7  # probability to sample from boosted list
responserange = 3

normaliseweights = True

sentinel = "<$>"

def writeontosentence(word, printword=True): #i am writing this function while drunk i think ive done a good job
    word = word.lower()
    if not shared.finalsentence: #why tf cant we just put = None bro i hate being punctual
        shared.finalsentence = [word]
        shared.wordcounter = 1
        if printword:
            print(word, end=" ")
        return

    if shared.finalsentence[-1][-1] in ['?', '!', '.', '|'] or shared.wordcounter == 1:
        word.capitalize()
    if printword:
        print(word,end=" ")
    shared.finalsentence.append(word)      
    shared.wordcounter += 1 # folk idk why i have this??? why not honestly

def choosefirstword(prompt): #each model should output firstword

    print() #makes new line for cosmetics innit

    closestreponseindexes, similarities = tfidf.findclosest(prompt, shared.vocab, shared.idf, shared.tfidfvectors, responserange)

    # Pick ONE random matched response instead of concatenating all 3
    chosenidx = random.choice(closestreponseindexes)
    responsewords = [w.lower() for w in shared.userresponses[chosenidx].split()]

    shared.finalsentence = []
    shared.wordcounter = 0

    # Always start from beginning of this response
    for w in responsewords[:3]:
        writeontosentence(w, printword=True)
    
    # extract prompt words too, before this it would just go off topic after like 4 words
    prompt_words = set(w.lower().strip(string.punctuation) for w in prompt.split())
    shared._current_response_wordset = set(responsewords) | prompt_words  #pipe combines in python

    shared._local_continuations = {}
    for a, b, c in zip(responsewords, responsewords[1:], responsewords[2:]):
        shared._local_continuations.setdefault((a, b), []).append(c)
    shared._seed_len = 3

    generate(shared.finalsentence[-1])

def findsuffixinterval(context): # basically just find the range of suffix array positions that match context
    
    tokens = shared.tokens
    positions = shared.suffixarray
    context_len = len(context)
   
    # binary search leftmost pos
    left = 0
    right = len(positions)
    while left < right:
        mid = (left + right) // 2
        pos = positions[mid]
        # Compare at position without slicing
        suffix = tokens[pos:pos + context_len]
        
        if suffix < context:
            left = mid + 1
        else:
            right = mid
    leftbound = left

    # binary search rightmost pos
    left = leftbound
    right = len(positions)
    while left < right:
        mid = (left + right) // 2
        pos = positions[mid]
        suffix = tokens[pos:pos + context_len]
        
        if suffix <= context:
            left = mid + 1
        else:
            right = mid
    rightbound = left

    return leftbound, rightbound


def nexttoken(): #find next token using suffix array (obviously using the longest context first)
    tokens = shared.tokens
    positions = shared.suffixarray
    finalwords = shared.finalsentence
    if not finalwords:
        return None
    
    for contextlen in range(len(finalwords), 0, -1):
        context = finalwords[-contextlen:]
        left, right = findsuffixinterval(context)
        
        if left < right:
            # collect tokens that follow context
            nexttokens = []
            for pos in positions[left:right]:
                nextpos = pos + contextlen
                if nextpos < len(tokens):
                    nextword = tokens[nextpos]
                    if nextword != sentinel:
                        nexttokens.append(nextword)
            
            #count frequencies
            if nexttokens:
                counts = Counter(nexttokens)
                return list(counts.items())  # return as [(word, freq), ...]
    
    return None
    
def wordweigh(matches):
    if not matches:
        return None
    
    # if only 1 match return immediately
    if len(matches) == 1:
        return matches[0][0]
    
    meanfreq = sum(freq for _, freq in matches) / len(matches)
    nextwords = []
    weights = []
    responseset = getattr(shared, "_current_response_wordset", set())
    for word, freq in matches:
        w = freq
        if word in responseset:
            w *= weightmult
            nextwords.append(word)
            weights.append(w)
            continue

        if freq >= meanfreq * matchesthreshold:
            nextwords.append(word)
            weights.append(w)
                
    if not nextwords:
        return None
    
    # Early exit: if only 1 candidate, skip weighting
    if len(nextwords) == 1:
        return nextwords[0]
    
    steps_since_seed = shared.wordcounter - getattr(shared, "_seed_len", 0)
    if steps_since_seed < 1:
        local = shared._local_continuations.get(tuple(shared.finalsentence[-2:]), [])
        for i, w in enumerate(nextwords):
            if nextwords[i] in local:
                weights[i] *= (1.0 + weightmult * 0.5)

    # Skip normalization if very few candidates
    if normaliseweights and len(nextwords) > 2:
        weights = [w / sum(weights) for w in weights]

    boostedpairs = [(w, wt) for w, wt in zip(nextwords, weights) if w in responseset]
    normalpairs = [(w, wt) for w, wt in zip(nextwords, weights) if w not in responseset]

    if boostedpairs and random.random() < p_boost:
        words, wts = zip(*boostedpairs)
        return random.choices(words, weights=wts)[0]
    elif nextwords:
        words, wts = zip(*normalpairs if normalpairs else boostedpairs)
        return random.choices(words, weights=wts)[0]

    return None

def generate(firstword):
    while True:
        # Stop at endoftext
        if shared.finalsentence[-1] == "" or shared.finalsentence[-1] == "<|eot|>":
            break
        
        # Stop at word limit + next punctuation
        if shared.wordcounter >= maxwords and shared.finalsentence[-1][-1] in ['?', '!', '.', '|']:
            break
        
        nextword = None
        suffixmatches = nexttoken()

        if suffixmatches:
            nextword = wordweigh(suffixmatches)

        if nextword is None: #no match found
            print("\nCould not find match for word. Ending generation...")
            return

        if nextword.lower() == "<|eot|>": #stop generation
            return
        
        writeontosentence(nextword, printword=True) # append final choice to sentence and allat