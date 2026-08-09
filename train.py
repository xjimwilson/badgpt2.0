import re
import tfidf, shared

#rewrote basically everything after moving to infinigram

textend = [""]
sentinel = "<$>" # end of ALL suffixes (holy aura)

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s<>|.!?,;:\-'\"()]", "", text)
    return text.split()



def buildsuffixarray(): #!!!please excuse these comments i put for this part, i won't remember what i did here in a weeks time so i need babying

    #BASICALLY:
    # rank positions by first token (2 positions with "hello" get same rank, 2 with "world" get a diff rank)
    # double window, compare pairs (firsttoken, secondtoken) so more positions are uniquely ranked
    # double until every pos has a unique rank (fully sorted)
    # gawd i want fried chicken so bad rn

    print("Tokenising data...")

    corpus = " ".join(shared.responseblocks)
    tokens = tokenize(corpus)
    tokens.append(sentinel)

    n = len(tokens)

    #numer 1: start by ranking the first token at each pos
    positions = list(range(n))
    ranks = tokens[:] + [None] # you can actually put None at the end so you don't need to dick about with issues at the end


    #nuby 2: double the comparison window until all suffixes are uniquely sorted

    print("Starting arrangement with comparison window of 1...")
    compwindow = 1
    while True:
        # sort positions by comparing current token and next token window (first 2, then first 4, 8, 16, etc.)
        positions.sort(key=lambda i: (ranks[i], ranks[i + compwindow] if i + compwindow < n else None))

        # nubrub 3: assign
    
        newranks = [0] * n
        newranks[positions[0]] = 0

        for i in range(1,n): #WAIT THIS IS JUST A SORTING ALGORITHM I'VE BEEN SCAMMED
            prevposition = positions[i-1]
            currposition = positions[i]

            prevpair = (ranks[prevposition], ranks[prevposition + compwindow] if prevposition + compwindow < n else None)
            currpair = (ranks[currposition], ranks[currposition + compwindow] if currposition + compwindow < n else None)

            # if pair is diff from previous, rank higher
            if prevpair != currpair:
                newranks[currposition] = newranks[prevposition] + 1
            else:
                newranks[currposition] = newranks[prevposition]

        ranks = newranks

        #bunny 4: stop when suffixes have unique ranks (sorted fing)
        if ranks[positions[-1]] == n - 1:
            break

        compwindow *= 2
        print(f"Arrangement layer complete, re-arranging with comparison window of {compwindow}...")
    
    shared.tokens = tokens
    shared.suffixarray = positions
    shared.suffixn = n
    
def generatetfidf():
    global vocab, idf, tfidfvectors

    vocab, idf, tfidfvectors = tfidf.buildtfidf(shared.userquestions) #only makes vectors for userquestions