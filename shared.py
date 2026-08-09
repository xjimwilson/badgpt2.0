# i know its not the best coding practices to have shared variables but who gaf

workercount = 3

wordcounter = 0

finalsentence = [] #stores generated data

#stores trained data
userquestions = []
userresponses = []

responseblocks = []

tokens = []
suffixarray = []
suffixn = 0

vocab = ""
idf = ""
tfidfvectors = ""

filepath = ""


"""
One day i found out about dictionaries and realised i could use it for data. Switching from searching the database with a for loop
to a dictionary was probably the biggest scientific revelation since the vaccine. Generation times went from 10 mins to less than 0.1
of a second. Thank you so much dictionaries.
"""