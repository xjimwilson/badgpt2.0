import saveloadfiles, train, shared, generate, zstconverter


text, choice, loaded, packaged = None, None, None, None

if __name__ == "__main__": # prevents pickles re-running top level script

    def calcsize(bytesize):
        convsize = bytesize, "bytes"
        if bytesize >= 1000: #1kb:
            convsize = f"{int(bytesize / 1000)} KB"
        if bytesize >= 1000000: #1mb:
            convsize = f"{int(bytesize / 1000000)} MB"
        if bytesize >= 1000000000: #1gb:
            convsize = f"{int(bytesize / 1000000000)} GB"
        return convsize

    while choice != 't' and choice != 'g' and choice != 'z':
        choice = input("Enter T to train, or G to generate:\n").lower()
        
    if choice == 't':
        fileinput = str(input("Enter file name:\n"))
        saveloadfiles.readfile(fileinput)
        
        print(f"Training on {fileinput}...")
        train.buildsuffixarray()
        train.generatetfidf()
        
        print("Saving to .npz file...")
        saveloadfiles.savefile(fileinput)

        if fileinput == "":
            fileinput = "fulltrained"
        fileinput.replace("/","")

        print(f"Successfully trained! Saved knowledge in memory/{fileinput}.npz, with size of {calcsize(saveloadfiles.getfilesize())}")

    elif choice == 'g':
        while loaded == None:
            file = str(input("Enter the file to load:\n"))
            loaded = saveloadfiles.loadfile(file)

        print("File successfully loaded!")

        while True:
            prompt = input("\n")

            generate.choosefirstword(prompt) #starts generation

            print("\n\nWords:", str(shared.wordcounter).replace("\\n","\n"))

    elif choice == 'z':
        while packaged == None:
            folder = str(input("Enter the folder to package:\n"))
            packaged = zstconverter.zstpackage(folder)
        print(f"Successfully packaged! Saved as {packaged}.jsonl")


            