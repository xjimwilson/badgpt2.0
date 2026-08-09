import zstandard as zstd
import os, json
import shared

def zstpackage(folder):

    shared.filepath = f"datasets/{folder}"

    try:
        files = [f for f in os.listdir(shared.filepath) if f.endswith(".jsonl.zst")]
    except:
        return None
    
    print(f"Started merging files in {shared.filepath}...")

    with open(f"datasets/{folder}.jsonl", "w", encoding="utf-8") as out:
        for file in files:
            print(f"Merging {file}...")
            path = os.path.join(shared.filepath, file)

            with open(path, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                stream = dctx.stream_reader(fh)

                for line in stream.read().splitlines():
                    out.write(line.decode("utf-8") + "\n")
    
    return shared.filepath