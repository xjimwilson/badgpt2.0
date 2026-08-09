import json
import shared

#what the columns are named in their json file
inputcol = "question"
outputcol = "answer"

jsontype = "jsonl"

def convjson(filename):
    if jsontype == "json":
        print("Processing JSON file...")

        if ".json" not in filename:
            filename += ".json"

        with open(f"datasets/{filename}", "r", encoding="utf-8") as file:
            data = json.load(file)

    elif jsontype == "jsonl":
        print("Processing JSONL file...")

        if ".jsonl" not in filename:
            filename += ".jsonl"

        data = []
        with open(f"datasets/{filename}","r", encoding="utf-8") as file:
            for line in file:
                if line.strip(): #skip blanks
                    data.append(json.loads(line))
        
    else:
        print("jsontype does not exist")
        return

    for item in data:
        if "messages" in item:
            q_raw = a_raw = None
            for msg in item["messages"]:
                if msg.get("role") == "user":
                    q_raw = msg.get("content")
                elif msg.get("role") == "assistant":
                    a_raw = msg.get("content")
        else:
            q_raw = item.get(inputcol)
            a_raw = item.get(outputcol)

        q = q_raw.strip() if isinstance(q_raw, str) else ""
        a = a_raw.strip() if isinstance(a_raw, str) else ""

        shared.userquestions.append(q)
        shared.userresponses.append(a)

        block = f"{a}\n<|EOT|>\n"
        shared.responseblocks.append(block)

    print("Successfully processed JSON file!")