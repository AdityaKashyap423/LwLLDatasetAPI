import os
import sys

input_path = os.path.abspath(sys.argv[1])
output_path1 = os.path.abspath(sys.argv[2])
output_path2 = os.path.abspath(sys.argv[3])
skip_titles = len(sys.argv) > 4 and sys.argv[4] == "skip"

skipped = 0
with open(input_path, "r") as reader, open(output_path1, "w") as w1, open(output_path2, "w") as w2:
    for line in reader:
        spl = line.strip().split(" ||| ")
        if len(spl) < 2: continue
        if len(spl) == 2 and skip_titles:
            skipped += 1
            continue
        for j in range(1, len(spl)):
            w1.write(spl[0] + "\n")
            w2.write(spl[j].strip() + "\n")
print("skipped", skipped)
