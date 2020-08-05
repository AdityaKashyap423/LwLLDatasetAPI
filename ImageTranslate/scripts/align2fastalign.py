import os
import sys

r1 = open(os.path.abspath(sys.argv[1]), 'r')
r2 = open(os.path.abspath(sys.argv[2]), 'r')
w = open(os.path.abspath(sys.argv[3]), 'w')

l1 = r1.readline()
count = 0
while l1:
    l2 = r2.readline()
    if len(l1.strip())>1 and len(l2.strip().lower())>1:
        w.write(l1.strip().lower() + ' ||| ' + l2.strip().lower() + '\n')
    else:
        print("\nSkipped", count+1, l1.strip().lower() + ' ||| ' + l2.strip().lower())
    l1 = r1.readline()
    count+=1
    if count %10000==0:
        print(count, end="\r")
w.close()
print('\nDone')
