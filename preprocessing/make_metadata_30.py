import pandas as pd 

infile = '/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test.csv'
outfile = '/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test_30.csv'

with open('eval30list.txt', 'r') as f:
    eval30 = f.read().splitlines()

keep_lines = []
with open(infile, 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        audio = line.split('|')[0] + '.wav'
        if audio in eval30:
            keep_lines.append(line)
print(len(keep_lines))
print(*keep_lines, file=open(outfile, 'w'), sep='\n')