# Unzip the gzip file
import gzip
import shutil

with gzip.open('data/numberbatch-en-19.08.txt.gz', 'rb') as input:
    with open('data/numberbatch-en.txt','wb') as output:
        shutil.copyfileobj(input, output)