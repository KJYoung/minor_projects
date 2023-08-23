import re
import collections
import pandas as pd
from data.useless_words import useless_words


def split_sentence(sentence):
    return (x.lower() for x in re.split('\W+', sentence.strip()) if x)


def generate_vocabulary(train_captions):
    word_count = collections.Counter()

    for current_sentence in train_captions:
        word_count.update(split_sentence(str(current_sentence)))

    return {key: value for (key, value) in word_count.items() if value >= 3 and len(key) >= 3}


pd.options.display.max_colwidth = 200

# df = pd.read_excel('./data/CVPR2023.xlsx')
# df = pd.read_excel('./data/ECCV/eccv2022_combined.xlsx')
df = pd.read_excel('./data/RSS/RSS2023.xlsx')
title_str = df.get('Title').to_string(index=False)

text_name = './data/RSS/RSS2023.txt'
# 1. Read
with open(text_name, 'w') as f:
    f.write(title_str)

# 2. Write
with open(text_name, 'r') as f:
    strs = f.readlines()

vocab = generate_vocabulary(strs)

# CVPR 2023
# value >= 1 : 3797 words
# value >= 2 : 1520 words
# value >= 2 and len(key) >= 2 : 1503 words
# value >= 3 and len(key) >= 2 : 1074 words
# value >= 3 and len(key) >= 3 : 1045 words

# ECCV 2022
# value >= 3 and len(key) >= 3 : 755 words
# 731 after removing useless words

# RSS 2023
# value >= 3 and len(key) >= 3 : 81 words
# 72 after removing useless words

print("Value >= 3, Lenght >= 3", len(vocab))

for word in useless_words:
    if word in list(vocab.keys()):
        del vocab[word]

print("Useless words were removed", len(vocab))

vocab = dict(sorted(vocab.items(), key=lambda item: -item[1]))

# To Excel
series = pd.Series(vocab)
# series.to_excel('./data/VocabCVPR2023.xlsx')
# series.to_excel('./data/VocabECCV2022.xlsx')
series.to_excel('./data/VocabRSS2023.xlsx')

# print(" Successfully Saved!")
# Print
for item in vocab.items():
    print(item[0], end=' ')
