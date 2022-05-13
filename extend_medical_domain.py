import csv
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from matplotlib import pyplot as plt
import json
import os
import numpy as np

# Create domain-specific vocabulary
doc = {}
vocab = set()

# Build documents
with open('./datasets/n2c2_datasets/train_sentences.txt') as f:
    rd = csv.reader(f)
    for r in rd:
        doc.setdefault(r[0], list()).append(','.join(r[2:]))
with open('./datasets/n2c2_datasets/test_sentences.txt') as f:
    rd = csv.reader(f)
    for r in rd:
        doc.setdefault(r[0], list()).append(','.join(r[2:]))

# Extract medical terms
with open('./datasets/n2c2_datasets/synthetic_n2c2_datasets/train_w_to_idx.txt') as f:
    rd = csv.reader(f)
    for r in rd:
        vocab.add(r[0])
with open('./datasets/n2c2_datasets/synthetic_n2c2_datasets/test_w_to_idx.txt') as f:
    rd = csv.reader(f)
    for r in rd:
        vocab.add(r[0])

notes = [' '.join(d) for d in doc.values()]
print(f"Number of notes: {len(notes)}")
print(f"Number of words in the vocabulary: {len(vocab)}")
vectorizer = CountVectorizer(vocabulary=vocab,
                             lowercase=False,
                             analyzer='word',
                             tokenizer=lambda x: x.split(' '))
idf_count = vectorizer.fit_transform(notes)

# Compute inverse document frequency
# (the lower the score the more frequent the term)
# idf = ln(n/df) + 1
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(idf_count)

plt.hist(transformer.idf_, bins=50)
plt.title(f"Inverse document frequency distribution of the {len(vocab)} terms")
plt.show()

print(f"Dropping words with IDF>={np.log(len(notes) / 10) + 1}, i.e., words that occur in less than 10 notes")
thrsh = np.log(len(notes) / 10) + 1
medical_terms_idf = {w: idf for w, idf in zip(vectorizer.vocabulary_, transformer.idf_) if idf <= thrsh}
print(f"Dropped {len(vectorizer.vocabulary_) - len(medical_terms_idf)} words")
print(f"|V| = {len(medical_terms_idf)}")

print("Extend ClinicalBert vocabulary")
tokenizer = AutoTokenizer.from_pretrained('./models/pretrained_tokenizer/clinicalBERT')
print(f"Old vocabulary size {len(tokenizer.vocab)}")
print(f"Adding {len(medical_terms_idf) - len(set(medical_terms_idf.keys()).intersection(tokenizer.vocab))} tokens")
tokenizer.add_tokens(list(medical_terms_idf.keys()))
print("Adding [DATE] and [TIME] special tokens")
tokenizer.add_tokens(["[DATE]", "[TIME]"], special_tokens=True)
print(f"New vocabulary size {len(tokenizer.vocab)}")
tokenizer.save_pretrained('./models/pretrained_tokenizer/clinicalBERTmed')

# Extract added tokens
with open('./models/pretrained_tokenizer/clinicalBERTmed/added_tokens.json', 'r') as json_file:
    added_tokens = json.load(json_file)
    sorted_tokens = dict(sorted(added_tokens.items(), key=lambda item: item[1]))
# Add tokens to vocabulary to speed up loading of the tokenizer
with open('./models/pretrained_tokenizer/clinicalBERTmed/vocab.txt', 'a') as f:
    for w in sorted_tokens.keys():
        f.writelines(w + '\n')
# Save new vocabulary to file
print("Saving new words to file...")
with open('./models/pretrained_tokenizer/clinicalBERTmed/new_vocab_list.txt', 'w') as f:
    for w in sorted_tokens.keys():
        f.writelines(w + '\n')
os.remove('./models/pretrained_tokenizer/clinicalBERTmed/added_tokens.json')

# Run:
