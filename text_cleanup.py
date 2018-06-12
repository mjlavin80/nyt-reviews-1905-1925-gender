#/usr/bin/env python3

# This block of code assumes a folder called ocr with .txt files in it. 
# To run the code you should build your own ocr corpus or remake mine by downloading all pdfs and running OCR software on them
# Each .txt file should be named after its NYT ID (e.g. 4fc03b9245c1498b0d1e86a2.txt)
# The code will loop through the folder and perform various cleanup operations, including lemmatization and 
# Running this code snippet will produce a folder of csv files with lemmas and counts (see the lemma-data folder) 

import glob
import spacy
import enchant
from collections import Counter
import pandas as pd 


# load spacy model for english
nlp = spacy.load('en')

# define functions
def spellcheck(wordlist):
    result = []
    d = enchant.Dict("en_US")
    for i in wordlist:
        if d.check(i) or d.check(i.capitalize()):
            result.append(i)
    return result

def clean_text(list_of_texts):
    fully_cleaned =[]
    #normalize ocr errors
    for i in list_of_texts:
        #lowercase all
        ocr_lower = i.lower()
        #tokenize, remove punctuation and numbers, remove tabs, newlines, etc.
        ocr_cleaner = ocr_lower.replace("\n", " ").replace("\t", " ")
        doc = nlp(ocr_cleaner)
        ocr_tokens = []
        for token in doc:
            
            if token.lemma_ == u'-PRON-' or token.lemma_.isupper():
                ocr_tokens.append(unicode(token))
            else:
                ocr_tokens.append(token.lemma_)
        #ocr_tokens = ocr_cleaner.split(" ")
        
        no_numbers_or_punct = []
        for token in ocr_tokens:
            if token.isalpha():
                no_numbers_or_punct.append(token)
            else:
                
                new_token = ""
                for letter in token:
                    if letter.isalpha():
                        new_token += letter
                if new_token != "":
                    no_numbers_or_punct.append(new_token)  
        
        
        spellchecked = spellcheck(no_numbers_or_punct)
        fully_cleaned.append(spellchecked)
    return fully_cleaned


# loop files
all_text = glob.glob('ocr/*.txt')

ocr_text_raw = []
for one_textfile in all_text:
	with open(one_textfile) as f:
		one_raw = f.read()
	ocr_text_raw.append(one_raw)	

# this operation will produce a list of cleaned lists in the same order as all_text filenames
ocr_cleaned = clean_text(ocr_text_raw)

# this operation will convert each sublist into a dictionary of lemmas and counts
ocr_counters = [Counter(i) for i in ocr_cleaned]

# this operation will loop the dictionaries of lemmas and counts and save csv files using names that match the all_text filenames
for h,i in enumerate(all_text):
    new_filename = i.replace("ocr/", "lemma-data/").replace(".txt", ".csv")
    rows = list(ocr_counters[h].items())
    df = pd.DataFrame.from_records(rows, columns=['lemma', 'count']).sort_values(by="count", ascending=False)
    df.to_csv(new_filename, index=False)