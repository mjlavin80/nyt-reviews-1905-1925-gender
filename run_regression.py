#/usr/bin/env python3

# This block of code will run various logistic regression scenarios and store results in an sqlite database called 
# regression_scores.db. 

# import modules
import pandas as pd
import pickle
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np 
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import sqlite3

# local library functions
from application.selective_features import dictionaries_without_features, dictionaries_of_features
from application.regression import vectorize_and_predict
from application.regression import ocr_dicts_all, nyt_ids_all, gender_label_stats, label_cutoff


conn = sqlite3.connect('regression_scores.db')
c = conn.cursor()

#load feature set pickles

with open('pickled-data/fullstops.pickle', 'rb') as handle2:
    fullstops = pickle.load(handle2)
with open('pickled-data/fullstops_and_pronouns.pickle', 'rb') as handle4:
    fullstops_and_pronouns = pickle.load(handle4)
with open('pickled-data/fullstops_pronouns_and_names.pickle', 'rb') as handle5:
    fullstops_pronouns_and_names = pickle.load(handle5)

ocr_dicts_no_stops = dictionaries_without_features(ocr_dicts_all, fullstops)

ocr_dicts_no_stops_or_pro = dictionaries_without_features(ocr_dicts_all, fullstops_and_pronouns)

# reopen source pickle to avoid shallow copy error
with open('pickled-data/ocr_dicts_all.pickle', 'rb') as handle:
    ocr_dicts_all = pickle.load(handle)

ocr_dicts_no_stops_pro_names = dictionaries_without_features(ocr_dicts_all, fullstops_pronouns_and_names)

# reopen source pickle to avoid shallow copy error
with open('pickled-data/ocr_dicts_all.pickle', 'rb') as handle:
    ocr_dicts_all = pickle.load(handle)

create_main = """CREATE TABLE IF NOT EXISTS main (id INTEGER PRIMARY KEY, features_used TEXT, words_removed TEXT, 
f1_score_f REAL, precision_f REAL, recall_f REAL, f1_score_m REAL, precision_m REAL, recall_m REAL, accuracy REAL,
male_match INTEGER, male_mismatch INTEGER, female_match INTEGER, female_mismatch INTEGER, train_size INTEGER, 
test_size INTEGER, random_seed INTEGER, test_ids TEXT, train_ids TEXT)"""

create_results = """CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, main_id INTEGER, nyt_id TEXT, 
predicted_gender INTEGER, labeled_gender INTEGER, probability_male REAL, probability_female REAL, FOREIGN KEY(main_id) 
REFERENCES main(id))"""

create_coef = """CREATE TABLE IF NOT EXISTS coefficients (id INTEGER PRIMARY KEY, main_id INTEGER, feature TEXT, 
score REAL, odds REAL, FOREIGN KEY(main_id) REFERENCES main(id))"""

# CREATE INDICES
coef_index = """CREATE INDEX IF NOT EXISTS coef_index ON coefficients (id,main_id,feature,score,odds)"""
result_index = """CREATE INDEX IF NOT EXISTS result_index ON results (id,main_id,nyt_id,predicted_gender,
labeled_gender,probability_male, probability_female)"""

c.execute(create_main)
c.execute(create_results)
c.execute(create_coef)
c.execute(coef_index)
c.execute(result_index)

cw = (gender_label_stats['male']*1.0)/(gender_label_stats['male']+gender_label_stats['female']) 

print(cw)

train_size_int = 1354
test_size_int = 400

# nothing removed
vectorize_and_predict(ocr_dicts_all, c, conn, {'features_used':'tfidf_full_text_lemmas', 'stopwords':'nothing_removed'}, cw, label_cutoff, train_size_int, test_size_int, 'random')

# stopwords only removed
vectorize_and_predict(ocr_dicts_no_stops, c, conn, {'features_used':'tfidf_full_text_lemmas', 'stopwords':'stopwords_only'}, cw, label_cutoff, train_size_int, test_size_int, 'random')

# stopwords and pronouns removed
vectorize_and_predict(ocr_dicts_no_stops_or_pro, c, conn, {'features_used':'tfidf_full_text_lemmas', 'stopwords':'stopwords_and_pronouns'}, cw, label_cutoff, train_size_int, test_size_int, 'random')

# stopwords and pronouns and proper names removed
vectorize_and_predict(ocr_dicts_no_stops_pro_names, c, conn, {'features_used':'tfidf_full_text_lemmas', 'stopwords':'stopwords_pronouns_and_names'}, cw, label_cutoff, train_size_int, test_size_int, 'random')
