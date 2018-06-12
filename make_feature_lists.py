#/usr/bin/env python3

# This block of code will make feature lists needed to run various logistic regression scenarios 
# and store those feature lists as python pickles. The regression script uses these pickles to construct vector space models 

# import libraries
import requests, pickle
from nltk.corpus import names 
from nltk.corpus import stopwords
from string import ascii_lowercase

# make list of combined stopwords
words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words')
stoplist1 = words.text.split("\r\n")
stoplist2 = set(stopwords.words('english'))
stoplist1.extend(stoplist2)

fullstops = list(set(stoplist1))
fullstops = [i for i in fullstops if i !='']

# this list of gender terms was generated iteratively by running the logistic regression with all terms, 
# seeing what correlated the most with gender, and removing words that seemed to have direct gender info in them

gender_terms = ["mr", "he", "his", "him", "himself", "man", "men", "boy", "boys", "manly", "masculine", "boyish", "father", \
                "brother", "girls", "men", "women", "sisters", "daughters", "brothers", "sons", "wife", "husband", "niece",\
                "uncle", "nephew", "dad", "grandfather", "son", "mrs", "miss", "her", "hers", "she", "herself", "woman",\
                "girl", "nieces", "nephews", "fer", "mme", "mlle", "lady", "womanly", "girlish", "girly", "mother", "daughter", \
                "aunt", "niece" "grandmother", "mom", "sister" ]

male = [o.lower() for o in names.words('male.txt')]
female = [o.lower() for o in names.words('female.txt')]

fullstops_and_pronouns = []

for u in [fullstops, gender_terms]:
    for i in u:
        fullstops_and_pronouns.append(i)

fullstops_and_pronouns = list(set(fullstops_and_pronouns))

for u in [fullstops, gender_terms]:
    for i in u:
        fullstops_and_pronouns.append(i)

fullstops_and_pronouns = list(set(fullstops_and_pronouns))

fullstops_pronouns_and_names = []

for u in [fullstops_and_pronouns, male, female]:
    for i in u:
        fullstops_pronouns_and_names.append(i)

# this instance of hand correction is applied to correct for the frequency of a unepxectedly common OCR error in this corpus
# in most cases, the text cleanup function removes errors like these by dropping non-dictionary words, but Thoma is a word in  
# enchant's US English dictionary (pertaining to Richard Thoma, a German histologist)

fullstops_pronouns_and_names.append("thoma")

for ltr in ascii_lowercase:
    fullstops_pronouns_and_names.append(ltr)

fullstops_pronouns_and_names = list(set(fullstops_pronouns_and_names))

with open('pickled-data/fullstops.pickle', 'wb') as handle2:
    pickle.dump(fullstops, handle2, protocol=pickle.HIGHEST_PROTOCOL)
with open('pickled-data/fullstops_and_pronouns.pickle', 'wb') as handle4:
    pickle.dump(fullstops_and_pronouns, handle4, protocol=pickle.HIGHEST_PROTOCOL)
with open('pickled-data/fullstops_pronouns_and_names.pickle', 'wb') as handle5:
    pickle.dump(fullstops_pronouns_and_names, handle5, protocol=pickle.HIGHEST_PROTOCOL)
