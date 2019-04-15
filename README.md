# Gender Dynamics and Critical Reception:

## A Study of Early 20th-century Book Reviews from _The New York Times_

Code and data for "Gender Dynamics and Critical Reception: A Study of Early 20th-century Book Reviews from _The New York Times_"

# Contents

This repository contains several python scripts that can be used to reproduce results. The scripts contain a 'shuffle and repeat' operation that includes a certain amount of randomness, so your results will likely vary from my own in very minor ways, but mean accuracy scores and coefficient lists should be very similar if not identical. Raw ocr files are not included, but links to original pdf files can be found in `metadata.csv` (gender labels, ids, etc. are also in this csv). Additionally, the repository contains:

- lemma count tables (in the 'lemma-data' folder)
- a python script called text_cleanup.py which show the steps used to generate files in the 'lemma-data' folder
- csv/html/png versions of tables and figures from the article (in the 'tables-and-figures' folder)
- a 'pickled-data' folder for pickle files, which are generated when various scripts are executed

# Steps To Reproduce

0. Set up repo locally 
1. Install Python 3.5.x (if necessary)
2. Set up a virtual environment and activate (for best results)
3. Install python dependencies
4. Install two nltk corpora
5. Execute make_feature_lists.py script
6. Execute run_regression.py
7. Inspect results in `regression_scores.db`

## 0. Set up repo locally 

From your terminal app or other command prompt, clone this repository using `git clone https://github.com/mjlavin80/nyt-reviews-1905-gender` (requires git)

Move into the newly cloned folder, which should be called `nyt-reviews-1905-gender`

## 1. Install Python 3.5.x (if necessary) and pip3

Instructions for installing Python 3.5.x are different depending on your operating system. See https://www.python.org/downloads/ (these scripts should also work with Python 3.6, but I haven't tested them on that version)

You may also want to use the Anaconda installer, as it comes packaged with several of this repo's dependencies. See  https://anaconda.org/anaconda/python

If you are already a homebrew user or you want to maintain two different version of python on your system, this link may be of use: https://docs.brew.sh/Homebrew-and-Python

Pip3 comes with the brew and anaconda versions. Otherwise install with `sudo easy_install pip3` (requires admin privileges) 

## 2. Set up a virtual environment and activate (for best results)

See https://docs.python.org/3/library/venv.html or https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/ if using Ananconda

## 3. Install python dependencies 

Run the command `pip3 install -r requirements.txt` (works best if inside a virtual environment)

## 4. Install nltk corpora (names and stopwords)

See https://www.nltk.org/data.html for detailed instructions

## 5. Execute make_feature_lists.py script

`python 3 make_feature_lists.py` (works best if inside a virtual environment)

## 6. Execute run_regression.py

`python 3 run_regression.py ` (works best if inside a virtual environment). 

This block of code will run various logistic regression scenarios and store results in an sqlite database called 'regression_scores.db'. If you wish to save an older database and rerun the script, rename the existing regression_scores.db before attempting to reproduce the results.

## 7. Inspect results in regression_scores.db

The generated database has three tables: 'main', 'results', and 'coefficients':

- 'main' stores 
- 'results' stores
- 'coefficients' stores 

The best way to work with this database is probably to explore it using a graphical tool like sqlitebrowser and then write python scripts to collect and tabulate data. Be warned that, once generated, this file will be 1-2 GBs in size using the default setup in run_regression.py. As a result, querying the database can be slow, depending on the query type. 

# License and DOI 

TBD
