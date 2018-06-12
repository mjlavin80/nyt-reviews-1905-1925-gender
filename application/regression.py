# import libraries
import pandas as pd
import pickle
import numpy as np 
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

# make lemma list pickles if they don't exist (saves time if you plan to rerun) 
try:
    with open('pickled-data/ocr_dicts_all.pickle', 'rb') as handle:
        ocr_dicts_all = pickle.load(handle)
    with open('pickled-data/nyt_ids_all.pickle', 'rb') as handle3:
        nyt_ids_all = pickle.load(handle3)
except:
    # get ids and sort by gender (this code is overly complex, as I retrofitted it to a prior script)
    all_rows = pd.read_csv("metadata.csv")
    female_rows = all_rows.loc[all_rows['assumed_gender'] == 'f']
    male_rows = all_rows.loc[all_rows['assumed_gender'] == 'm']
    
    nyt_ids_male = list(male_rows['nyt_id'])
    nyt_ids_female = list(female_rows['nyt_id'])
    
    nyt_ids_all = nyt_ids_male + nyt_ids_female
    
    ocr_dicts_all = []
    for i in nyt_ids_all:
        df = pd.read_csv("lemma-data/%s.csv" % i).fillna("$$$$$")
        # convert df to dictionary    
        my_dict = {}
        for i in df.itertuples():
            if i[1] != "$$$$$":
                my_dict[i[1]] = int(i[2])
            
        ocr_dicts_all.append(my_dict)
    ocr_dicts_all = [i for i in ocr_dicts_all]
    with open('pickled-data/ocr_dicts_all.pickle', 'wb') as handle:
        pickle.dump(ocr_dicts_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickled-data/nyt_ids_all.pickle', 'wb') as handle3:
        pickle.dump(nyt_ids_all, handle3, protocol=pickle.HIGHEST_PROTOCOL)

# begin functions
#
#
#
#

def vectorize_and_predict(list_of_dicts, c, conn, parameters):
    
    v = DictVectorizer()
    X = v.fit_transform(list_of_dicts)
    y = TfidfTransformer()
    Z = y.fit_transform(X)
    scaled_vsm = Z.toarray()
    y_as_list = list(range(len(scaled_vsm)))

    for i in range(1000):
        #(703, 159)
        train_size_int = 662
        test_size_int = 200
        X_train, X_test, y_train, y_test = train_test_split(scaled_vsm, y_as_list, train_size=train_size_int, \
                                                            test_size=test_size_int, random_state=i)
        train_labels = []
        # 0 = male here
        for pos in y_train:
            if pos < 703:
                value = 0
            else:
                value = 1
            train_labels.append(value)
        test_labels = []
        test_nyt_ids = []
        for pos in y_test:
            test_nyt_ids.append(nyt_ids_all[pos])
            if pos < 703:
                value = 0
            else:
                value = 1
            test_labels.append(value)
        
        #spend more time optimizing?
        #cw = 703*1.0/(703+159)
        cw = 700.0/(700+140)
        
        # Create classifiers
        lr = LogisticRegression(class_weight={0:1-cw, 1:cw})
        lr.fit(X_train, train_labels)
        results = lr.predict(X_test)
        probs = lr.predict_proba(X_test)

        #score = lr.score(X_test, test_labels)
        
        
        f_score_f = f1_score(test_labels, results, pos_label=1, average='binary')  
        prec_f = precision_score(test_labels, results, pos_label=1, average='binary')
        rec_f = recall_score(test_labels, results, pos_label=1, average='binary')
        
        f_score_m = f1_score(test_labels, results, pos_label=0, average='binary')  
        prec_m = precision_score(test_labels, results, pos_label=0, average='binary')
        rec_m = recall_score(test_labels, results, pos_label=0, average='binary')
        
        acc = accuracy_score(test_labels, results)
        
        y_train = [str(u) for u in y_train]
        y_test = [str(z) for z in y_test]

        a = ', '.join(y_train)
        b = ", ".join(y_test)

        cnf_matrix = confusion_matrix(test_labels, results)

        #row = [i, a, b, score, train_size_int, test_size_int, ]
        row = [parameters['features_used'], parameters['stopwords'], f_score_f, prec_f, rec_f, f_score_m, prec_m, \
               rec_m, acc, cnf_matrix[0][0], cnf_matrix[0][1], cnf_matrix[1][1], cnf_matrix[1][0], train_size_int, \
               test_size_int, i, a, b]
        cols= ["features_used", "words_removed", "f1_score_f", "precision_f", "recall_f", "f1_score_m", "precision_m", \
               "recall_m", "accuracy","male_match", "male_mismatch", "female_match", "female_mismatch", "train_size", \
               "test_size", "random_seed", "train_ids", "test_ids"]
        df_main = pd.DataFrame.from_records([row,], columns=cols)
        
        df_main.to_sql('main', conn, 'sqlite', if_exists='append', index=False)

        words = v.vocabulary_.keys()
        labeled_coefs = {}
        for z in words:
            try:
                labeled_coefs[z] = lr.coef_[0][v.vocabulary_[z]]
            except: 
                pass

        #define main_id
        main_id = c.execute("SELECT MAX(id) FROM main").fetchone()[0]
        
        results_rows = []
        #main_id, nyt_id, predicted_gender, labeled_gender
        for j, k in enumerate(results):
            results_row = (main_id, test_nyt_ids[j], k, test_labels[j], probs[j][0], probs[j][1])
            results_rows.append(results_row)
        
        result_cols = ["main_id", "nyt_id", "predicted_gender", "labeled_gender", "probability_male", "probability_female"]
        df_results = pd.DataFrame.from_records(results_rows, columns = result_cols)
        df_results.to_sql('results', conn, 'sqlite', if_exists='append', index=False, chunksize=100)
        pairs = list(labeled_coefs.items())                                      
        df_coef = pd.DataFrame.from_records(pairs, columns =["feature", "score"])
        df_coef['main_id'] = main_id
        df_coef['odds'] = np.exp(df_coef['score'])
        
        #only save the first 1000 rows
        df_coef = df_coef.sort_values(by="score", ascending=False).reset_index(drop=True)
        
        df_coef_f = df_coef.iloc[0:1000]
        
        df_coef_m = df_coef.iloc[-1001:-1]
        df_coef = pd.concat([df_coef_f,  df_coef_m]).reset_index(drop=True)
        df_coef.to_sql('coefficients', conn, 'sqlite', if_exists='append', index=False, chunksize=100) 
    print("finished %s, %s") % (parameters['features_used'], parameters['stopwords'])

#
#
#
#
# end function