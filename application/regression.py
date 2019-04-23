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

def feature_dicts_from_nyt_ids(list_of_ids):
    feature_dicts = []
    for i in list_of_ids:
        df = pd.read_csv("lemma-tables/%s.csv" % i).fillna("$$$$$")
        # convert df to dictionary    
        my_dict = {}
        for i in df.itertuples():
            if i[1] != "$$$$$":
                my_dict[i[1]] = int(i[2])
            
        feature_dicts.append(my_dict)
    return feature_dicts

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
    with open('pickled-data/ocr_dicts_male.pickle', 'rb') as handle2:
        ocr_dicts_male = pickle.load(handle2)
    with open('pickled-data/ocr_dicts_female.pickle', 'rb') as handle4:
        ocr_dicts_female = pickle.load(handle4)

    try:
        with open('pickled-data/trainset_1905_labels.pickle', 'rb') as handle5:
            trainset_1905_labels = pickle.load(handle5)
        with open('pickled-data/trainset_1905_nyt_ids.pickle', 'rb') as handle6:
            trainset_1905_nyt_ids = pickle.load(handle6)
        with open('pickled-data/trainset_ocr_dicts_1905.pickle', 'rb') as handle7:
            trainset_ocr_dicts_1905 = pickle.load(handle7)
        with open('pickled-data/trainset_1925_labels.pickle', 'rb') as handle8:
            trainset_1925_labels = pickle.load(handle8)
        with open('pickled-data/trainset_1925_nyt_ids.pickle', 'rb') as handle9:
            trainset_1925_nyt_ids = pickle.load(handle9)
        with open('pickled-data/trainset_ocr_dicts_1925.pickle', 'rb') as handle10:
            trainset_ocr_dicts_1925  = pickle.load(handle10)

    except:
        meta_rows = pd.read_csv("metadata.csv")
        cluster_rows = pd.read_csv("meta_cluster.csv")
        cluster_rows['nyt_id'] = cluster_rows['nyt_id'].map(str) + "-" + cluster_rows['cluster_id'].map(str)
        meta_rows = meta_rows.append(cluster_rows, sort=False).reset_index(drop=True)
        
        trainset_1905 = meta_rows.loc[meta_rows['year'] == 1905].loc[~meta_rows['nyt_id'].isin(nyt_ids_all)].reset_index(drop=True) 
        #just m, just f, append
        trainset_1905_male = trainset_1905.loc[trainset_1905['perceived_author_gender'] == 'm'].reset_index(drop=True) 
        trainset_1905_female = trainset_1905.loc[trainset_1905['perceived_author_gender'] == 'f'].reset_index(drop=True)
        trainset_1905_sorted = trainset_1905_male.append(trainset_1905_female, sort=False) 

        trainset_1925 = meta_rows.loc[meta_rows['year'] == 1925].loc[~meta_rows['nyt_id'].isin(nyt_ids_all)].reset_index(drop=True)
        #just m, just f, append
        trainset_1925_male = trainset_1925.loc[trainset_1925['perceived_author_gender'] == 'm'].reset_index(drop=True) 
        trainset_1925_female = trainset_1925.loc[trainset_1925['perceived_author_gender'] == 'f'].reset_index(drop=True)
        trainset_1925_sorted = trainset_1925_male.append(trainset_1925_female, sort=False) 

        trainset_1905_nyt_ids = list(trainset_1905_sorted['nyt_id'])
        trainset_1925_nyt_ids = list(trainset_1925_sorted['nyt_id'])

        ocr_dicts_1905 = feature_dicts_from_nyt_ids(trainset_1905_nyt_ids)
        ocr_dicts_1925 = feature_dicts_from_nyt_ids(trainset_1925_nyt_ids)

        gender_1905_labels = list(trainset_1905_sorted['perceived_author_gender'])
        gender_1925_labels = list(trainset_1925_sorted['perceived_author_gender'])

        trainset_1905_labels = [0 if z == 'm' else 1 for z in gender_1905_labels]
        trainset_1925_labels =[0 if z == 'm' else 1 for z in gender_1925_labels]

        with open('pickled-data/trainset_1905_labels.pickle', 'wb') as handle5:
            pickle.dump(trainset_1905_labels, handle5, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickled-data/trainset_1905_nyt_ids.pickle', 'wb') as handle6:
            pickle.dump(trainset_1905_nyt_ids, handle6, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickled-data/trainset_ocr_dicts_1905.pickle', 'wb') as handle7:
            pickle.dump(trainset_ocr_dicts_1905, handle7, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickled-data/trainset_1925_labels.pickle', 'wb') as handle8:
            pickle.dump(trainset_1925_labels, handle8, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickled-data/trainset_1925_nyt_ids.pickle', 'wb') as handle9:
            pickle.dump(trainset_1925_nyt_ids, handle9, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pickled-data/trainset_ocr_dicts_1925.pickle', 'wb') as handle10:
            pickle.dump(trainset_ocr_dicts_1925, handle10, protocol=pickle.HIGHEST_PROTOCOL)

    gender_label_stats= {'male': len(ocr_dicts_male), 'female': len(ocr_dicts_female)} 
    label_cutoff = gender_label_stats['male']-1
except:
    # get ids and sort by gender 
    
    # get all rows from metadata
    meta_rows = pd.read_csv("metadata.csv")
    
    # get all rows from cluster_meta
    cluster_rows = pd.read_csv("meta_cluster.csv")
    cluster_rows['nyt_id'] = cluster_rows['nyt_id'].map(str) + "-" + cluster_rows['cluster_id'].map(str)

    female_rows_cluster = cluster_rows.loc[cluster_rows['perceived_author_gender'] == 'f']
    male_rows_cluster = cluster_rows.loc[cluster_rows['perceived_author_gender'] == 'm']
    #generate a list of nyt_ids with male labels
    nyt_ids_male_cluster = list(male_rows_cluster['nyt_id'])
    #generate a list of nyt_ids with female labels
    nyt_ids_female_cluster = list(female_rows_cluster['nyt_id'])

    f_1905 = meta_rows.loc[meta_rows['year'] == 1905].loc[meta_rows['perceived_author_gender'] == 'f'].sample(35).reset_index(drop=True)
    m_1905 = meta_rows.loc[meta_rows['year'] == 1905].loc[meta_rows['perceived_author_gender'] == 'm'].sample(100).reset_index(drop=True)
    f_1925 = meta_rows.loc[meta_rows['year'] == 1925].loc[meta_rows['perceived_author_gender'] == 'f'].sample(35).reset_index(drop=True)
    m_1925 = meta_rows.loc[meta_rows['year'] == 1925].loc[meta_rows['perceived_author_gender'] == 'm'].sample(100).reset_index(drop=True)

    female_rows = meta_rows.loc[meta_rows['perceived_author_gender'] == 'f'].loc[meta_rows['year'] > 1905].loc[meta_rows['year'] < 1925].reset_index(drop=True)
    male_rows = meta_rows.loc[meta_rows['perceived_author_gender'] == 'm'].loc[meta_rows['year'] > 1905].loc[meta_rows['year'] < 1925].reset_index(drop=True)
    male_rows = male_rows.append(m_1905).append(m_1925, sort=False)
    female_rows = female_rows.append(f_1905).append(f_1925, sort=False)
    
    #generate a list of nyt_ids with male labels
    nyt_ids_male = list(male_rows['nyt_id']) + nyt_ids_male_cluster
    #generate a list of nyt_ids with female labels
    nyt_ids_female = list(female_rows['nyt_id']) + nyt_ids_female_cluster
    
    #append ids so the list is a group of male labels followed by female 
    nyt_ids_all = nyt_ids_male + nyt_ids_female  
    
    ocr_dicts_male = []
    for i in nyt_ids_male:
        df = pd.read_csv("lemma-tables/%s.csv" % i).fillna("$$$$$")
        # convert df to dictionary    
        my_dict = {}
        for i in df.itertuples():
            if i[1] != "$$$$$":
                my_dict[i[1]] = int(i[2])
            
        ocr_dicts_male.append(my_dict)
    
    ocr_dicts_female = []
    for i in nyt_ids_female:
        df = pd.read_csv("lemma-tables/%s.csv" % i).fillna("$$$$$")
        # convert df to dictionary    
        my_dict = {}
        for i in df.itertuples():
            if i[1] != "$$$$$":
                my_dict[i[1]] = int(i[2])
            
        ocr_dicts_female.append(my_dict)

    ocr_dicts_all = ocr_dicts_male + ocr_dicts_female

    with open('pickled-data/ocr_dicts_all.pickle', 'wb') as handle:
        pickle.dump(ocr_dicts_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickled-data/nyt_ids_all.pickle', 'wb') as handle3:
        pickle.dump(nyt_ids_all, handle3, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickled-data/ocr_dicts_male.pickle', 'wb') as handle2:
        pickle.dump(ocr_dicts_male, handle2, protocol=pickle.HIGHEST_PROTOCOL)    
    with open('pickled-data/ocr_dicts_female.pickle', 'wb') as handle4:
        pickle.dump(ocr_dicts_female, handle4, protocol=pickle.HIGHEST_PROTOCOL)
    
    gender_label_stats= {'male': len(ocr_dicts_male), 'female': len(ocr_dicts_female)} 
    label_cutoff = gender_label_stats['male']-1

    meta_rows = pd.read_csv("metadata.csv")
    cluster_rows = pd.read_csv("meta_cluster.csv")
    cluster_rows['nyt_id'] = cluster_rows['nyt_id'].map(str) + "-" + cluster_rows['cluster_id'].map(str)
    meta_rows = meta_rows.append(cluster_rows, sort=False).reset_index(drop=True)
    
    trainset_1905 = meta_rows.loc[meta_rows['year'] == 1905].loc[~meta_rows['nyt_id'].isin(nyt_ids_all)].reset_index(drop=True) 
    #just m, just f, append
    trainset_1905_male = trainset_1905.loc[trainset_1905['perceived_author_gender'] == 'm'].reset_index(drop=True) 
    trainset_1905_female = trainset_1905.loc[trainset_1905['perceived_author_gender'] == 'f'].reset_index(drop=True)
    trainset_1905_sorted = trainset_1905_male.append(trainset_1905_female, sort=False) 

    trainset_1925 = meta_rows.loc[meta_rows['year'] == 1925].loc[~meta_rows['nyt_id'].isin(nyt_ids_all)].reset_index(drop=True)
    #just m, just f, append
    trainset_1925_male = trainset_1925.loc[trainset_1925['perceived_author_gender'] == 'm'].reset_index(drop=True) 
    trainset_1925_female = trainset_1925.loc[trainset_1925['perceived_author_gender'] == 'f'].reset_index(drop=True)
    trainset_1925_sorted = trainset_1925_male.append(trainset_1925_female, sort=False) 

    train_1905_ids = list(trainset_1905_sorted['nyt_id'])
    train_1925_ids = list(trainset_1925_sorted['nyt_id'])

    trainset_ocr_dicts_1905 = feature_dicts_from_nyt_ids(train_1905_ids)
    trainset_ocr_dicts_1925 = feature_dicts_from_nyt_ids(train_1925_ids)

    gender_1905_labels = list(trainset_1905_sorted['perceived_author_gender'])
    gender_1925_labels = list(trainset_1925_sorted['perceived_author_gender'])

    trainset_1905_labels = [0 if z == 'm' else 1 for z in gender_1905_labels]
    trainset_1925_labels =[0 if z == 'm' else 1 for z in gender_1925_labels]

    with open('pickled-data/trainset_1905_labels.pickle', 'wb') as handle5:
        pickle.dump(trainset_1905_labels, handle5, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickled-data/trainset_1905_nyt_ids.pickle', 'wb') as handle6:
        pickle.dump(trainset_1905_nyt_ids, handle6, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickled-data/trainset_ocr_dicts_1905.pickle', 'wb') as handle7:
        pickle.dump(trainset_ocr_dicts_1905, handle7, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickled-data/trainset_1925_labels.pickle', 'wb') as handle8:
        pickle.dump(trainset_1925_labels, handle8, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickled-data/trainset_1925_nyt_ids.pickle', 'wb') as handle9:
        pickle.dump(trainset_1925_nyt_ids, handle9, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickled-data/trainset_ocr_dicts_1925.pickle', 'wb') as handle10:
        pickle.dump(trainset_ocr_dicts_1925, handle10, protocol=pickle.HIGHEST_PROTOCOL)

# begin functions
#
#
#

def set_train_and_test_year(vsm_array, label_cutoff, year):
    # train is len() of trainset_1905_nyt_ids or trainset_1925_nyt_ids, depending on year
    if year == 1905:
        train_len = len(trainset_1905_nyt_ids)
        train_ids = trainset_1905_nyt_ids
        train_labels = trainset_1905_labels
    if year == 1925:
        train_len = len(trainset_1925_nyt_ids)
        train_ids = trainset_1925_nyt_ids
        train_labels = trainset_1925_labels


    # assume vsm array contains train and test
    X_train = vsm_array[0:train_len]
    X_test = vsm_array[train_len:]
    
    m = [0 for m in range(label_cutoff)] 
    f =[1 for n in range(label_cutoff, len(X_test))]
 
    # set test_labels using range and the label_cutoff
    test_labels = m + f 
    
    y_train = ['',]
    y_test = ['',]
    
    # assume training_1905_labels and training_1925_labels are set
    test_nyt_ids = nyt_ids_all

    # return all
    return X_train, X_test, y_train, y_test, train_labels, test_labels, test_nyt_ids

def set_train_and_test_random(vsm_array, label_cutoff, train_size_int, test_size_int, random_state):
    
    y_as_list = list(range(len(vsm_array)))
    # y_train and y_test here are lists of positions from scaled vsm and need to be converted to values 0 or 1       
    X_train, X_test, y_train, y_test = train_test_split(vsm_array, y_as_list, train_size=train_size_int, \
                                                        test_size=test_size_int, random_state=random_state)
    train_labels = []
    # 0 = male here
    for pos in y_train:
        if pos < label_cutoff:
            value = 0
        else:
            value = 1
        train_labels.append(value)
    test_labels = []
    test_nyt_ids = []
    for pos in y_test:
        test_nyt_ids.append(nyt_ids_all[pos])
        if pos < label_cutoff:
            value = 0
        else:
            value = 1
        test_labels.append(value)
    return X_train, X_test, y_train, y_test, train_labels, test_labels, test_nyt_ids


def vectorize_and_predict(list_of_dicts, c, conn, parameters, cw, label_cutoff, train_size_int, test_size_int, train_mode='random'):
    """ 
    This function take a list of dictionaries, an sqlite3 connection and cursor, and variables relevant for logistic regression 
    and inserts machine learning results into a results database (sqlite3). 
    parameters is a dictionary of information about the machine learning scenarios
    cw is a number between 0 and 1 used for class weights
    label_cutoff is an integer describing how many male reviews there are (used for generating training and test labels)
    train_size_int is an integer used to determine the size of the training set
    test_size_int is an integer used to determine the size of the test set

    """
    v = DictVectorizer()
    X = v.fit_transform(list_of_dicts)
    y = TfidfTransformer()
    Z = y.fit_transform(X)
    
    scaled_vsm = Z.toarray()
        
    for i in range(1):
        
        if type(train_mode) == int:
            # here set train and test data ands labels based on year value
            X_train, X_test, y_train, y_test, train_labels, test_labels, test_nyt_ids = set_train_and_test_year(scaled_vsm, label_cutoff, train_mode)
        else:
            X_train, X_test, y_train, y_test, train_labels, test_labels, test_nyt_ids = set_train_and_test_random(scaled_vsm, label_cutoff, train_size_int, test_size_int, i)

        # Create classifiers
        if type(train_mode) == int:
            lr = LogisticRegression(class_weight='balanced', random_state=i)
        else:
            lr = LogisticRegression(class_weight={0:1-cw, 1:cw})

        lr.fit(X_train, train_labels)
        results = lr.predict(X_test)
        probs = lr.predict_proba(X_test)

        df = pd.DataFrame()
        df['prob'] = [k[1] for k in probs]
        top_f = list(df.sort_values(by='prob', ascending=False).iloc[0:542].index)

        results_by_prob = []
        for e,z in enumerate(probs):
            if e in top_f:
                results_by_prob.append(1)
            else:
                results_by_prob.append(0)
        
        #score = lr.score(X_test, test_labels)
        
        f_score_f = f1_score(test_labels, results_by_prob, pos_label=1, average='binary')  
        prec_f = precision_score(test_labels, results_by_prob, pos_label=1, average='binary')
        rec_f = recall_score(test_labels, results_by_prob, pos_label=1, average='binary')
        
        f_score_m = f1_score(test_labels, results_by_prob, pos_label=0, average='binary')  
        prec_m = precision_score(test_labels, results_by_prob, pos_label=0, average='binary')
        rec_m = recall_score(test_labels, results_by_prob, pos_label=0, average='binary')
        
        acc = accuracy_score(test_labels, results_by_prob)
        
        y_train = [str(u) for u in y_train]
        y_test = [str(z) for z in y_test]

        a = ', '.join(y_train)
        b = ", ".join(y_test)

        cnf_matrix = confusion_matrix(test_labels, results_by_prob)

        row = [parameters['features_used'], parameters['stopwords'], f_score_f, prec_f, rec_f, f_score_m, prec_m, \
               rec_m, acc, cnf_matrix[0][0], cnf_matrix[0][1], cnf_matrix[1][1], cnf_matrix[1][0], len(X_train), \
               len(X_test), i, a, b]
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
        for j, k in enumerate(results_by_prob):
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
    print(parameters['features_used'], parameters['stopwords'])

#
#
#
#
# end function