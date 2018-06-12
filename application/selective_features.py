def dictionaries_of_features(list_of_dictionaries, feature_list):
    import pandas as pd
    """Loops through the list of dictionaries supplied, gathers counts for each term in the feature list,
    and returns a new list of smaller dictionaries. We then pass the reesults to sklearn CountVectorizer for zerofill
    and other model processing"""
    reduced_dictionaries = []
    for d in list_of_dictionaries:
        processing_dictionary = {}
        for feature in feature_list:
            # Here we just try to find the term in the source dictionary and skip if there's an exception,
            # which can only happen if the term is not in the source dictionary
            # This is more memory performant than an if-then approach
            try:
                processing_dictionary[feature] = d[feature]
            except:
                pass
        #finally we append the processing_dictionary to the new list of dicts, preserving their original order
        reduced_dictionaries.append(processing_dictionary)
    return reduced_dictionaries

def dictionaries_without_features(list_of_dictionaries, feature_list):
    import pandas as pd
    """Loops through the list of dictionaries supplied, gathers counts for each term in the feature list,
    and returns a new list of smaller dictionaries with terms from feature list removed. We then pass the reesults to sklearn CountVectorizer for zerofill
    and other model processing"""
    reduced_dictionaries = []
    for d in list_of_dictionaries:
        processing_dictionary = dict(d)
        for feature in feature_list:
            # Here we just try to find the term in the source dictionary and skip if there's an exception,
            # which can only happen if the term is not in the source dictionary
            # This is more memory performant than an if-then approach
            try:
                del processing_dictionary[feature]
            except:
                pass
        #finally we append the processing_dictionary to the new list of dicts, preserving their original order
        reduced_dictionaries.append(processing_dictionary)
    return reduced_dictionaries

def make_genres_big_and_lavin(piped_genres):
    import pandas as pd
    big_genres = pd.read_csv("meta/datadictionary.csv")
    gen_dict = {}
    for i in big_genres.itertuples():
        gen_dict[i[1]] = i[2]
    gen_dict["chimyst"] = "crime"
    gen_dict["locghost"] = "gothic"
    gen_dict["lockandkey"] = "crime"
    gen_dict["lochorror"] = "gothic"
    gen_dict["chihorror"] = "gothic"
    genres_main = []
    genres_lavin = []
    for i in piped_genres:
        gen = i.split(" | ")
        g = []
        lavin_gens = []

        for z in gen:
            if "lavin" in z:
                lavin_gens.append(z)

            if z != "teamred" and z!= "teamblack" and z!= "stew" and z != "juvenile" and z != "drop" and "random" not in z:
                #look up and append big genre
                try:
                    g.append(gen_dict[z])
                except:
                    pass
                    #g.append(z)
        if len(lavin_gens) == 0:
            genres_lavin.append("no_lavin_tag")
        if len(lavin_gens) == 1:
            genres_lavin.append(lavin_gens[0])
        if len(lavin_gens) > 1:
            genres_lavin.append("lavin_multi")
        #merge duplicates
        g = list(set(g))
        if len(g) > 1:
            final_genre = "multi"
        if len(g) == 0:
            final_genre = "non_genre"
        if len(g) == 1:
            final_genre = g[0]
        genres_main.append((g, final_genre))
    processed_genre = [i[0] for i in genres_main]
    final_genre = [i[1] for i in genres_main]
    return processed_genre, final_genre, genres_lavin

def make_feature_list(csv, col, N):
    import pandas as pd
    df = pd.read_csv(csv)
    #sort by col
    ## make sure it's descending
    df[col] = [abs(i) for i in list(df[col])]
    # Convert correlations to absolute values
    new_df = df.sort_values(by=col)
    #convert top N features to list
    list_of_features = list(new_df["term"])[-N:]
    return list_of_features
