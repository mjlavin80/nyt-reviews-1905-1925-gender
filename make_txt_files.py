#/usr/bin/env python3

#####
# This script assumes an sqlite database called nyt_reviews_datastore.db populated with data about book reviews. 
# My user agreement with The New York Times does not allow me to share the full text of book reviews.
# This script, instead, is here so that readers of my work can see the scripts I used to generate lemma and term frequency tables
#####

import sqlite3

conn = sqlite3.connect('nyt_reviews_datastore.db')
c = conn.cursor()

meta_query = """ SELECT nyt_id, nyt_ocr FROM metadata WHERE review_type='single_focus'"""

cluster_query = """SELECT nyt_id, cluster_id, nyt_ocr FROM cluster_meta WHERE review_type='single_focus'"""

metarows = c.execute(meta_query).fetchall()
clusterrows = c.execute(cluster_query).fetchall()

for row in metarows:
    filename = ''.join(['ocr/', row[0], '.txt'])
    with open(filename, 'a') as f:
        f.write(row[1])
    f.close()

for cluster_row in clusterrows:
    filename = ''.join(['ocr/', cluster_row[0], '-', cluster_row[1],'.txt'])
    with open(filename, 'a') as f:
        f.write(cluster_row[2])
    f.close()