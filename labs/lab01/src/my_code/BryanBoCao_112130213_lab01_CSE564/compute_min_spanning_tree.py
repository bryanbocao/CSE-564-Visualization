'''
Student: Bryan Bo Cao
SBU ID: 112130213
Email: bo.cao.1@stonybrook.edu or boccao@stonybrook.edu
'''

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import csv
import json
from sklearn.preprocessing import normalize

num_data_points = 70
print("num_data_points: %d" % num_data_points)

data = pd.read_csv("./data/College.csv")
df = data.iloc[:num_data_points,2:] # first num_data_points data points
matrix = pd.DataFrame.as_matrix(df)
distances = pdist(matrix, metric='euclidean')
dist_matrix = squareform(distances)

mst = minimum_spanning_tree(dist_matrix)
mst = mst.toarray().astype(int)
normalized_mst_matrix = normalize(mst, axis=1, norm='l1')
print("minimun number of edges: %d" % np.count_nonzero(normalized_mst_matrix))
# print(normalized_mst_matrix)

collage_name = data.iloc[:num_data_points,0:1]
collage_name_mtx = pd.DataFrame.as_matrix(collage_name)

write_data = {}
# construct nodes
write_data['nodes'] = []
for i in range(num_data_points):
    node = {}
    node['collage_name'] = collage_name_mtx[i][0]
    write_data['nodes'].append(node)

# construct links
write_data['links'] = []
for i in range(num_data_points):
    for j in range(num_data_points):
        if mst[i][j] > 0:
            link = {}
            link['source'] = collage_name_mtx[i][0]
            link['target'] = collage_name_mtx[j][0]
            link['value'] = str(normalized_mst_matrix[i][j])
            write_data['links'].append(link)

# print(write_data)

json_file_path = "./data/minimum_spanning_tree_mtx.json"
with open(json_file_path,"w+") as json_file:
    json.dump(write_data, json_file)
print("Finished writing minimum spanning tree matrix to %s." % json_file_path)

'''
Reference:
https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html
https://stackoverflow.com/questions/20303323/distance-calculation-between-rows-in-pandas-dataframe-using-a-distance-matrix
https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html
https://stackoverflow.com/questions/44691524/write-a-2d-array-to-a-csv-file-with-delimiter
'''
