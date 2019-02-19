import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import csv

num_data_points = 50
print("num_data_points: %d" % num_data_points)

data = pd.read_csv("./data/College.csv")
df = data.iloc[:num_data_points,2:20] # first 50 data points
matrix = pd.DataFrame.as_matrix(df)
distances = pdist(matrix, metric='euclidean')
dist_matrix = squareform(distances)

mst = minimum_spanning_tree(dist_matrix)
mst = mst.toarray().astype(int)
print("minimun number of edges: %d" % np.count_nonzero(mst))

collage_name = data.iloc[:num_data_points,0:1]
collage_name_mtx = pd.DataFrame.as_matrix(collage_name)

cse_file_path = "./data/minimum_spanning_tree_mtx.csv"
with open(cse_file_path,"w+") as csv_file:
    csvWriter = csv.writer(csv_file,delimiter=',')
    csvWriter.writerows(mst)

print("Finished writing minimum spanning tree matrix to %s." % cse_file_path)

'''
Reference:
https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html
https://stackoverflow.com/questions/20303323/distance-calculation-between-rows-in-pandas-dataframe-using-a-distance-matrix
https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html
https://stackoverflow.com/questions/44691524/write-a-2d-array-to-a-csv-file-with-delimiter
'''
