'''
Student: Bryan Bo Cao
SBU ID: 112130213
Email: bo.cao.1@stonybrook.edu or boccao@cs.stonybrook.edu

Dataset:
College 777 data points, 18 dimensions
https://vincentarelbundock.github.io/Rdatasets/datasets.html

Reference:
https://getbootstrap.com/docs
https://www.w3schools.com
Flask code base from TA: https://github.com/hawkeye154/d3_flask_tutorial
'''

import json

from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/", methods = ['POST', 'GET'])
def index():

    global df
    df_all_data = df[['Apps','Accept','Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate',
    	       'Room.Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate']]
    print("df_all_data:")
    print(df_all_data)
    print("Population of all_data: %d" % len(df_all_data))
    print("Number of dimension of all_data: %d" % len(df_all_data.columns))

    # ======================== Task1 ========================
    # Task 1: data clustering and decimation (30 points)
    # implement random sampling and stratified sampling
    # the latter includes the need for k-means clustering (optimize k using elbow)
    # ======================== Bryan Bo Cao ========================
    # TA mentioned in the post @123 and @124 in Piazza that we can use in-built
    # pandas or sklearn method, so I use them directly for task1.
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
    # https://datascience.stackexchange.com/questions/16700/confused-about-how-to-apply-kmeans-on-my-a-dataset-with-features-extracted
    # https://pythonprogramminglanguage.com/kmeans-elbow-method/

    # =================== random sampling ====================
    df_sampled_data = df_all_data.sample(n=int(len(df_all_data) / 2))
    print("df_sampled_data:")
    print(df_sampled_data)
    print("Number of instance in sampled_data: %d" % len(df_sampled_data))
    print("Number of dimension of sampled_data: %d" % len(df_sampled_data.columns))

    # ================= stratified sampling ==================
    # optimize k using elbow
    distortions = []
    n_k = range(1,10)
    decs = []
    pre_distortion = 0
    print("KMeans optimizing k using elbow")
    for k in n_k:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df_all_data)

        # using standard euclidean distance
        distortion = np.sum(np.min(cdist(df_all_data, kmeans.cluster_centers_, 'euclidean'), axis=1)) \
                            / df_all_data.shape[0]
        distortions.append(distortion)
        if k == 1: dec = -float("inf")
        else: dec = distortion - pre_distortion
        decs.append(dec)
        pre_distortion = distortion
        print("k: %d, distortion: %.4f, dec: %0.4f" % (k, distortion, dec))

    elbow_k = 0
    for i in range(2, len(decs) + 1):
        diff = decs[i + 1] - decs[i]
        if diff < 60:
            elbow_k = i
            break
    print("elbow_k: %d" % elbow_k)

    # stratified sampling
    kmeans = KMeans(n_clusters=elbow_k)
    kmeans.fit(df_all_data)
    labels = kmeans.labels_
    # print(labels)

    chart_data = df_all_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("index.html", data=df_all_data)


if __name__ == "__main__":
    df = pd.read_csv('College.csv')
    app.run(debug=True)
