'''
Student: Bryan Bo Cao
SBU ID: 112130213
Email: bo.cao.1@stonybrook.edu or boccao@cs.stonybrook.edu

Reference:
https://getbootstrap.com/docs
https://www.w3schools.com
Flask code base from TA: https://github.com/hawkeye154/d3_flask_tutorial
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
https://datascience.stackexchange.com/questions/16700/confused-about-how-to-apply-kmeans-on-my-a-dataset-with-features-extracted
https://pythonprogramminglanguage.com/kmeans-elbow-method/
https://datascience.stackexchange.com/questions/16700/confused-about-how-to-apply-kmeans-on-my-a-dataset-with-features-extracted
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
'''

import json

from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import decomposition
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/", methods = ['POST', 'GET'])
def index():

    print("====== index.html backend computing starts ===================================")

    print("====== index.html backend computing ends ===================================")
    print("Visualizing data in the frontend...")
    if request.method == 'POST':
        return render_template("index.html")
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
