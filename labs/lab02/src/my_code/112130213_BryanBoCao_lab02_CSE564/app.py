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
from sklearn.utils.random import sample_without_replacement

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

    # sample half
    df_sampled_data = df_all_data.sample(n=int(len(df_all_data) / 2))
    print("df_sampled_data:")
    print(df_sampled_data)
    print("Number of instance in sampled_data: %d" % len(df_sampled_data))
    print("Number of dimension of sampled_data: %d" % len(df_sampled_data.columns))

    chart_data = df_all_data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("index.html", data=df_all_data)


if __name__ == "__main__":
    df = pd.read_csv('College.csv')
    app.run(debug=True)
