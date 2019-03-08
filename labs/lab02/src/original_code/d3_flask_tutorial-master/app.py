import json

from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd
#First of all you have to import it from the flask module:
app = Flask(__name__)
#By default, a route only answers to GET requests. You can use the methods argument of the route() decorator to handle different HTTP methods.
@app.route("/", methods = ['POST', 'GET'])
def index():
    #df = pd.read_csv('data.csv').drop('Open', axis=1)
    global df
    #The current request method is available by using the method attribute
    if request.method == 'POST':
        # if request.form['data'] == 'received':
        data = df[['date','open']]
        data = data.rename(columns={'open':'close'})
        print(data)
        print("Hello World!")
        chart_data = data.to_dict(orient='records')
        chart_data = json.dumps(chart_data, indent=2)
        data = {'chart_data': chart_data}
        # data = {'chart_data': chart_data}
        return jsonify(data) # Should be a json string

        # print(request.form['data'])
        # if request.form['data'] == 'received':
        #     data = df[['date','open']]
        #     data = data.rename(columns={'open':'close'})
        #     print(data)
        #     chart_data = data.to_dict(orient='records')
        #     chart_data = json.dumps(chart_data, indent=2)
        #     data = {'chart_data': chart_data}
        #     # data = {'chart_data': chart_data}
        #     return jsonify(data) # Should be a json string
        # else:

    data = df[['date','close']]
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("index.html", data=data)

# @app.route("/member", methods = ['POST', 'GET'])
# def index():
    ###


if __name__ == "__main__":
    df = pd.read_csv('data2.csv')
    app.run(debug=True)
