from utils_functions import *
from utils_classes import *
from MetaApi import *
import __main__
from flask import Flask, render_template,request , jsonify
import threading

setattr(__main__ , 'BaggingBootstrapper' ,BaggingBootstrapper)


api = MetaApi()

def primary_task():
    api.UpdateHistory()
    api.GenerateSignals()


def secondary_task():
    api.place_order()

def task():
    primary_task()
    secondary_task()


app = Flask(__name__)

@app.route('/')
def Homepage():
    title = 'Algomate-GPT (AI Driven Model)'
    return render_template('index.html', title=title)

@app.route('/ping')
def pinger():
    return jsonify({"ping": "ok"})


@app.route('/process_signals')
def process_signals():

    t = threading.Thread(target=task)
    t.start()

    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True)