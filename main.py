from ClassLib import *
from MetaApp import MetaApi
from flask import Flask, render_template,request , jsonify
import threading
import __main__


# setting Attributes to Main
setattr(__main__ , 'NoiseEnhancer' ,NoiseEnhancer)
setattr(__main__ , 'BaggingBootstrapper' ,BaggingBootstrapper)

Algo = MetaApi()
Crypto = ['AXIS' , 'TCS'   , 'Reliance' , 'Adaniports' , 'MM' , 'CIPLA' , 'JSWSTEEL']

def primary_task():
    Algo.Refresh_Var()
    Algo.symbol_list = Crypto
    Algo.UpdateHistory()
    Algo.GenerateSignals()


def secondary_task():
    Algo.place_order()

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
