#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 22:54:12 2022

@author: rj
"""




from flask import Flask
import random


app = Flask(__name__)


@app.route("/")  # bind a function to a URL
def index():
    '''
    returns Hello World
    and also new random numbers every time you refresh page
    '''
    
    return f'''<p>Hello, World!</p> 
               <p>your random number is 
               <strong>{random.random()}</strong></p>'''


if __name__ == "__main__":
    app.run(debug = True,  # so that I can refresh anytime I want
            port  = 9999   # go to http://127.0.0.1:9999/
            )











