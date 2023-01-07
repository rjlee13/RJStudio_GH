#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 18:34:05 2023

@author: rj
"""











"""
JSON data -> Pandas DataFrame
 (▰˘◡˘▰)


In this video, I use Python `json` module to read JSON data, and then
show one way to convert JSON data into Pandas DataFrame! 🚀


I created a simple JSON file called "json_data.json" in my V123 folder.
It contains simple information about 3 people 🚶‍!


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""
# =============================================================================
# import json & pandas
# =============================================================================

import json
import pandas as pd
















# =============================================================================
# Load my JSON data
# =============================================================================

# path to JSON data (in my V123 folder)
file_path = "/Users/rj/Desktop/RJstudio/V123/json_data.json"

# load my JSON data, using `json.load()`  ; save it as 'data'
with open(file_path, 'r') as file:
    data = json.load(file) 

# Check my JSON data
data
# we see basic information about 3 people

# it is a dictionary 
type(data) # dict

# so we can get its keys & values
data.keys()
data.values()

# len() returns 3 
# since there are 3 keys (person1, person2, person3)
len(data)

# person1's values
data['person1'] # basic information about person1

# person1's name
data['person1']['name']














# =============================================================================
# Convert to Pandas DataFrame
# =============================================================================

# since data is a dictionary, we can print its keys using for loop
for key in data:
    print(key)
# each iteration prints key, one by one:
    # 1st iteration: person1
    # 2nd iteration: person2
    # 3rd iteration: person3


# create an EMPTY pandas DataFrame
# but specify column names
data_pandas = pd.DataFrame(columns = ['name', 'age', 'hobby'])


# for each person (key), SAVE name, age, hobby into a list called 'row'
# and then APPEND row to data_pandas
i = 0
for key in data:
    
    row = [] # list
    row.append(data[key]['name'])  # SAVE name
    row.append(data[key]['age'])   # SAVE age
    row.append(data[key]['hobby']) # SAVE hobby
    
    data_pandas.loc[i,] = row      # APPEND row to data_pandas
    i = i+1

# check our Pandas DataFrame, data_pandas
data_pandas # ✅






















"""
This is the end of "json + pandas" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""
















