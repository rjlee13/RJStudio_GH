#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:42:59 2022

@author: rj
"""













"""
sqlite3: SQLite Database -> Pandas DataFrame
 (▰˘◡˘▰)


sqlite3 module makes it easy for us to examine SQLite databases in Python 🚀

In this video, I show how to view SQLite database in Python, and then
convert it into Pandas DataFrame!



I prepared SQLite iris 🪴 database (iris.db) in my folder called V122.


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""























# =============================================================================
# Intro to sqlite3 module
# =============================================================================

# import!
import sqlite3

# create a Connection that represents the database
connect = sqlite3.connect("/Users/rj/Desktop/RJstudio/V122/iris.db")

# create a Cursor object 
cursor = connect.cursor()

# use execute() method to perform SQL commands
cursor.execute("SELECT * FROM iris;")
cursor.fetchall() # fetch all rows

# SQL command 
cursor.execute("SELECT * FROM iris;")
cursor.fetchone()  # fetch first row only

# SQL command 
cursor.execute("SELECT * FROM iris where species='versicolor';")
# for loop, another way to print out all rows 
for row in cursor:
    print(row)
# we can see all rows where species='versicolor'















# =============================================================================
# Extract Column Names
# =============================================================================

# we can get table information using the following command
cursor.execute("PRAGMA table_info(iris);")
cursor.fetchall()
# notice column names are shown


# extract column name element for each row using for loop!
cursor.execute("PRAGMA table_info(iris);")
for row in cursor:
    print(row[1])


# so create an empty list and append column name from each row
column_name = [] # empty list
cursor.execute("PRAGMA table_info(iris);")
for row in cursor:
    column_name.append(row[1]) # append column name

# check column names
column_name # column names collected, ✅
















# =============================================================================
# Creating Pandas 🐼 DataFrame 
# =============================================================================

# import pandas
import pandas as pd

# SQL query to get all rows
cursor.execute("SELECT * FROM iris;")
# Fetch all and then put it into DataFrame
iris_pd = pd.DataFrame(cursor.fetchall()) 

# check our Pandas DataFrame
iris_pd # notice there is NO proper column names


# BUT we collected column names earlier, column_name
column_name # list of column names

# so repeat the above process EXCEPT include `columns = column_name`
cursor.execute("SELECT * FROM iris;")
iris_pd = pd.DataFrame(cursor.fetchall(),
                       columns = column_name)

# NOW we do have proper column names!
iris_pd

# finally close our Connection
connect.close()



















"""
This is the end of "sqlite3" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""

















