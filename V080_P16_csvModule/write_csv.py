#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:01:29 2022

@author: rj
"""







"""
csv module
 (▰˘◡˘▰)


In this video, I create a Python script
that collects information of files in a directory,
and then save the result in a CSV file!

Python script collects
    1) file name
    2) word count + character count
    3) file size in bytes
The commands used to collect above information were
covered in previous video: "Run Python Script from Terminal" 

So, I only 'skim' through those commands.  😀
Instead, I am focusing more on steps needed to write CSV file 🚀



Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

# =============================================================================
# Modules needed
# =============================================================================

import sys
import os
import csv  # focus for today's video















# =============================================================================
# EXACTLY 2 arguments allowed  🚨
# 1) python script name: write_csv.py
# 2) name of CSV file output
# =============================================================================

# if exactly 2 arguments are NOT provided
# print/guide proper usage 
# then exit

if len(sys.argv) != 2:
    
    print('\nfor correct script usage, follow this syntax: ') 
    print('python3 write_csv.py <csv_output>',
          end = '\n\n')
    
    exit()


# following is executed if EXACTLY 2 arguments are given:
output_csv = sys.argv[1]
print(f"\nyour output will be saved in file called {output_csv}",
      end = '\n\n')


# Test Cases
    # python3 write_csv.py                    <- only 1 argument 🙅
    # python3 write_csv.py file_info.csv      <- 2 arguments     👍
    # python3 write_csv.py file_info.csv huh  <- 3 arguments     🙅














# =============================================================================
# Hidden/Invisible file...
# =============================================================================

# my Python script will collect information in
# '../txt' directory

# I can list all files in a directory, 
# using os.listdir()
os.listdir('/Users/rj/Desktop/RJstudio/V80/txt')
# ['.DS_Store', 'test1.txt', 'test.txt']


"""
.DS_Store is a 'hidden' (invisible) file.
Please look up "Desktop Services Store" if you are interested~

I do NOT want my Python script to waste time on 'hidden' files.
So I will later 🚨SKIP🚨 them! FYI 
"""












# =============================================================================
# Preparing for CSV output
# =============================================================================

# set up CSV column names
csv_data = [
    ["file_name","char_count","word_count","file_size"]
]


for file in os.listdir('/Users/rj/Desktop/RJstudio/V80/txt'):
# loop through each file in the specified directory
    
    if not file.startswith('.'):
    # 🚨SKIP🚨 'hidden' file: file that starts with '.' , namely .DS_Store
        
        # get the file name
        file_name = file
        
        # open & read file, and then use len() to count characters
        f = open(f'./txt/{file_name}')
        f_read = f.read()  # read() displays the file content
        char_count = len(f_read) 
        
        # find len() of a list that contains every word in the file
        # to count words
        f_read_split = f_read.split()
        word_count = len(f_read_split)
        
        # find file size using os.path.getsize()
        file_size = os.path.getsize(f'./txt/{file_name}')
        
        # put all information above into a list
        csv_row = [file_name, char_count, word_count, file_size]
        
        # print the list to check if everything is collected
        print(csv_row)
        
        # then append it to csv_data
        csv_data.append(csv_row)

# print & check our CSV format data
print(f"\nsee csv_data \n{csv_data}\n")



# Test
    # python3 write_csv.py file_info.csv











# =============================================================================
# Write CSV
# =============================================================================

# open the output file in 'w' mode
# remember that our output filename is stored in output_csv
# (in this video, it is file_info.csv)
with open(f'/Users/rj/Desktop/RJstudio/V80/txt/{output_csv}', 
          'w') as f:
    
    # create csv writer
    writer = csv.writer(f)
    
    # write csv_data!
    # remember our data is stored in csv_data
    writer.writerows(csv_data)


# Test  -  CSV output will show up in txt directory!
    # python3 write_csv.py file_info.csv 



















"""
This is the end of "csv module" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""
















