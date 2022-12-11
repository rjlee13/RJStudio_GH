





"""
SQLite Browser: csv -> db
 (▰˘◡˘▰)

In this video, I show how to prepare famous iris ☘️ CSV data into (SQLite) database db, using DB Browser for SQLite.


I downloaded DB Browser for SQLite from: https://sqlitebrowser.org/


I also perform basic SQL queries after creating my database. 

Please 🌟PAUSE🌟 the video any time you want to study the commands/steps used.
"""











## Database: rj-db

iris.csv data is in a file called V121.

Now, go to DB Browser for SQLite
And follow steps below:
1) Click New Database 
2) Go to V121 directory to save my db 
3) Write db name: rj-db
4) Cancel Edit Table definition
   (since we import iris.csv)
5) File -> Import -> Table from CSV file
6) Click iris.csv in V121 -> OK

View iris data by clicking "Browse Data"











## Basic Filter & SQL queries

1) 2 ways to filter samples whose Petal.Length is greater than 6
>6
SELECT * from iris where "Petal.Length">6;


2) Create a new integer column called 'is_setosa'
ALTER TABLE iris ADD is_setosa INTEGER;


3) Add 1 in 'is_setosa' column if Species is setosa
UPDATE iris SET is_setosa = 1 WHERE Species='setosa'


4) Delete 'is_setosa' column
ALTER TABLE iris DROP is_setosa;











"""
This is the end of "SQLite Browser" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""











