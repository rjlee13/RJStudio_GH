#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 22:01:24 2022

@author: rj



This is material from:
    https://www.youtube.com/playlist?list=PLuHgQVnccGMCVyDPCW8_gXhxuAo44obWW

This is NOT my work

"""








from flask import Flask, request, redirect

app = Flask(__name__)





# =============================================================================
# Data
# =============================================================================


nextId = 4
topics = [
 {'id': 1, 'title': 'html', 'body': 'html is ...'},
 {'id': 2, 'title': 'css', 'body': 'css is ...'},
 {'id': 3, 'title': 'javascript', 'body': 'javascript is ...'}
] 

topics[0]
topics[0].keys()   # ['id', 'title', 'body']
topics[0].values() # [1, 'html', 'html is ...']
topics[0]['title'] # 'html'











# =============================================================================
# Repetitive parts as functions
# =============================================================================

def template(contents, content, id = None):
    
    contextUI = ''
    if id != None:
        contextUI = f'''
            <li><a href = "/update/{id}/">update</a></li>
            <li><form action = "/delete/{id}/" method = "POST"><input type = "submit" value = "delete"></form></li>
        '''
    
    
    return f'''<!doctype html>
    <html>
        <body>
            <h1><a href = "/">WEB</a></h1>
            <ol>
                {contents}
            </ol>
            {content}
            <ul>
                <li><a href = "/create/" >create</a></li>
                {contextUI}
            </ul>
        </body>
    </html>
    '''




def getContents():
    """
    return liTags, hyperlink list
    """
    liTags = ''
    for topic in topics:
        liTags = liTags + \
            f'<li><a href = "/read/{topic["id"]}">{topic["title"]}</a></li>'
            
    return liTags
















# =============================================================================
# HomePage / Index
# =============================================================================

@app.route("/")  # bind a function to a URL
def index():
        
    return template(getContents(), '<h2>Welcome to HomePage</h2>')
















# =============================================================================
# Read Page
# =============================================================================

@app.route("/read/<int:id>/")
def read(id):

    # initiate variables
    title = ''
    body  = ''
    
    for topic in topics:
        if id == topic['id']:
            title = topic['title']
            body  = topic['body']
            break
    
    return template(getContents(), f'<h2>{title}</h2>{body}', id)

    













# =============================================================================
# Create Page
# =============================================================================

@app.route("/create/", methods = ["GET", "POST"])
def create():
    if request.method == "GET":
        content = '''
            <form action = "/create/" method = "POST">
                <p><input type = "text" name = "title" placeholder = "title"></input></p>
                <p><textarea name = "body" placeholder = "body"></textarea></p>
                <p><input type = "submit" value = "create"></p>
            </form>
        '''
        return template(getContents(), content)
    
    elif request.method == "POST":
        global nextId
        
        title = request.form['title']
        body  = request.form['body']
        newTopic = {'id': nextId, 'title': title, 'body': body}
        topics.append(newTopic)
        url = '/read/' + str(nextId) + '/'
        nextId = nextId + 1
        return redirect(url)
    










# =============================================================================
# Update Page
# =============================================================================

@app.route("/update/<int:id>/", methods = ["GET", "POST"])
def update(id):
    if request.method == "GET":
        title = ''
        body  = ''
        
        for topic in topics:
            if id == topic['id']:
                title = topic['title']
                body  = topic['body']
                break
        content = f'''
            <form action = "/update/{id}" method = "POST">
                <p><input type = "text" name = "title" placeholder = "title" value = "{title}"></input></p>
                <p><textarea name = "body" placeholder = "body">{body}</textarea></p>
                <p><input type = "submit" value = "update"></p>
            </form>
        '''
        return template(getContents(), content)
    
    elif request.method == "POST":
        
        title = request.form['title']
        body  = request.form['body']
        
        for topic in topics:
            if id == topic['id']:
                topic['title'] = title
                topic['body']  = body
                break
             
        url = '/read/' + str(id) + '/'
        return redirect(url)














# =============================================================================
# Delete 
# =============================================================================

@app.route("/delete/<int:id>/", methods = ["POST"])
def delete(id):
    for topic in topics:
        if id == topic['id']:
            topics.remove(topic)
            break
    
    return redirect("/")















# =============================================================================
# App Run
# =============================================================================

if __name__ == "__main__":
    app.run(debug = True,  # so that I can refresh anytime I want
            port  = 9999   # go to http://127.0.0.1:9999/
            )
















