#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:36:11 2022

@author: rj


conda create -n spyder-env -y
conda activate spyder-env
conda install requests -y

conda install <module name>
"""

#[print(i) for i in [12,2,3]]
#[2*x for x in [1,2,3]]












"""
YouTube Video Titles
 (▰˘◡˘▰)

In this video I am going to use 
1. Selenium
2. Beautiful Soup 4 🥣

to extract my RJStudio YouTube video titles!

Both Selenium & Beautiful Soup 4 are popular & commonly used modules
for "Web Scraping" 🚀


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""

# =============================================================================
# Modules needed
# =============================================================================

from selenium import webdriver
from bs4 import BeautifulSoup # 🥣















# =============================================================================
# ChromeDriver
# I use Chrome as my default browser! :D 
# =============================================================================

# Create a Chrome Driver!
    # To do so, I have to download the ChromeDriver executable 🏭
    # https://chromedriver.chromium.org/downloads <- downloaded from here
    # then give the PATH to the executable
chrome_driver = webdriver.Chrome(
    executable_path = '/Users/rj/Desktop/RJstudio/V85/chromedriver')
# Chrome initiated itself!! 🔥
# notice how it says:
    # "Chrome is being controlled by automated test software."
    # Don't worry too much about it!


# Now, let's go to my YouTube channel's "VIDEOS" section
# I give my YouTube URL as the argument:
chrome_driver.get(url = 'https://www.youtube.com/c/RJStudio13/videos')

















# =============================================================================
# Page Source
# We can view it directly from Chrome:
    # right click -> Inspect -> Element (shown by default)
# =============================================================================

# I can retrieve page source using `.page_source`
YT_source = chrome_driver.page_source
# check page source
YT_source
# it's super long... 🫠
# Beautiful Soup 🥣 can help us find the information we want soon below!


# close Chrome since we don't need it anymore
chrome_driver.close()
# Chrome closed!




















# =============================================================================
# Beautiful Soup to the rescue 🆘
# =============================================================================

"""
As we saw a little earlier,
YT_source is TOO LONG for us to parse manually... 🫠
Beautiful Soup can find what I need right away! 🌟

first, we need to find out 
HOW to locate YouTube video titles in YT_source!
right-click on a title -> Inspect

Notice YouTube titles are located at
<a id="video-title" ...>

okay, now I know where to find titles, and BeautifulSoup can help!
"""


# Beautiful Soup to the rescue 🆘
# parse YT_source with HTML parser
YT_soup = BeautifulSoup(markup   = YT_source, 
                        features = "html.parser")


# find all 'a' tag with id = 'video-title' to look for
# <a id="video-title" ...>
YT_titles = YT_soup.find_all('a', id = 'video-title')
# check our result!
YT_titles 

# let's take a look at first 2 titles, should be 2 most recent videos
YT_titles[:2]



# hmm, I still think there is too much redundant info!
# let me do a simple string manipulation to make it more concise~
YT_titles_concise = [str(x).split("title=",1)[1] for x in YT_titles]

# take a look at the first 2 titles now
YT_titles_concise[:2] # better!


















"""
This is the end of "YouTube Video Titles" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""














