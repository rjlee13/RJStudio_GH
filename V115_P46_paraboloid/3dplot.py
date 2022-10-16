#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 17:47:41 2022

@author: rj
"""













"""
Paraboloid
 (▰˘◡˘▰)



In this short video, I plot a Paraboloid, a very famous 3D geometric shape 
taught usually in Multivariate Calculus. 🚀

It is shown in the `Plots` panel  👉


In particular, the mathematical formula of the Paraboloid I plan to plot is:

                    z = x**2 + y**2

-----


Frequently used Numpy functions like `np.linspace` & `np.meshgrid` 
                        And
3 different ways to plot 3D projections will be covered


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

# =============================================================================
# Getting ready~!
# =============================================================================

# import what I need
import numpy as np
import matplotlib.pyplot as plt


# Just to get started, an empty 3D projection Axes can be created like this:
fig = plt.figure()
ax  = plt.axes(projection = '3d')
# 😄

















# =============================================================================
# Numpy: linspace & meshgrid
# =============================================================================

# np.linspace
x = np.linspace(start = 7, 
                stop  = -7,
                num   = 100)

y = np.linspace(start = 7, 
                stop  = -7,
                num   = 100)
# check x & y
x
y
# x & y are arrays with 100 evenly spaced numbers, between -7 and 7 



# np.meshgrid, preparing all possible combinations using x & y from above
X, Y = np.meshgrid(x, y)
X    # each ROW    has 100 evenly spaced numbers, starting from -7 to 7 
Y    # each COLUMN has 100 evenly spaced numbers, starting from -7 to 7 


# Z, our Paraboloid formula using np.meshgrid result X & Y from above
Z = X**2 + Y**2
Z

















# =============================================================================
# ax.contour3D
# =============================================================================


# Paraboloid with 30 levels 
ax  = plt.axes(projection = '3d')
ax.contour3D(X, Y, Z, 
             levels = 30,        # <- 30 levels
             cmap   = 'viridis')
ax.set_title('Paraboloid | 30 levels')


# Paraboloid with 100 levels 
ax  = plt.axes(projection = '3d')
ax.contour3D(X, Y, Z, 
             levels = 100,       # <- 100 levels
             cmap   = 'viridis')
ax.set_title('Paraboloid | 100 levels')
# looks smoother with more levels 


# ------


# Paraboloid, side view / elevation = 0
ax  = plt.axes(projection = '3d')
ax.contour3D(X, Y, Z, 
             levels = 100, 
             cmap = 'viridis')
ax.view_init(elev = 0)
ax.set_title('Paraboloid | elev = 0')


# Paraboloid, 30 degree elevation
ax  = plt.axes(projection = '3d')
ax.contour3D(X, Y, Z, 
             levels = 100, 
             cmap = 'viridis')
ax.view_init(elev = 30)
ax.set_title('Paraboloid | elev = 30')


# Paraboloid, birdview 🦜 / 90 degree elevation
ax  = plt.axes(projection = '3d')
ax.contour3D(X, Y, Z, 
             levels = 100, 
             cmap = 'viridis')
ax.view_init(elev = 90)
ax.set_title('Paraboloid | elev = 90')


# Upsidedown 🙃 Paraboloid
ax  = plt.axes(projection = '3d')
ax.contour3D(X, Y, Z, 
             levels = 100, 
             cmap = 'viridis')
ax.view_init(elev = 180)
ax.set_title('Paraboloid | elev = 180')




# =============================================================================
# 4 Paraboloids together
# =============================================================================

# store 4 elevation values I used above
elevation = [0, 30, 90, 180]

# create a new figure
fig = plt.figure()

# FOR loop to put 4 Paraboloids together!
# using fig.add_subplot()
for i in range(4):
    ax  = fig.add_subplot(2, 2, i+1,     # divide into 4 spaces
                          projection = '3d')
    ax.contour3D(X, Y, Z,                # Paraboloid
                 levels = 100, 
                 cmap = 'viridis')
    ax.view_init(elev = elevation[i])    # 4 elevations
    


















# =============================================================================
# WireFrame & PlotSurface
# =============================================================================

# WireFrame
ax  = plt.axes(projection = '3d')
ax.plot_wireframe(X, Y, Z,  
                  rcount = 25,
                  ccount = 25,
                  color  = 'blue')
ax.set_title('Blue Wireframe')


# PlotSurface
ax  = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z,
                  cmap = 'Paired')


# WireFrame + PlotSurface TOGETHER
ax  = plt.axes(projection = '3d')
ax.plot_wireframe(X, Y, Z,  
                  rcount = 25,
                  ccount = 25,
                  color  = 'blue')
ax.plot_surface(X, Y, Z,
                cmap = 'Paired')

























"""
This is the end of "Paraboloid" video~



Hope you enjoyed it!
 
"""


















