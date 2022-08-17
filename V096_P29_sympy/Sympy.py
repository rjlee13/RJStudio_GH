#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:37:10 2022

@author: rj
"""

# conda activate spyder-env
# conda install -c anaconda sympy












"""
SymPy!
 (▰˘◡˘▰)


SymPy is a very useful Python Library for lots of (coding) work 
that involves Mathematics 🚀

I wanted to spread / promote this awesome Library to more people 
by showing a simple differentiation example in this short video~! 

(I assume you know how to differentiate)



To learn more about SymPy, please check out:
    https://github.com/sympy/sympy


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

# =============================================================================
# Import sympy
# =============================================================================

import sympy as sp # abbreviating it as sp














# =============================================================================
# Simple Derivative
# =============================================================================
'''
Suppose 
        f(x) = x**2 + 5*x

If we differentiate f with respect to x:
        f'(x) = 2*x + 5        

Let's see if Sympy can do this for us! 🔮
'''

# assign x as sympy Symbol
x = sp.Symbol('x')

# define f(x) like above
f_x = x**2 + 5*x



# differentiate f_x wrt x, using sp.diff()
f_dx = sp.diff(f       = f_x, 
               symbols = x)
f_dx # 2*x + 5 , Nice! 

# we can substitute a value for x and find the final value of f_dx
# find the value of f_dx when x = 1:
f_dx.subs({x:1}) # 7, because 7 = 2*1 + 5















# =============================================================================
# Partial Derivative
# =============================================================================
'''
Suppose we have multivariable function g:
        g(x,y) = x**2 + 5*x*y

Derivative wrt x is:
        g_dx = 2*x + 5*y

Derivative wrt y is:
        g_dy = 5*x

Let's see if Sympy can do this for us! 🔮
'''

# assign y as sympy Symbol (in addition to x from earlier)
y = sp.Symbol('y')

# define g(x,y) like above
g_xy = x**2 + 5*x*y


# Derivative wrt x 
g_dx = sp.diff(g_xy, x)
g_dx # 2*x + 5*y , good!

# let me find the value of g_dx when x = 1 and y = 3
g_dx.subs({x:1, y:3}) # 17, because 17 = 2*1 + 5*3


# Derivative wrt y
g_dy = sp.diff(g_xy, y)
g_dy # 5*x, good!






















"""
This is the end of "SymPy" video~



Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""















