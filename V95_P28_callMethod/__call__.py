#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 23:23:38 2022

@author: rj
"""














"""
__call__() in Python Class
 (▰˘◡˘▰)



__call__() is a special built-in function in Python language.


It is "special" in the sense that it can make Python Class 
behave just like a FUNCTION! ✨✨


Instead of explaining in words,
it is much easier to understand if you see an example,
so let's do that right away~


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""


















# =============================================================================
# Here is a simple class to START with...
# =============================================================================
'''
`mathematics` class tells you what 'x' & 'y' are.
and it has a simple function to calculate the product of x & y
'''

class mathematics():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print(f"x is {self.x} ; y is {self.y}")
    
    def product(self):
        print(f"product of x and y is {self.x * self.y}")
                
# let's instantiate above simple class
math = mathematics(x = 3, y = 5) # tells you what x & y are

# product function
math.product()   # 15 

# you get an 🚨ERROR🚨 if you do the following:
math() # TypeError

'''
Now, let me put __call__() inside the class so that 
`mathematics` class can behave like a FUNCTION.

UNlike above, math() will NOT get TypeError!
'''














# =============================================================================
# __call__(self)
# =============================================================================
'''
I will use  __call__(self) to calculate the product of x & y
'''
class mathematics():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print(f"x is {self.x} ; y is {self.y}")

    def product(self):
        print(f"product of x and y is {self.x * self.y}")
    
    
    '''
    __call__(self) ADDED below
    '''
    def __call__(self):
        print(f"product using __call__() is {self.x * self.y}")
        
# instantiate class
math = mathematics(x = 3, y = 5) # tells you x & y


# the following will work THANKS to __call__(self) 🚀
math()           # 15, gives you the product; behaving like a FUNCTION

# math() above is actually a shorthand of this:
math.__call__()  # ALSO gives you 15



        
        










# =============================================================================
# Arguments to __call__()
# =============================================================================
'''
I am going to 'customize' __call__() slightly so that 
it can find the SUM of 2 numbers I give.
'''

class mathematics():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print(f"x is {self.x} ; y is {self.y}")
    
    def product(self):
        print(f"product of x and y is {self.x * self.y}")
    
    '''
    'customizing' __call__() below
    '''
    def __call__(self, a, b):
        print(f"using __call__; sum of {a} and {b} is {a+b}")
        
# instantiate class
math = mathematics(x = 3, y = 5) # tells you x & y

# use __call__()
math(2.5, 5)           # tells you it is 7.5, behaving like a function
math.__call__(2.5, 5)  # same result, just explicitly using __call__() 
        


















"""
This is the end of "__call__()" video~


You will encounter __call__() frequently as you study more in Python!


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""
















        