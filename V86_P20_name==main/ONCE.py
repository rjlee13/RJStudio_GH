#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:24:46 2022

@author: rj
"""













"""
If you are studying / coding in Python,
then you probably have seen 

        if __name__ == '__main__':
    
OR if you have NOT seen it yet, then you will come across sooner or later 😊     


This video is about understanding
        if __name__ == '__main__':


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""





















# =============================================================================
# __name__
# =============================================================================

# __name__ is a "special" variable we can print out!
# print('__name__ is ' + __name__)

    # it says: __name__ is __main__


# Okay, then let me `import` ONCE.py from TWICE.py,
# and see what output we get when we run TWICE.py script






















# =============================================================================
# main() function
# =============================================================================

# very simple main() function:
def main():
    print("I am ONCE.py's MAIN method \n") # just printing




if __name__ == '__main__':
    """
    EXECUTED when this script is run DIRECTLY
        -> python3 ONCE.py
    """
    main()
    
else:    # when __name__ is NOT __main__
    """
    EXECUTED when this script is IMPORTED
        -> python3 TWICE.py
    """
    print("IMPORTED ONCE.py ; NOT executing ONCE.main() \n")


















"""
So 

    `if __name__ == "__main__":` 
    
is a nice way to either prevent OR allow parts of code to be executed!
It is a very important concept to know when designing your code~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""
















