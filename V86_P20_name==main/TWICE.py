#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:40:38 2022

@author: rj
"""



import os
os.chdir("/Users/rj/Desktop/RJstudio/V86")




















# =============================================================================
# __name__
# =============================================================================

# import from ONCE.py, it will run the print() function I wrote in ONCE.py
# import ONCE
    # Notice it has returned: __name__ is ONCE
    # (as OPPOSED to '__name__ is __main__')



"""
So, from our experiment, we can conclude that
value of __name__ will CHANGE depending on HOW the script is run


1) When script is run DIRECTLY:
    __name__ is __main__

2) When script is IMPORTED:
    __name__ is <name-of-imported-script>
"""



















# =============================================================================
# main() function when IMPORTED
# =============================================================================


print("\nwe are in TWICE.py script \n")
import ONCE
    # Notice it has returned: "IMPORTED ONCE.py ; NOT executing ONCE.main()"
    # which was under ELSE statement




# if you want to run ONCE's main() function in TWICE.py, then
# ONCE.main()














