#####################################################################
# Name: Yash Patel                                                  #
# File: Switch.py                                                   #
# Description: Cythonized version of Switch class                   #
#####################################################################

class switch(object):
    #################################################################
    # Replicates the behavior of a switch statement (for clarity)   #
    #################################################################
    def __init__(self, value):
        self.value = value
        self.fall = False

    #################################################################
    # Return the match method once, then stop                       #
    #################################################################
    def __iter__(self):
        yield self.match
        raise StopIteration
    
    #################################################################
    # Indicate whether or not to enter a case suite                 #
    #################################################################
    def match(self, *args):
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False