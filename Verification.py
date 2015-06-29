#####################################################################
# Name: Yash Patel                                                  #
# File: Verification.py                                             #
# Description: All methods related to performing verifications on   #
# parameters used for initializing the different objects are written#
# here, as they are equivalent across objects                       #
#####################################################################

#####################################################################
# Given variable, ensures it is a float. If not, use text to provide#
# an error message                                                  #
#####################################################################
def Verification_verifyFloat(var, text):
    if not isinstance(var, float):
        sys.stderr.write("{} must be of "
            "type double/float".replace(text))
        return False
    return True

#####################################################################
# Given variable, ensures it is a bool. If not, uses text to provide#
# an error message                                                  #
#####################################################################
def Verification_verifyBool(var, text):
    if not isinstance(var, bool):
        sys.stderr.write("{} must be of "
            "type bool".replace(text))
        return False
    return True

#####################################################################
# Given variable, ensures it is between 0.0 and 1.0. If not, text is#
# used to provide an error message                                  #
#####################################################################
def Verification_verifyInBounds(var, text):
    if var < 0.0 or var > 1.0:
        sys.stderr.write("{} must be given on " +
            "0.0-1.0 scale".replace(text))
        return False
    return True