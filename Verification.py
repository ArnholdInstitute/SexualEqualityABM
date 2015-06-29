#####################################################################
# Name: Yash Patel                                                  #
# File: Verification.py                                             #
# Description: All methods related to performing verifications on   #
# parameters used for initializing the different objects are written#
# here, as they are equivalent across objects                       #
#####################################################################

#####################################################################
# Given variable, performs a generic check on if the variable is of #
# type typeCheck                                                    #
#####################################################################
def Verification_genericVerify(var, text, typeCheck, typeText):
    if not isinstance(var, typeCheck):
        sys.stderr.write("{} must be of "
            "type {}".format(text, typeText))
        return False
    return True

#####################################################################
# Given variable, ensures it is a float. If not, use text to provide#
# an error message                                                  #
#####################################################################
def Verification_verifyFloat(var, text):
    return Verification_genericVerify(var, text, float, "float")

#####################################################################
# Given variable, ensures it is a bool. If not, uses text to provide#
# an error message                                                  #
#####################################################################
def Verification_verifyBool(var, text):
    return Verification_genericVerify(var, text, bool, "bool")

#####################################################################
# Given variable, ensures it is a int. If not, uses text to provide #
# an error message                                                  #
#####################################################################
def Verification_verifyInt(var, text):
    return Verification_genericVerify(var, text, int, "int")

#####################################################################
# Given variable, ensures it is a str. If not, uses text to provide #
# an error message                                                  #
#####################################################################
def Verification_verifyStr(var, text):
    return Verification_genericVerify(var, text, str, "str")

#####################################################################
# Given variable, ensures it is between 0.0 and 1.0. If not, text is#
# used to provide an error message                                  #
#####################################################################
def Verification_verifyInBounds(var, text):
    if var < 0.0 or var > 1.0:
        sys.stderr.write("{} must be given on " +
            "0.0-1.0 scale".format(text))
        return False
    return True