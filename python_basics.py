# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 07:49:17 2017

@author: prashant
"""
#==============================================================================
# Python Data Types
#    float - Real Numbers
#    int - Integer Numbers
#    str - String/Text
#    bool - True/False or Logical
#==============================================================================

height = 1.79
weight = 70
bmi = height/weight**2

type(height)    # float
type(weight)    # int
type(bmi)       # float

x = "body mass index "
y = "same thing also"

z = x + y
z

type(x)     # str
type(y)     # str
type(z)     # str


#==============================================================================
# List
#    Collection of different types of values
#==============================================================================

# A list can be simply defined by writing comma separated values in sq. bracket
squares_list = [0, 1, 4, 9, 16, 25]
squares_list

# Individual elements can be accessed by writing index no. in square bracket.
# Note: First index is 0
squares_list[2]

# A range of script can be accessed by first and last index separated by colon
squares_list[2:5]
# for m:n it shows (n - m) elements from mth index to n-1 index

# A negative index accesses the index from end
squares_list[-2]

mix_list = ["boy", 1.73, "girl", 1.68, "mom", 1.71, "dad", 1.89]
mix_list

type(mix_list)  # list

## A few common methods applicable on lists are:
# append(), extend(), insert(), remove(), pop(), count(), sort(), reverse()

# Editing elements
mix_list[7] = 1.86

# Cannot add elements directly
mix_list[9] = "Hey"
mix_list = mix_list + ["Hey"]

# Deleting elements
del(mix_list[8])
mix_list

# List are linked. Changing element of one will change element of other (if they are assigned equally)
new_list = mix_list
new_list

new_list[1] = 1.75
new_list
mix_list

# Assigning via the list command will keep both the list separate
new_list2 = list(mix_list)
new_list2

new_list2[1] = 1.80
new_list2
mix_list

#==============================================================================
# Built-in functions
#==============================================================================

# max(), len(), capitalize(), etc.

# List methods
mix_list.index("dad")
mix_list.count(1.75)
mix_list.append("There")
mix_list
mix_list.extend("was")
mix_list

# str methods
sister = 'liz'

sister.capitalize()
sister.upper()
sister.replace('z', 'sa')
sister.find('z')    # -1 means could not be found

# NOTE: Everything is object and every object has method associated with them depend on the type of object (str, list, int, etc.)
