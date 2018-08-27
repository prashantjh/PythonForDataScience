# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 07:49:17 2017

@author: prashant
"""

"""
Python programs can be decomposed into modules, statements,
	expressions and objects
	1. Programs are composed of modules
	2. Modules contain statements
	3. Statements contain expressions
	4. Expressions create and process objects

"""

#======================================================================
# Python Data Types
#    float - Real Numbers
#    int - Integer Numbers
#    str - String/Text
#    bool - True/False or Logical
#
# Python Object Types
#	 Numbers - 123, 12.3, 3+4j, 0b111, Decimal(), Fraction()
# 	 Strings - 'spam, "Bob's", u'sp\xc4m'
#	 Lists - [1], [1, [2, 3], 3.1], list(range(10))
#	 Dictionary - {'food' : 'eat', 'wine' : 'drink'}
# 	 Tuples -	(1,), (1, 'spam', 4, 'U'), tuple('spam')
# 	 Sets - set('abc'), {'a', 'b', 'c'}
#	 Files - open('eggs.txt'), open(r'C:\eggs.txt', 'wb')
#======================================================================

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


#======================================================================
# List
#    Collection of different types of values
#======================================================================

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

#### Built-in functions ####

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


#======================================================================
# String
#    Strings are sequences (a positionally ordered collection of other objects) of one-character strings
#======================================================================

## Sequence Operations:
S = 'Spam'
len(S)

S[0]  # Indexing
S[-1]

S[1:] # Slicing
S[0:2]
S[:-1]
S[:]

S + 'xyz' # Concatenation
S*8     # Repetition

S[0] = 'z'    # IMMUTABILITY


### Built-in Methods ###

S.find('pa')
S.replace('pa', 'XY')
S   # Original string unchanged

S.upper()
S.isalpha()     # Content tests: isalpha, isdigit, etc

line = "aaa,bbb,ccc,dddddd\n"
line.split(',')     # Split on a delimiter into a list of substrings
line.rstrip()       # Remove white space character on right side
line.rstrip().split(',')    # Combining two operations

## Formating Expressions
'%s, eggs, and %s' %('spam', 'SPAM!')   # Formatting by type
'{0}, eggs, and {1}'.format(S, 'SPAM!') # Formatting by position
'{}, eggs, and {}'.format('spam', 'SPAM!') # Numbers optional


#======================================================================
# Dictionary
#    In python dictionaries are not sequences at all, but instead known as mappings
#    Mappings are also collections of other objects and store them as key:value pair
#======================================================================

# Example
D = {'food': 'Spam', 'quantity': 4, 'color': 'pink'}
D['food']
D['quantity'] += 1
D

# Dictionary creation - create keys by assignment
D = {}
D['name'] = 'Bob'
D['job'] = 'dev'
D['age'] = 40
D
print(D['name'])

D.keys()    # All keys
D.values()  # All values

# Dictionary creation - using dict function
bob1 = dict(name = 'Bob', job = 'dev', age = 40)
bob1

# Dictionary creation - using Zipping
bob2 = dict(zip(['name', 'job', 'age'], ['Bob', 'dev', 40]))
bob2


## Nesting ##
rec = {'name': {'first' : 'Bob', 'last': 'Smith'},
       'jobs': ['dev', 'mgr'],
       'age': 40.5}

rec['name']     # Name is a nested dictionary
rec['name']['last']     # Index the nested dictionary
rec['jobs']     # Jobs is a nested list
rec['jobs'][-1] # Index the nested list
rec['jobs'].append('janitor')   # Expand Bob's job description in place
rec

rec = 0     # Memory automatically cleaned by assigning object to something else


## Sorting Keys: for Loops ##
D = dict(a = 1, c = 2, b = 3)
D

Ks = list(D.keys()) # Unordered keys list
Ks
Ks.sort()       # Sorted keys list
Ks

for key in Ks:
    print(key, '=>', D[key])

# Sorting and iterating through loop in one step
for key in sorted(D):
    print(key, '=>', D[key])



#======================================================================
# Tuples
#    Tuples are like list that cannot be changed
#    Tuples are sequences, like lists, but are IMMUTABLE like strings
#======================================================================

T = (1, 2, 3, 4)    # A 4-item tuple
T
len(T)

T + (5, 6)      # Concatenation
T[0]        # Indexing
T.index(4)

T[1:4]      # Slicing

T.count(4)

T[0] = 2    # IMMUTABLE
T = (2, ) + T[1:]   # Make a new tuple for a new value
T



#======================================================================
# Files
#    File objects are Python code's main interface to external files on your computer.
#    They can be used to read and write text memos, audio clips, Excel documents, saved email etc.
#======================================================================


## Creating a file and Writing to it ##
f = open('./data.txt', 'w')     # Make a new file in output mode ('w' is write)
f.write('Hello\n')  # Write strings of characters to it
f.write('World\n')  # Returns no. of items written
f.close()       # Close to flush output buffers to disk

## Reading from a file ##
f = open('./data.txt')  # 'r' (read) is the default processing mode
text = f.read()     # Read entire file into a string
text
print(text)     # Print intercepts control characters

text.split()
f.close()

# Control flow through file
for line in open('./data.txt'): print(line)

# File methods
dir(f)
help(f.buffer)


## BINARY files ##
# Note: Binary files are useful for processing media, accesing data created by C programs, and so on.
    # Python's struct module can both create and unpack binary data

import struct
packed = struct.pack('>i4sh', 7, b'spam', 8)    # Create packed binary data
packed      # 10 bytes, not objects or text

# Writing binary files #
file = open('data.bin', 'wb')   # Open binary output file
file.write(packed)  # Write packed binary data - 10 bytes
file.close()

# Reading binary files 
data = open('data.bin', 'rb').read()    # Open/read binary file
data
data[4:8]   # Slice bytes in the middle
list(data)  # A sequence of 8-bit bytes
struct.unpack('>i4sh', data)    # Unpack into objects again

















