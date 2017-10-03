# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 13:13:18 2017

@author: prashant
"""

## Setting working directory ##
import os
os.chdir("E:/LearningPath/Projects/PythonForDS/code")


""" ********************************************* 
            Python Data Structures 
    ********************************************* """

#### 1. Series and Dataframe ####
    
# Series is a 1-D labelled/indexed array. 
# You can access individual elements of series through these labels

# A dataframe is similar to Excel workbook - columns & row
# Column Names are know as column and Row Numbers as row index

#### 1. Lists ####

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

## A few common methods applicable on lists are:
# append(), extend(), insert(), remove(), pop(), count(), sort(), reverse()


#### 2. Strings ####

# Strings can be identified by use of single('), double (") or triple ("') 
# inverted commas
greeting = 'Hello'
print(greeting[0:2])
print(greeting[2:])
print(len(greeting))
print(greeting+' World')

# Raw stings can be used to pass as is. Python interpretter does not alter the 
# string, if you specify a string to be rw.
stmt = r'\n is a newline character by default.'
print(stmt)

# Python strings are immutable and hence can't be changed.
greeting[1:] = 'i'

## Common string methods are:
# lower(), upper(), strip(), isdigit(), isspace(), find(), replace(), split() and join()


#### 3. Tuples ####

# A tuple is represented by a number of values separated by commas.
# Tuples are immutable and the output is surrounded by parantheses so that 
# nested tuples are processed correctly.
# Additionally, even though tuples are immutable, they can hold mutable data
# Tuples are faster. If possible use tuples, instead of lists

tuple_example = 0, 1, 4, 9, 16, 25
tuple_example
tuple_example[2:]
tuple_example[2] = 6


#### 4. Dictionary ####

# Dictionary is an unordered set of key:value pairs, with the requirement that 
# keys are unique (within a dictionary).
# A pair of braces creates an emmpty dictionary: {}

extensions = {'kunal': 9873, 'Tavish': 9218, 'Sunil': 9223, 'Nitin': 9330}
extensions
extensions['Mukesh'] = 9150     # Adds to the beginning
extensions.keys()
extensions.values()
extensions['Tavish']
extensions.items()









""" *********************************************
    Python Iteration & Conditional Constructs
    ********************************************* """
#==============================================================================
# Syntax:
# for i in [Python Iterable]:
#     expression(i)
# 
#==============================================================================

# Python iterable can be a list, tuple or other advanced data structures
N = 10
fact = 1
for i in range(1, N+1):
    fact *= i

fact

#==============================================================================
# Conditional constructs
# if [condition]:
#     __execution if true__
# else:
#     __execution if false__
#==============================================================================
if N%2 == 0:
    print('Even')
else:
    print("Odd")
 
   



"""
Up Next:
#==============================================================================
# Multiply 2 matrices
# Find the root of a quadratic equation
# Plot bar charts and histograms
# Make statistical models
# Access web-pages
#==============================================================================
"""    

""" ********************************************* 
                Python Libraries
    ********************************************* """
import math as m
from math import * 

m.factorial(N)
# or
factorial(N)    # because we have imported all the namespaces of math library

#==============================================================================
# Following are a list of libraries, you will need for any scientific computations and data analysis:
# 
# 1. NumPy stands for Numerical Python. The most powerful feature of NumPy is n-dimensional array. This library also contains basic linear algebra functions, Fourier transforms,  advanced random number capabilities and tools for integration with other low level languages like Fortran, C and C++
# 2. SciPy stands for Scientific Python. SciPy is built on NumPy. It is one of the most useful library for variety of high level science and engineering modules like discrete Fourier transform, Linear Algebra, Optimization and Sparse matrices.
# 3. Matplotlib for plotting vast variety of graphs, starting from histograms to line plots to heat plots.. You can use Pylab feature in ipython notebook (ipython notebook –pylab = inline) to use these plotting features inline. If you ignore the inline option, then pylab converts ipython environment to an environment, very similar to Matlab. You can also use Latex commands to add math to your plot.
# 4. Pandas for structured data operations and manipulations. It is extensively used for data munging and preparation. Pandas were added relatively recently to Python and have been instrumental in boosting Python’s usage in data scientist community.
# 5. Scikit Learn for machine learning. Built on NumPy, SciPy and matplotlib, this library contains a lot of effiecient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
# 6. Statsmodels for statistical modeling. Statsmodels is a Python module that allows users to explore data, estimate statistical models, and perform statistical tests. An extensive list of descriptive statistics, statistical tests, plotting functions, and result statistics are available for different types of data and each estimator.
# 7. Seaborn for statistical data visualization. Seaborn is a library for making attractive and informative statistical graphics in Python. It is based on matplotlib. Seaborn aims to make visualization a central part of exploring and understanding data.
# 8. Bokeh for creating interactive plots, dashboards and data applications on modern web-browsers. It empowers the user to generate elegant and concise graphics in the style of D3.js. Moreover, it has the capability of high-performance interactivity over very large or streaming datasets.
# 9. Blaze for extending the capability of Numpy and Pandas to distributed and streaming datasets. It can be used to access data from a multitude of sources including Bcolz, MongoDB, SQLAlchemy, Apache Spark, PyTables, etc. Together with Bokeh, Blaze can act as a very powerful tool for creating effective visualizations and dashboards on huge chunks of data.
# 10. Scrapy for web crawling. It is a very useful framework for getting specific patterns of data. It has the capability to start at a website home url and then dig through web-pages within the website to gather information.
# 11. SymPy for symbolic computation. It has wide-ranging capabilities from basic symbolic arithmetic to calculus, algebra, discrete mathematics and quantum physics. Another useful feature is the capability of formatting the result of the computations as LaTeX code.
# 12. Requests for accessing the web. It works similar to the the standard python library urllib2 but is much easier to code. You will find subtle differences with urllib2 but for beginners, Requests might be more convenient.
# 
# Additional libraries, you might need:
# 13. os for Operating system and file operations
# 14. networkx and igraph for graph based data manipulations
# 15. regular expressions for finding patterns in text data
# 16. BeautifulSoup for scrapping web. It is inferior to Scrapy as it will extract information from just a single webpage in a run.
#==============================================================================


  

#==============================================================================
# ## Into to Pandas ##
#==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating a series by passing a list of values
s = pd.Series([1, 3, 5, np.nan, 6, 8])
s

# Creating a dataframe by passing a numpy array, with datetime index & labelled col
dates = pd.date_range('20130101', periods = 6)
dates
df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list('ABCD'))
df

# Creating DataFrame by passing a dict of objects that can be converted to series-like
df2 = pd.DataFrame({'A': 1,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index = list(range(4)), dtype = 'float32'),
                    'D': np.array([3]*4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo' })
df2

# Having specific dtypes
df2.dtypes

# Accessing individual column
df2.D
df2.B

#==============================================================================
# ## Viewing Data ##
#==============================================================================
df.head()
df.tail(3)
df2.C.head(2)

# Displaying index, columns and underlying numpy data
df.index        # Assigned datetime index
df2.index       # By default index

# Quick summary
df.describe()

# Transposing data
df.T

# Sorting by an axis
df.sort_index(axis = 1, ascending = False)

# Sorting by values
df.sort_values(by = 'B')

#==============================================================================
# ## Selection ##
#==============================================================================

# Read about .at, .iat, .loc, .iloc & .ix

## Getting
# Selecting a single column - Series ~ df.A
df['A']
# Selecting by row
df[0:3]
df['20130102':'20130104']

## Selection by Label
# For getting a cross-section using a label
df.loc[dates[0]]

# Selection on multi-axis by label
df.loc[:, ['A', 'B']]

# Showing Label slicing, both endpoints are included
df.loc['20130102':'20130104', ['A', 'B']]

# Reducing in the dimensions of returned object
df.loc['20130102', ['A', 'B']]

# For getting a salar value
df.loc[dates[0], 'A']

# For getting fast access to scalar
df.at[dates[0], 'A']

## Selection by Position
# Selecting via position of passed integers
df.iloc[3]      # 3rd row
df.iloc[3:5, 0:2]   

# By list of integer position locations
df.iloc[[1,2,4], [0,2]]

# For slicing rows explicitly
df.iloc[1:3, :]

# For slicing columns explicitly
df.iloc[:, 1:3]

# For getting a value, explicitly
df.iloc[1,1]


#==============================================================================
# ## Bolean indexing
#==============================================================================
df[df.A > 0]        # All rows where A > 0

# Selecting from DataFrame where a boolean codition is met
df[df > 0]

# Using isin() method for filtering:
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2
df2[df2['E'].isin(['two', 'four'])]


## Setting
# Setting a new column automatically aligns the data by the indexes
s1 = pd.Series([1,2,3,4,5,6], index = pd.date_range('20130102', periods=6))
s1

# Setting values by label
df.at[dates[0], 'A'] = 0

# Setting value by position
df.iat[0, 1] = 0

# Setting by assigning with a numpy array
df.loc[:, 'D'] = np.array([5]*len(df))

df

# A 'where' operation with setting
df2 = df.copy()
df2[df2 > 0] = -df2
df2


#==============================================================================
# ## Missing Data ##
#==============================================================================

# Reindexing allows to change/add/delete the index on a specified axi.
# This returns a copy of the data
df1 = df.reindex(index = dates[0:4], columns = list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
df1

# To drop any rows that have missing data
df1.dropna(how = 'any')

# Filling missing data
df1.fillna(value = 5)

# To get the boolean mask where values are 'nan'
pd.isnull(df1)


#==============================================================================
# ## Operations
#==============================================================================

## Stats
df.mean()

# Same operation on other axis
df.mean(1)

## Apply
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())

## Histogramming
s = pd.Series(np.random.randint(0, 7, size = 10))
s
s.value_counts()


## String Methods
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()

## Concat
df = pd.DataFrame(np.random.randn(10, 4))
df
pieces = [df[:3], df[3:7], df[7:]]
pieces
pd.concat(pieces)

## Join
left = pd.DataFrame({'key':['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key':['foo', 'foo'], 'rval': [4, 5]})
left
right
# Full outer Join
pd.merge(left, right, on = 'key')

# Inner Join
left = pd.DataFrame({'key':['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key':['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on = 'key')


## Append
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]

df.append(s, ignore_index = True)


## Grouping - SQL group by
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                           'foo', 'bar', 'foo', 'foo'],
                       'B' : ['one', 'one', 'two', 'three',
              'two', 'two', 'one', 'three'],
                       'C' : np.random.randn(8),
    'D' : np.random.randn(8)})

df
df.groupby('A').sum()
df.groupby(['A', 'B']).sum()


## Reshaping

# Stack - stack() method compresses a level in dataframe's columns
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz','foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index = index, columns = ['A', 'B'])
df
df2 = df[:4]
df2

stacked = df2.stack()
stacked

# unstack() Unstacks the last level
stacked.unstack(1)





""" *********************************************
        DATA EXPLORATION - PANDAS
    ********************************************* """
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/train_loan.csv")

df.head(5) #Printing first 5 rows

df.describe()   #Summary of numerical variables

# Frequency table for non numeric variables
df['Property_Area'].value_counts()
df['Gender'].value_counts()
df['Married'].value_counts()
df['Education'].value_counts()

# Histogram of application income
df['ApplicantIncome'].hist(bins=50)
# Box plot of Applicant income
df.boxplot(column = 'ApplicantIncome')
# Both plot confirms presence of outliers. This could be due to income disparity in society

df.boxplot(column = 'ApplicantIncome', by = 'Education')
# No substantial diff b/w the mean income of graduates and non-graduates

# Histogram & Boxplot of LoanAmount
df['LoanAmount'].hist(bins = 50)
df.boxplot(column = 'LoanAmount')
# Again, there are some extreme values. Clearly, both ApplicantIncome and LoanAmount
# require some data munging. LoanAmount has missing as well as extreme values

#==============================================================================
# Categorical variable analysis
#==============================================================================

# Pivot table: Chances of getting a loan based on credit card
temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status', index = ['Credit_History'], 
                       aggfunc = lambda x: x.map({'Y':1, 'N':0}).mean())
   
print('Frequency table of credit History:\n', temp1)
print('\nProbability of getting loan for each credit history class:\n', temp2)

# Ploting the pivot table
fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicant')
ax1.set_title('Applicants by Credit History')
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title('Probability of getting loand by credit history')
# This shows that chance of getting a loan increases eight fold if applicant
# has a valid credit history

# Stacked chart of above 
temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind = 'bar', stacked = 'True', color = ['red', 'blue'], grid = False)

# Adding gender to the mix
#temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
#temp3.plot(kind = 'bar', stacked = 'True', color = ['red', 'blue'], grid = False)














