#!/usr/bin/env python
# coding: utf-8

# # Introduction to Tools for Digitalization and Computational Thinking
# 
# There are a range free tools and online services which can be used by teachers and students to support digitalized teaching and learning. 
# 
# One of the most important tools is the Jupyter notebook - an online environment for programming.
# 
# There are shared programming environments available, including https://colab.research.google.com/ and http://cocalc.com, as well as online programming environments built into other learning spaces such as http://Kaggle.com
# 
# 
# ## Digital tools and techniques
# 
# The two principal computer languages which are used for digital activities are Python and R. Both these langauges present powerful ways of manipulating and visualising data, and both have huge support on the web. 
# 
# Jupyter Notebooks work with both Python and R.

# ## Using Python online
# 
# The worksheet for this workshop has been produced from a type of file called a "Jupyter Notebook". This file type can be run in live programming environments. This means that a simple line of code, like:

# In[1]:


print ("hello world")


# Can be run within the same web environment within which students do their learning.
# 
# This means that students can be guided through computational activities with videos, text and code snippets so that they can run code step-by-step
# 

# # Using Python to analyse data from the questionnaires we answered earlier
# 
# Here is a link to a folder in which the excel spreadsheet files can be found which contain data which you typed in response to questions earlier.
# 
# https://alumni-my.sharepoint.com/:f:/g/personal/xgl567_ku_dk/EuwSaUN9SPtEoNjFVa7MgzYBHTmNxFhYRe_BBcNchw-Gzw?e=u6j1df
# 
# Click on the link and choose the spreadsheet file which you want to download
# 

# # Getting started in Python with Data
# 
# We are going to use the Cocalc programming environment. 
# 
# Cocalc can be found at http://cocalc.com
# 

# # Download the file for this notebook
# 
# 1. To use cocalc, you need to import the file for this notebook into cocalc. To do that, click on the download icon at the top right of this page, and select "ipynb"
# 2. To import your file into cocalc, upload the ipynb file.

# # Analyse questionnaire data
# 
# To analyze the questionnaire data you produced in the previous exercise, you need to load the data into Python's data structure, which is called a "Panda dataframe"
# 
# Basically, you need to run the code below, and ensure that the pathname matches the name of the data file you have uploaded
# 

# In[2]:


import pandas as pd

mydata = pd.load_excel ("mydatafile.xlsx", encoding="latin1")


# # Viewing the imported data
# 
# if you have run the code above, then it is easy to see the data that has been imported by simply typing:
# 

# In[ ]:


mydata


# Equally, you can print the contents of the data file with:

# In[ ]:


print (mydata)


# note that these approaches display the data differently.

# # Selecting and filtering specific fields of data
# 
# You can select and filter specific fields of data by specifying which columns you want, and which criteria you want to select

# In[ ]:


import pandas as pd

mydata = pd.load_excel ("mydatafile.xlsx", encoding="latin1")

mydata_selection = mydata[["username"] == "a_user"]

print (mydata_selection)


# This will display just the data where the username column contains the name specified as "a_user" 

# In[29]:


import graphviz 

g = graphviz.Digraph('G', filename='g_c_n.gv')
# g.attr(bgcolor='purple:pink', label='agraph', fontcolor='white')
g.node ("A","test", shape="square")

with g.subgraph(name='cluster1') as c:
    c.attr(fillcolor='blue:cyan', label='identity', fontcolor='white',
           style='filled', gradientangle='270')
    c.attr('node', shape='box', fillcolor='red:yellow',
           style='filled', gradientangle='90')

    c.node ("asdf")
g.edge ("A", "asdf")
g.edge ("asdf", "A")

g.node ("A1","test1", shape="square")

with g.subgraph(name='cluster2') as c:
    c.attr(fillcolor='blue:cyan', label='identity', fontcolor='white',
           style='filled', gradientangle='270')
    c.attr('node', shape='box', fillcolor='red:yellow',
           style='filled', gradientangle='90')

    c.node ("asdf1")
g.edge ("A1", "asdf1")
g.edge ("asdf1", "A1")
# g.edge ("A", "A1")
# g.edge ("A1", "A")

g


# # Doing more complex things
# 
# Obviously, simple selection can be done in Excel, but Python allows us to take deeper control of the data, and reformat it in various ways.
# 
# At the simplest level, you might want to export a new spreadsheet with data for a particular user. 
# 

# In[1]:


import graphviz

dot = graphviz.Digraph(comment='The Round Table')


dot.node('A', 'Study Board', shape='doublecircle')
dot.node('B', 'Tech \nenthusiast', shape="doublecircle")
dot.node('L', 'Employability')
dot.node("I", "innovative law")
dot.node("f", "Law firms")

dot.edge ("A", "B", label="+")
dot.edge ("B","A" ,label="+")
dot.edge('B', 'L', label="-")
dot.edge ("L", "A", label="+")
dot.edge ("I", "f", "+")
dot.edge ("f", "A", "+")
dot.edge ("I", "A", "+")
dot.edge ("I", "B", "+")

dot


# 
# # Exporting data into a new spreadsheet

# In[ ]:


import pandas as pd

mydata = pd.load_excel ("mydatafile.xlsx", encoding="latin1")

mydata_selection = mydata[mydata["username"] == "a_user"]

print (mydata_selection)

mydata_selection.to_excel ("my_new_excel_file.xlsx")

