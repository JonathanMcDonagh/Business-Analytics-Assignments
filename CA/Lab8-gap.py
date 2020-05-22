# -*- coding: utf-8 -*-
"""
@author: Jonathan McDonagh / 20074520
"""

#Imports
import pandas
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf

##### Data Preparation ##### 
#load dataset from the csv file in the dataframe called nesarc_data
gapminder_data = pandas.read_csv('gapminder.csv', low_memory=False)

#set PANDAS to show all columns in Data frame
pandas.set_option('display.max_columns', None)

#set PANDAS to show all rows in Data frame
pandas.set_option('display.max_rows', None)

#replace blanks with Nan
gapminder_data['internetuserate'] = gapminder_data['internetuserate'].replace(" ", numpy.NaN)
gapminder_data['urbanrate'] = gapminder_data['urbanrate'].replace(" ", numpy.NaN)

#converting strings to numeric data for better output
gapminder_data['internetuserate'] = pandas.to_numeric(gapminder_data['internetuserate'],errors='ignore')
gapminder_data['urbanrate'] = pandas.to_numeric(gapminder_data['urbanrate'],errors='ignore')


#regression for association between urbandrate and internet use rate
print('OLS regression model for the association between urbanrate and internet use rate')
reg1 = smf.ols('internetuserate ~ urbanrate',data=gapminder_data).fit()
print(reg1.summary())