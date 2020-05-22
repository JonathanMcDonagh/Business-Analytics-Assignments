# -*- coding: utf-8 -*-
"""
@author: Jonathan McDonagh / 20074520
"""

#Imports
import pandas
import numpy
import seaborn
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#load dataset from the csv file in the dataframe called nesarc_data
gapminder_data = pandas.read_csv('gapminder.csv', low_memory=False)

#set PANDAS to show all columns in Data frame
pandas.set_option('display.max_columns', None)

#set PANDAS to show all rows in Data frame
pandas.set_option('display.max_rows', None)

#replace blanks with Nan
gapminder_data['lifeexpectancy'] = gapminder_data['lifeexpectancy'].replace(" ", numpy.NaN)
gapminder_data['employrate'] = gapminder_data['employrate'].replace(" ", numpy.NaN)

#converting strings to numeric data for better output
gapminder_data['lifeexpectancy'] = pandas.to_numeric(gapminder_data['lifeexpectancy'],errors='ignore')
gapminder_data['employrate'] = pandas.to_numeric(gapminder_data['employrate'],errors='ignore')

#PART 1: Write the null and alternate hypothesis in your code file (commented in python file)
'''
#H0 means there is a relationship
#H1 means there is not a relationship
'''

#PART 2: Present the mean and std dev for both variables.
#Mean & STD
print("Life Expectancy & Employ Rate")

print('Life Expectancy')
LeGroup = gapminder_data['lifeexpectancy'].mean()
leMean = gapminder_data.groupby('lifeexpectancy').mean()
leSTD = gapminder_data.groupby('lifeexpectancy').std()

print('Mean Life Expectancy - Group')
print(LeGroup)

print('Mean Life Expectancy')
print(leMean)

print('STD Life Expectancy')
print(leSTD)

print("Employ Rate")
ErGroup = gapminder_data['employrate'].mean()
erMEan = gapminder_data.groupby('employrate').mean()
erSTD = gapminder_data.groupby('employrate').std()

print('Mean Employ Rate - Group')
print(ErGroup)

print('Mean Employ Rate')
print(erMEan)

print('STD Employ Rate')
print(erSTD)

#PART 3: Conduct regression analysis to determine the relationship
#regression for association between urbandrate and internet use rate
print('OLS regression model for the association between lifeexpectancy and employrate')
reg1 = smf.ols('employrate  ~ lifeexpectancy ',data=gapminder_data).fit()
print(reg1.summary())

#PART 4: Report on the relationship in terms of p value. (commented in python file)
'''
The p value is 0.000 for both variables which means that it really small and less than 0.05, which is 
generally accepted point at which you reject the null hypothesis, seeing as the here the p value is 
p < 0.0001 this would mean that the likelihood of these results showing a relationship between the variables
coming up in a random distrubution of data is less than 5%.

The intercept is 82.64 
y = slope(x) + intercept

= -0.3440(0) + 82.6425
= 82.64

= -0.3440(1) + 82.6425
= 164.936

The intercept is 164.936 and is statistically significant with a p value of < 0.01
'''

#PART 5: Apply your findings to discover what would the employrrate rate be for a life expectancy age of 80 (commented in python file)
#Moderate positive correlation 
seaborn.regplot(x="lifeexpectancy", y="employrate", fit_reg=True, data=gapminder_data)
plt.xlabel('lifeexpectancy')
plt.ylabel('employrate')
plt.title('Scatterplot for the Association between Employ Rate and Life Expectancy')
'''
We can see with this plot that that there is a decline in employrate with the older people get,
we can see here from the scatter plot that people aged 80 would have a employment rate of about 60.

#H0 means there is a relationship between these two variables seeing as the higher the age is the lower
the employ rate.
'''