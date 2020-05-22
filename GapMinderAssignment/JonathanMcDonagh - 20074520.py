# -*- coding: utf-8 -*-
"""
Jonathan McDonagh

20074520
"""

#Imports
import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

##### Data Preparation ##### 
#load dataset from the csv file in the dataframe called nesarc_data
gapminder_data = pandas.read_csv('gapminder.csv', low_memory=False)

#bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format',lambda x:'%f'%x)

#set PANDAS to show all columns in Data frame
pandas.set_option('display.max_columns', None)
#set PANDAS to show all rows in Data frame
pandas.set_option('display.max_rows', None)

# PRINT number of columns and rows
print('data details')
print('fetched ' + str(len(gapminder_data)) + ' rows')  # print length of data
print('fetched ' + str(len(gapminder_data.columns)) + ' columns')  # print number of columns
print()

#replace blanks with Nan
gapminder_data['incomeperperson'] = gapminder_data['incomeperperson'].replace(" ", numpy.NaN)
gapminder_data['internetuserate'] = gapminder_data['internetuserate'].replace(" ", numpy.NaN)
gapminder_data['alcconsumption'] = gapminder_data['alcconsumption'].replace(" ", numpy.NaN)
gapminder_data['lifeexpectancy'] = gapminder_data['lifeexpectancy'].replace(" ", numpy.NaN)
gapminder_data['suicideper100th'] = gapminder_data['suicideper100th'].replace(" ", numpy.NaN)

#converting strings to numeric data for better output
gapminder_data['incomeperperson'] = pandas.to_numeric(gapminder_data['incomeperperson'],errors='ignore')
gapminder_data['internetuserate'] = pandas.to_numeric(gapminder_data['internetuserate'],errors='ignore')
gapminder_data['alcconsumption'] = pandas.to_numeric(gapminder_data['alcconsumption'], errors='ignore')
gapminder_data['lifeexpectancy'] = pandas.to_numeric(gapminder_data['lifeexpectancy'], errors='ignore')
gapminder_data['suicideper100th'] = pandas.to_numeric(gapminder_data['suicideper100th'], errors='ignore')

#my focused countries 
ireland_data = gapminder_data.loc[gapminder_data['country'] == 'Ireland']
netherlands_data = gapminder_data.loc[gapminder_data['country'] == 'Netherlands']
main_countries = ('Ireland', 'Netherlands')

subset_countries = gapminder_data[gapminder_data['country'].isin(main_countries)]

pandas.DataFrame.describe(gapminder_data)

#Creates a html file with just Ireland and Netherlands
c2 = gapminder_data.query('country in ["Ireland", "Netherlands"]').to_html('temp.html')
print("Temp file has been created in the project location with all data for Ireland and Netherlands")


##### Analyse #####
##### Countries and Focused Variables ######
##Ireland 
print('\n----------------------------------------------------------------------------\n')
print("***** IRELAND Focused Variables *****")
print('')
#Count - Income Per Person
print('counts for Income Per Person in Ireland ($) ')
i1 = ireland_data.groupby(['country', 'incomeperperson']).size()
print(i1)
print()

#Count - Internet Use Rate
print('counts for Internet Use Rate in Ireland (Per 100 People) ')
i2 = ireland_data.groupby(['country', 'internetuserate']).size()
print(i2)
print()

#Count - Alcohol Consumption
print('counts for Alcohol Consumption in Ireland (Age = 15+, Litres) ')
i3 = ireland_data.groupby(['country', 'alcconsumption']).size()
print(i3)
print()

#Count - Life Expectancy
print('counts for Life Expectancy in Ireland (Years) ')
i4 = ireland_data.groupby(['country', 'lifeexpectancy']).size()
print(i4)
print()

#Count - Life Expectancy
print('counts for Suicide Per 100th in Ireland (Per 100th) ')
i5 = ireland_data.groupby(['country', 'suicideper100th']).size()
print(i5)
print('\n----------------------------------------------------------------------------\n')

##Netherlands 
print("***** NETHERLANDS Focused Variables *****")
print('')
#Count - Income Per Person
print('counts for Income Per Person in Netherlands ($) ')
n1 = netherlands_data.groupby(['country', 'incomeperperson']).size()
print(n1)
print()

#Count - Internet Use Rate
print('counts for Internet Use Rate in Netherlands (Per 100 People) ')
n2 = netherlands_data.groupby(['country', 'internetuserate']).size()
print(n2)
print()

#Count - Alcohol Consumption
print('counts for Alcohol Consumption in Netherlands (Age = 15+, Litres) ')
n3 = netherlands_data.groupby(['country', 'alcconsumption']).size()
print(n3)
print()

#Count - Life Expectancy
print('counts for Life Expectancy in Netherlands (Years) ')
n4 = netherlands_data.groupby(['country', 'lifeexpectancy']).size()
print(n4)
print()

#Count - Life Expectancy
print('counts for Suicide Per 100th in Netherlands (Per 100th) ')
n5 = netherlands_data.groupby(['country', 'suicideper100th']).size()
print(n5)
print('\n----------------------------------------------------------------------------\n')


#Ireland and Netherlands with variables
print("***** Ireland & Netherlands Variables *****")
print('')
ct1 = subset_countries.groupby(['country', 'incomeperperson', 'internetuserate', 'alcconsumption', 'lifeexpectancy', 'suicideper100th']).size()
print(ct1)
print('\n----------------------------------------------------------------------------\n')

      
##### Descriptive Statistics #####
#To compare with Ireland and Netherlands to see if above or below average of all countries
#descriptive stats for all countries
print('***** Descriptive Stats for all countries *****')
print('')
descDS = gapminder_data[['incomeperperson', 'internetuserate', 'alcconsumption', 'lifeexpectancy', 'suicideper100th']].describe()
print(descDS)
print('')
median = gapminder_data[['country', 'incomeperperson', 'internetuserate', 'alcconsumption', 'lifeexpectancy', 'suicideper100th']].median()
print('Median for all countries')
print(median)
print('\n----------------------------------------------------------------------------\n')
      
      
#descriptive stats for Ireland and Netherlands
print('***** Descriptive Stats for Ireland and Netherlands *****')
print('')
INdescDS = subset_countries[['incomeperperson', 'internetuserate', 'alcconsumption', 'lifeexpectancy', 'suicideper100th']].describe()
print(INdescDS)
print('')
INmedian = subset_countries[['country', 'incomeperperson', 'internetuserate', 'alcconsumption', 'lifeexpectancy', 'suicideper100th']].median()
print('Median for Ireland and Netherlands')
print(INmedian)
print('')
print('Mode for Ireland and Netherlands')
mode = subset_countries[['country', 'incomeperperson', 'internetuserate', 'alcconsumption', 'lifeexpectancy', 'suicideper100th']].mode()
print(mode)
print()


# restrict to those observations that are between the income rate of Ireland and Netherlands
subset1 = gapminder_data[(gapminder_data['incomeperperson']>=26550) & (gapminder_data['incomeperperson']<=27596)]

#descriptive stats for countries with a income between Ireland and Netherlands
print('\n----------------------------------------------------------------------------\n')
print('***** Descriptive stats for Income Per Person for countries with income between Ireland and Netherlands *****')
print('')
IANDNdescDS = subset1[['incomeperperson', 'internetuserate', 'alcconsumption', 'lifeexpectancy', 'suicideper100th']].describe()
print(IANDNdescDS)
print('')
IANDNmedian = subset1[['country', 'incomeperperson', 'internetuserate', 'alcconsumption', 'lifeexpectancy', 'suicideper100th']].median()
print('')
print('Median for countries with income between Ireland and Netherlands')
print(IANDNmedian)
print('')
print('Mode for countries with income between Ireland and Netherlands')
IANDNmode = subset1[['country', 'incomeperperson', 'internetuserate', 'alcconsumption', 'lifeexpectancy', 'suicideper100th']].mode()
print(IANDNmode)
print()
print('\n----------------------------------------------------------------------------\n')

print('***** Counts for incomeperperson between Ireland and Netherlands *****')
c1 = subset1.groupby(['country', 'incomeperperson', 'alcconsumption', 'internetuserate', 'lifeexpectancy', 'suicideper100th']).size()
print('')
print (c1) #Shows 4 countries with income similar to Ireland and Netherlands
print()
subset2 = subset1.copy()
print('\n----------------------------------------------------------------------------\n')

# New variable for Alcohol Consumption in pints (500ml)
print('***** Two new variables *****')
print('')
subset2['AlcoholConsumptionInNumberOfPints'] = subset2['alcconsumption'] * 2
newAlcoconsumptionCount = subset2.groupby(['country', 'AlcoholConsumptionInNumberOfPints']).size()
print(newAlcoconsumptionCount)
print()

# New variable for Life Expectancy in seconds
subset2['LifeExpectancyInSeconds'] = subset2['lifeexpectancy'] * 365 * 24 * 60 * 60
newLifeExpectancyCount = subset2.groupby(['country', 'LifeExpectancyInSeconds']).size()
print(newLifeExpectancyCount)
print('\n----------------------------------------------------------------------------\n')


##### Charts ##### 
#Moderate positive correlation
seaborn.regplot(x="alcconsumption", y="suicideper100th", fit_reg=True, data=gapminder_data)
plt.xlabel('suicideper100th')
plt.ylabel('alcconsumption')
plt.title('Scatterplot for the Association between Alcohol Consumption and Suicide Per 100,000')


#bivariate bar charts
seaborn.catplot(x='country',y='incomeperperson', data=subset2, kind='bar', ci=None)
plt.xlabel('Country')
plt.ylabel('Income Per Person')
plt.title('Income Per Person for Countries with a income between Ireland and Netherlands')

seaborn.catplot(x='country',y='internetuserate', data=subset2, kind='bar', ci=None)
plt.xlabel('Country')
plt.ylabel('Internet Use Rate')
plt.title('Internet Use Rate for Countries with a income between Ireland and Netherlands')

seaborn.catplot(x='country',y='lifeexpectancy', data=subset2, kind='bar', ci=None)
plt.xlabel('Country')
plt.ylabel('Life Expectancy')
plt.title('Life Expectancy for Countries with a income between Ireland and Netherlands')

seaborn.catplot(x='country',y='alcconsumption', data=subset2, kind='bar', ci=None)
plt.xlabel('Country')
plt.ylabel('Alcohol Consumption')
plt.title('Alcohol Consumption for Countries with a income between Ireland and Netherlands')

seaborn.catplot(x='country',y='suicideper100th', data=subset2, kind='bar', ci=None)
plt.xlabel('Country')
plt.ylabel('suicideper100th')
plt.title('Suicide Per 100th for Countries with a income between Ireland and Netherlands')


#Ireland and Netherlands Alcohol Consumption and Internet Use Rate
'''
seaborn.catplot(x='country',y='internetuserate', data=subset_countries, kind='bar', ci=None)
plt.xlabel('Country')
plt.ylabel('Internet Use Rate')
plt.title('Internet Use Rate for Ireland and Netherlands')

seaborn.catplot(x='country',y='alcconsumption', data=subset_countries, kind='bar', ci=None)
plt.xlabel('Country')
plt.ylabel('Alcohol Consumption')
plt.title('Alcohol Consumption for Ireland and Netherlands')
'''

# using ols function for calculating the F-statistic and associated p value
model1 = smf.ols(formula='suicideper100th ~ C(alcconsumption)', data=subset2)
results1 = model1.fit()
print(results1.summary())

#calculate the means and standard deviations for monthly smoking for each category of MAJORDEPLIFE
print('means for suicide per 100,000 by alcohol consumption')

#subset of only two variables
subset3=subset2[['suicideper100th', 'alcconsumption']].dropna()

#Sample size
print(len(subset3)) 


m1=subset3.groupby('suicideper100th').mean()
print(m1)

print('standard deviations for suicide per 100,000 by alcohol consumptionn')
sd1 = subset3.groupby('alcconsumption').std()
print(sd1)
