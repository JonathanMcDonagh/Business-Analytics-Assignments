# -*- coding: utf-8 -*-
"""
Jonathan McDonagh - 20074520
"""


#1) import in the necessary libraries to run cluster analysis and graph the results
#imports 
import pandas as pd
import numpy
import matplotlib.pylab as plt
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans


#2) load the data file gapmminder.csv into your data frame
#load dataset from the csv file in the dataframe called nesarc_data
gapminder_data = pd.read_csv('gapminder.csv', low_memory=False)


#set PANDAS to show all columns in Data frame
pd.set_option('display.max_columns', None)

#set PANDAS to show all rows in Data frame
pd.set_option('display.max_rows', None)

#upper-case all DataFrame column names
gapminder_data.columns = map(str.upper, gapminder_data.columns)


#3) check for empty values and replace with NaN, convert each column to a number
#replace blanks with Nan
gapminder_data['INCOMEPERPERSON'] = gapminder_data['INCOMEPERPERSON'].replace(" ", numpy.NaN)
gapminder_data['FEMALEEMPLOYRATE'] = gapminder_data['FEMALEEMPLOYRATE'].replace(" ", numpy.NaN)
gapminder_data['INTERNETUSERATE'] = gapminder_data['INTERNETUSERATE'].replace(" ", numpy.NaN)
gapminder_data['LIFEEXPECTANCY'] = gapminder_data['LIFEEXPECTANCY'].replace(" ", numpy.NaN)
gapminder_data['ALCCONSUMPTION'] = gapminder_data['ALCCONSUMPTION'].replace(" ", numpy.NaN)
gapminder_data['URBANRATE'] = gapminder_data['URBANRATE'].replace(" ", numpy.NaN)

#converting strings to numeric data for better output
gapminder_data['INCOMEPERPERSON'] = pd.to_numeric(gapminder_data['INCOMEPERPERSON'],errors='ignore')
gapminder_data['FEMALEEMPLOYRATE'] = pd.to_numeric(gapminder_data['FEMALEEMPLOYRATE'],errors='ignore')
gapminder_data['INTERNETUSERATE'] = pd.to_numeric(gapminder_data['INTERNETUSERATE'],errors='ignore')
gapminder_data['LIFEEXPECTANCY'] = pd.to_numeric(gapminder_data['LIFEEXPECTANCY'],errors='ignore')
gapminder_data['ALCCONSUMPTION'] = pd.to_numeric(gapminder_data['ALCCONSUMPTION'],errors='ignore')
gapminder_data['URBANRATE'] = pd.to_numeric(gapminder_data['URBANRATE'],errors='ignore')

data_clean = gapminder_data.dropna()


#4) subset the data
#subset clustering variables
cluster=data_clean[['INCOMEPERPERSON','FEMALEEMPLOYRATE','INTERNETUSERATE','LIFEEXPECTANCY','ALCCONSUMPTION',
'URBANRATE']]
cluster.describe()


#5) standardize all of the variables
#standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
clustervar['INCOMEPERPERSON']=preprocessing.scale(clustervar['INCOMEPERPERSON'].astype('float64'))
clustervar['FEMALEEMPLOYRATE']=preprocessing.scale(clustervar['FEMALEEMPLOYRATE'].astype('float64'))
clustervar['INTERNETUSERATE']=preprocessing.scale(clustervar['INTERNETUSERATE'].astype('float64'))
clustervar['LIFEEXPECTANCY']=preprocessing.scale(clustervar['LIFEEXPECTANCY'].astype('float64'))
clustervar['ALCCONSUMPTION']=preprocessing.scale(clustervar['ALCCONSUMPTION'].astype('float64'))
clustervar['URBANRATE']=preprocessing.scale(clustervar['URBANRATE'].astype('float64'))


#6) split the data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)


#7) cluster analysis using K-means for 1-9 clusters
# k-means cluster analysis for 1-9 clusters
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(numpy.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
    / clus_train.shape[0])
    
    cdist(clus_train, model.cluster_centers_, 'euclidean')
    
    
#8) plot the curve and determine how many is the fewest clusters that provides a low average distance
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
'''
Here in the plot curve above we can see a couple of bends at the line at 
two clusters and at four clusters. What we can see here in the plot 
above is that the elbow shows where the average distance value might 
be leveling off.
'''


#9) cluster solution for the number of clusters you consider appropriate based on the elbow curve
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()
'''
In this scatter plot above we can see that the three clusters are spaced 
apart showing us they're not densely packed, meaning that the observations 
within the cluster are pretty low correlated with each other.
We can also see that they appear to not have much overlap meaning that there
is very good separation between these clusters.
They also have a high cluster variance. 
'''
