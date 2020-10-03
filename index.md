## Airbnb Price Prediction Project 


### Introduction

Airbnb is an online platform for listing and renting local homes. It is popular these years as many renters and tenants join this type of business model. One problem that is commonly faced by renters and tenants is the pricing. Appropriate prices will satisfy renters and tenants. 

The (rent) price of a commodity is usually predicted by its component attributes[4]. Previous works of rental price prediction include statistical methods[1], clustering[2,5,6], sentiment analysis[5], and supervised learning methods[5]. We could find that the methods of machine learning or applying clustering before other methods usually lead to better results for price prediction. Thus, we choose to implement current methods in machine learning to solve this problem. 


### Problem Definition

The problem for rental price estimation is a regression problem. In the set of {(Xi, Yi)}, giving features Xi = <x1, x2, x3, ...>, we want to predict Yi - rental price for the ith item, where x represents different attributes and can be discrete or continuous.

We will analyze Airbnb open data for NYC provided on Kaggle[3]. The raw dataset consists of 16 distinct features like Airbnb name, neighborhood, location, price etc and thousands of data points.


### Methods

For cleaning the dataset before further implementation, we will replace meaningless attributes with dummies to run models later. We may apply feature analysis to remove irrelevant features which may negatively impact our model’s accuracy.

For supervised learning, we will apply linear regression models(OLS), Lasso Regularization, and Random Forest Regressor. We will evaluate the results on RMSE, F1 scores etc. Finally, we will tune our parameters to prevent overfitting in training models.

To improve the result, we can apply unsupervised learning to capture the non-linearity of the dataset, which is effective in the past literature. For instance, Yang employs Multi-Scale Affinity Propagation to first cluster the landmarks based on popularity and then cluster the houses based on infrastructures, so, after applying linear regression, the result is more promising compared to the one without clustering [6]. 

In order to apply unsupervised learning, the step can be as follows: choose features that we want to cluster on and then apply K-means/GMM/DBSCAN utilizing these features. To determine the features, we can apply principal component analysis to select features that contribute the most to the dataset. After that, we can use supervised learning methods on each cluster.


### Result

A price prediction model that can predict Airbnb prices for renters and tenants based on 16 features including owner information, property specification and customer reviews on the listings. We hope to achieve the best results with low root-mean-squared error (RMSE) and high R^2 score.


### Discussion

Potential difficulty could occur during feature selection. The feature vector of our data has a very high dimension, which could result in a high variance of error. Choosing an effective method to select important features is very crucial but difficult. 

The potential impact is to generate a price prediction model that is reliable and useful to owners and customers when lending or renting Airbnb homes in NYC. 


### Reference 

1. Azme Bin Khamis, Azme Azme, and Nur Khalidah Khalilah Binti Kamarudin. INTERNATIONAL JOURNAL OF SCIENTIFIC & TECHNOLOGY RESEARCH, Comparative-Study-On-Estimate-House-Price-Using-Statistical-And-Neural-Network-Model.

2. Brunsdon, Chris E., et al. Geographically Weighted Regression: A Method for Exploring ... 1996,onlinelibrary.wiley.com/doi/abs/10.1111/j.1538-4632.1996.tb00936.x.

3. [Dgomonov, (2019) New York City Airbnb Open Data, Version 3. Retrieved September 29, 2020](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)

4. Griliches, Z., ed., 1971, Price Indexes and Quality Changes: Studies in New Methods of Measurement, Cambridge, Mass: Harvard U. Press.

5. Rezazadeh, Pouya & Nikolenko, Liubov & Rezaei, Hoormazd. (2019). Airbnb Price Prediction Using Machine Learning and Sentiment Analysis. 

6. Y. Li, Q. Pan, T. Yang and L. Guo, "Reasonable price recommendation on Airbnb using Multi-Scale clustering," 2016 35th Chinese Control Conference (CCC), Chengdu, 2016, pp. 7038-7041, doi: 10.1109/ChiCC.2016.7554467.
