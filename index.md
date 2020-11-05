## Airbnb Price Prediction Project 


### Introduction

Airbnb is an online platform for listing and renting local homes. It is popular these years as many renters and tenants join this type of business model. One problem that is commonly faced by renters and tenants is the pricing. Appropriate prices will satisfy renters and tenants. 

The (rent) price of a commodity is usually predicted by its component attributes[Griliches, Z, et al]. Previous works of rental price prediction include statistical methods[Azme Bin Khamis, et al], clustering[Brunsdon, Chris E., et al; Rezazadeh, Pouya, et al; Y. Li, et al], sentiment analysis[ Rezazadeh, Pouya, et al], and supervised learning methods[ Rezazadeh, Pouya, et al]. We could find that the methods of machine learning or applying clustering before other methods usually lead to better results for price prediction. Thus, we choose to implement current methods in machine learning to solve this problem. 


### Problem Definition

The problem for rental price estimation is a regression problem. In the set of {(Xi, Yi)}, giving features Xi = <x1, x2, x3, ...>, we want to predict Yi - rental price for the ith item, where x represents different attributes and can be discrete or continuous.

### Data Collection

We collect our data from New York City Airbnb Open Data provided on Kaggle[3], which can be found through this link, https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data. The dataset mentioned above is one of the public datasets that belong to Airbnb, and its original source can be found on this website, http://insideairbnb.com/. The raw dataset contains roughly 48.9K entries, and it consists of 16 distinct features. A detailed description of each feature can be seen in Table 1. 
![image](/datacollection1.png)
![image](/datacollection2.png)

### Methods

For cleaning the dataset before further implementation, we will replace meaningless attributes with dummies to run models later. We may apply feature analysis to remove irrelevant features that may negatively impact our model’s accuracy.

### Part 1. Data Cleaning:
The original dataset imported from Airbnb Kaggle Page is not ideal for modeling; thus a number of prepossessing of data is required. 
![image](/datacleaning.png)

#### 1.1 Remove Null Values
After loading the dataset. We firstly checked for all null values, and we found feature name, host_name, last_review, and reviews_per_month have null values and we replaced the null values with dummies like “NoName”, “NonReview”, or 0 for consistency. 
![image](/removenull.png)

#### 1.2 Remove Price Outliers
When we looked into the histogram for “price” feature, we found most of the data lies below $1000 and only 239 samples have price over $1000 compared with the total of 48.9K entries. Thus, we found it should be helpful to remove those outliers before further modeling. Additionally, the distribution was still right skewed, so we focused on price less than $250 in particular, which is the price range for most rents falling into. In this way, the samples are more Gaussian distributed. By comparison, the refined dataset has a better standard derivation.
![image](/removeprice.png)

### Part 2. Feature Engineering
In data engineering, feature engineering acts as an important role in preprocessing for prediction models. In this process, we need to select relevant features as input to our machine learning models. Common feature engineering methods include filter and wrapper methods [Chandrashekar, Girish, et al], where the first one uses methods like correlation and mutual information to select relevant features and the last one uses a simple model to select features that might give us the best performance such as sequential feature selector (forward/backward selector). 

#### 2.1 Dropping Irrelevant Columns and Feature Visualization
We evaluated a number of features and removed those which are not relevant to “price”.
![image](/drop.png)

#### 2.2 Feature Transformation
For real estate or rent price prediction as noticed by Lu, Sifei, et al, applying log transformations might improve final prediction results. Thus, we perform a log transformation on continuous features. After the log transformation, we will apply a sequential feature selector that is based on lasso regression to select a subset of features that gives us the lowest mean-squared-error. The selector is tested on retaining all features, 90% and 75% of features for 5 times (experiment 1). Finally, we do a PCA analysis on the selected subset of features. This is tested on retaining 100%, 80%, 60%, 50% of dimensions in order to select the best models (experiment 2). These tests show that we should keep all of our features and need to do a pca transformation that keeps 60% of dimensions to improve supervised learning methods’ generalization score. 

### Part 3. Unsupervised Learning:

Unsupervised learning is a great way for us to understand various patterns within the data, so that it can help us later as we try to implement supervised learning algorithms. Indeed, there are two fields in the dataset that can help us to visualize the data, namely longitude and latitude. With these two fields, we are able to plot the data points on the map of New York City, shown in the image below, and then observe the distribution of the data. 

![image](/unsupervised.png)


In order to find patterns within our dataset, we utilize the Gaussian Mixture Model (GMM), K-means Clustering (KMC), and K-nearest neighbors (KNN) algorithms. In this discussion, we talk about the result generated from GMM and will elaborate more on other algorithms in the future. 

### 3.1 GMM
#### Method:
A Gaussian Mixture Model (GMM) is a probabilistic model that assumes all the data points are generated from a finite number of Gaussian distributions. Since we need to deal with a really huge dataset, we use this algorithm in order to find a way to cluster all the data points to different components based on features. Then, we can apply supervised learning algorithms on each component to get a better result because not only the data points in each component are fewer than the original dataset, but also it reduces the complexity of the problem as we cluster the data beforehand. Thus, in this part, we discuss the patterns we have found in the dataset. 

#### Loss Function:
In this section, we also introduce new terminology, Bayesian information criterion (BIC). It is a criterion that we can use to select the best model among a finite number of models [Bayesian information criterion]. The lower the value, the better the model. BIC is quite useful in this scenario because we don’t know how many components we need to create to cluster our data. With BIC, it can tell us how well the model did with this specific number of components.  

#### Assumptions:
We utilize the gmm function from scikit-learn to implement the GMM algorithm. The function takes in one parameter that represents the number of components. Since this gmm function can gradually become time-intensive as the number of components is large, we limit this number to be between 1 and 19. Thus, for every unique dataset we created from the original one, we created 19 models with the number of components for each model to be 1, 2, 3, … 19. 

In the following discussion, we used k to represent the number of components. 


### Result

Since we rely on latitude and longitude to visualize the data points on the map of NYC, the first question that comes to our mind is: do we need to include these two features in our dataset. In order to investigate this question, we first applied GMM to the dataset with these two features. The graph below shows the BIC value for gmm models with different k. It indicates that the k that corresponds to the lowest BIC value shall be 17, with the BIC value being 1,224,315.67. However, using the “elbow method”, the optimal value for k shall be 9. The BIC value is 1,409,026.63 when k is 9.
![image](/result1.png)
![image](/result2.png)

The image shows the belongings of all data points to 9 components. From this graph, it is relatively hard to find any patterns since there are a lot of data points with different colors overlapping with one another. 

Next, we applied the GMM to the dataset without longitude and latitude. The image below shows the same type of graph. The value of k with the lowest BIC is 16, with the BIC value being 938,918.716. Due to the same reason, we choose 6 as the best choice. The corresponding BIC value is 1,278,966.05. 
![image](/result3.png)
![image](/result4.png)

The image above shows the clustering for all the data points within the dataset. Compared to the previous one, we can see a huge amount of blue-colored data points aggregating in Manhattan and Brooklyn. This pattern is much more obvious than the previous graph. 

Thus, as we used the gmm algorithm on the dataset with longitude and latitude and the one without these features, we are quite confident that longitude and latitude shall be included in the dataset since not only the dataset with these two features give us a lower BIC value, meaning a better model, but also the corresponding map shows a more clear pattern. 

Hence, we have figured out the answer to the first question. After that, the second question we need an answer for is: is there a feature in the dataset generating clusters using gmm in a way that there is an obvious boundary between clusters? Since our primary goal is to use the various features within the dataset to predict the price of the housing, will the price be a good candidate for clustering our data?

To answer this question, we created a dataset with only the price field included and repeated the same procedure as described above. 
![image](/result5.png)
![image](/result6.png)

The BIC graph above shows a nearly perfect curve. The k value that corresponds to the lowest value is 18, with the BIC value being 562,866.099. With the same principle, 4 is a better option, and its corresponding BIC is 567,777.56. Furthermore, from the clustering map, we have some interesting findings. Most of the purple and pink data points are clustered in Manhattan and some parts of Brooklyn. Outside these areas, nearly all the data points are orange. Thus, it seems to be a good idea to separate the data points in Manhattan and Brooklyn from other areas due to a boundary between these two groups. 

### Experiments and Results:

#### Experiment 1: Sequential Feature Selector

During Sequential Feature Selection, we need to select a subset of features that give us the best generalization score on test datasets, which is the lowest mean squared error here. We choose to test on retaining 100%, 90%, and 75% of features by lasso regression. In order to get a general trend, we test it for 5 train-test splits: we use a random set of 80% of the transformed data by sequential selector to train lasso regression, and use the rest of transformed data to do tests. The test results are shown in the table below:
![image](/exp1.png)
We notice that in most of the cases, dropping any features will make lasso regression perform worse. By this result, we could assume that dropping any features will make other machine learning methods perform worse. Thus, we choose to keep all features. 

#### Experiment 2: PCA transformation
PCA will transform original data into new axises, which might be in lower dimensions, to get a large sample variance while minimizing the projection distance. However, we do not know whether PCA will make prediction tasks perform better nor do we know the least dimension to keep so that the generalization of models will perform better. Therefore, we compare the generalization score of lasso regression for pca transformed data that keep 100%, 80%, 60%, 50% dimensions with non-pca transformed data. We use 80% of data to fit the PCA and train lasso regression, and the rest to do tests. The results is shown in the table below: 
![image](/exp2.png)
![image](/exp3.png)
Our results show that doing PCA transformation that keeps at least 60% of dimensions will improve lasso regression’s generalization performance. We could also assume that doing PCA transformation keeping 60% of dimensions will improve other supervised machine learning methods’ generalization performance. 


### Discussion

So far we have completed data cleaning, feature selection, and some implementations of the unsupervised learning algorithms, in the next couple of weeks, we will dive deep into supervised learning in order to create a model that accurately predicts the price of the house based on our clustered data. There are several challenges that we need to tackle. First of all, our model might overfit since we have so much data available. Thus, to solve this problem, we might use techniques such as branch pruning, regularization, dropout layer, etc. Second, it is relatively easy for us to utilize different libraries to implement supervised learning methods, but it poses a great difficulty to have an optimal model since we need to spend huge amounts of time tuning the hyperparameter. Finally, we need to decide on an appropriate loss function to evaluate the accuracy of the model. There are so many choices available to us, such as RMSE, MAE, MSE, and LOG_RMSE. We shall select one of them in a way that it will not incur any bias as we compare different models. 

### Future Works:
For supervised learning, we will apply linear regression models(OLS), Lasso Regularization, and Random Forest Regressor. We will evaluate the results on RMSE, F1 scores, etc. Finally, we will tune our parameters to prevent overfitting in training models.


### Reference 

1. Azme Bin Khamis, Azme Azme, and Nur Khalidah Khalilah Binti Kamarudin. INTERNATIONAL JOURNAL OF SCIENTIFIC & TECHNOLOGY RESEARCH ... www.ijstr.org/final-print/dec2014/Comparative-Study-On-Estimate-House-Price-Using-Statistical-And-Neural-Network-Model-.pdf.

2. Brunsdon, Chris E., et al. Geographically Weighted Regression: A Method for Exploring ... 1996,onlinelibrary.wiley.com/doi/abs/10.1111/j.1538-4632.1996.tb00936.x.

3. Chandrashekar, Girish, and Ferat Sahin. "A survey on feature selection methods." Computers & Electrical Engineering 40.1 (2014): 16-28.

4. Dgomonov, (2019) New York City Airbnb Open Data, Version 3. Retrieved September 29, 2020 from https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

5. Griliches, Z., ed., 1971, Price Indexes and Quality Changes: Studies in New Methods of Measurement, Cambridge, Mass: Harvard U. Press.

6. Lu, Sifei, et al. "A hybrid regression technique for house prices prediction." 2017 IEEE International Conference on Industrial Engineering and Engineering Management (IEEM). IEEE, 2017.

7. Rezazadeh, Pouya & Nikolenko, Liubov & Rezaei, Hoormazd. (2019). Airbnb Price Prediction Using Machine Learning and Sentiment Analysis. 

8. Tipping, Michael E., and Christopher M. Bishop. "Probabilistic principal component analysis." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 61.3 (1999): 611-622.

9. Y. Li, Q. Pan, T. Yang and L. Guo, "Reasonable price recommendation on Airbnb using Multi-Scale clustering," 2016 35th Chinese Control Conference (CCC), Chengdu, 2016, pp. 7038-7041, doi: 10.1109/ChiCC.2016.7554467.

10. “Bayesian Information Criterion.” Wikipedia, Wikimedia Foundation, 12 Oct. 2020, en.wikipedia.org/wiki/Bayesian_information_criterion. 
