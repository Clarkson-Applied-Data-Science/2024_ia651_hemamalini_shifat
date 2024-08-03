# IA651-MACHINE LEARNING PROJECT 

## AUTISM PREDICTION IN ADULTS

#### Team Members: Hemamalini Venkatesh Vanetha & Shifat Nowrin

### Table of Contents

* [Introduction](#Introduction)
* [Data Overview](#Data-Overview)
* [Exploratory Data Analysis](#Exploratory-Data-Analysis)
* [Encoding Categorical Variables](#Encoding-Categorical-Variables)
* [Train Test Split](#Train-Test-Split)
* [Scaling](#Scaling)
* [Principal Component Analysis](#Principal-Component-Analysis)
* [SMOTE](#SMOTE)
* [Logistic Regression](#Logistic-Regression)
* [Decision Tree](#Decision-Tree)
* [Random Forest](#Random-Forest)
* [Support Vector Classifier](#Support-Vector-Classifier)
* [Conclusion](#Conclusion)
* [Future Scope](#Future-Scope)



# Introduction





# Data Overview

This dataset has been taken from the Kaggle Website

Dataset Link : [Autism Screening on Adults](https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults)

This dataset originally has 704 rows and 21 columns.

The dataset has the following parameters:

| Variables       | Description                                                                                                         |
|-----------------|---------------------------------------------------------------------------------------------------------------------|
| A1 - A10 score  | These columns represent the individual's responses to  10 different questions on the autism screening application.  |
| age             | The Age of the individual                                                                                           |
| gender          | The gender of the individual                                                                                        |
| ethnicity       | The ethnic backgroud of the individual                                                                              |
| jaundice        | Indicates whether the individual has jaundice                                                                       |
| autism          | Indicates whether the person were already diagnosed with  autism or not                                             |
| country_of_res  | Country of residence of the individual                                                                              |
| used_app_before | Indicates whether the individual has used this screening app before                                                 |
| result          | This is the sum of scores(A1 score - A10 score)                                                                     |
| age_desc        | Description of the age group                                                                                        |
| relation        | The person who completed the screening on the app on  behalf of the individual being assessed                       |
| Class/ASD       | The class label indicates whether the individual is likely to  autism                                               |

Among these 21 variables, 20 are independent variables and 'Class/ASD' will be dependent variable.

Columns A1 Score - A10 score is the response score of the questions and these questions were developed by the National Institute for Health Research.

The 10 questions given in this questionnaire is:

Q1: I often notice small sounds when others do not

Q2: I usually concentrate more on the whole picture, rather than the small details

Q3: I find it easy to do more than one thing at once.

Q4: If there is an interruption, I can switch back to what I was doing very quickly

Q5: I find it easy to 'read between the lines' when someone is talking to me

Q6: I know how to tell if someone is listening to me is getting bored

Q7: When I'm reading a story I find it difficult to work out the characters's intentions

Q8: I like to collect information about categories of things (eg.types of car, types of bird, types of train etc)

Q9: I find it easy to work out what someone is thinking or feeing just by looking at their face

Q10: I find it difficult to work out people's intentions

All these questions can be answered Yes or No, which is converted to 0 as No and 1 as Yes.

Below is a sample of this dataset :

| A1_Score | A2_Score | A3_Score | A4_Score | A5_Score | A6_Score | A7_Score | A8_Score | A9_Score | A10_Score | age | gender | ethnicity      | jaundice | autism | contry_of_res | used_app_before | result | age_desc      | relation | Class/ASD |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----|--------|----------------|----------|--------|---------------|-----------------|--------|---------------|----------|-----------|
| 1        | 1        | 1        | 1        | 0        | 0        | 1        | 1        | 0        | 0         | 26  | f      | White-European | no       | no     | United States | no              | 6      | 18 and more   | Self     | NO        |
| 1        | 1        | 0        | 1        | 0        | 0        | 0        | 1        | 0        | 1         | 24  | m      | Latino         | no       | yes    | Brazil         | no              | 5      | 18 and more   | Self     | NO        |
| 1        | 1        | 0        | 1        | 1        | 0        | 1        | 1        | 1        | 1         | 27  | m      | Latino         | yes      | yes    | Spain          | no              | 8      | 18 and more   | Parent   | YES       |
| 1        | 1        | 0        | 1        | 0        | 0        | 1        | 1        | 0        | 1         | 35  | f      | White-European | no       | yes    | United States | no              | 6      | 18 and more   | Self     | NO        |
| 1        | 0        | 0        | 0        | 0        | 0        | 0        | 1        | 0        | 0         | 40  | f      | ?              | no       | no     | Egypt          | no              | 2      | 18 and more   | ?        | NO        |


# Exploratory Data Analysis

Before implementing any of the models for the dataset, we would like to explore the data, their correlations, distribution among categorical and numerical variables , histograms to have more understanding of the dataset and also to see if there are any interesting parts.

First of all, using python we have obtained the summary of the numerical variables using describe() method:

|       | **A1_Score** | **A2_Score** | **A3_Score** | **A4_Score** | **A5_Score** | **A6_Score** | **A7_Score** | **A8_Score** | **A9_Score** | **A10_Score** | **age**    | **result ** |
|-----------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|------------|-------------|
| **count** | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000    | 702.000000 | 704.000000  |
| **mean**  | 0.721591     | 0.453125     | 0.457386     | 0.495739     | 0.498580     | 0.284091     | 0.417614     | 0.649148     | 0.323864     | 0.573864      | 29.698006  | 4.875000    |
| **std**   | 0.448535     | 0.498152     | 0.498535     | 0.500337     | 0.500353     | 0.451301     | 0.493516     | 0.477576     | 0.468281     | 0.494866      | 16.507465  | 2.501493    |
| **min**   | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000      | 17.000000  | 0.000000    |
| **25%**   | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000      | 21.000000  | 3.000000    |
| **50%**   | 1.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 1.000000     | 0.000000     | 1.000000      | 27.000000  | 4.000000    |
| **75%**   | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000      | 35.000000  | 7.000000    |
| **max**   | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000      | 383.000000 | 10.000000   |

### Dataset Cleaning

We tried to check if there are any null values in the dataset.There were 2 null values in the column age, so we dropped it.

Then, we checked for any outliers and found that age has a outlier and removed it from the dataset.

Also, removed the duplicate values. Replaced some values with proper notation like '?' with 'others'. Dropped the age_decs column because it has only one value which is 18 and more.

After all this process, we finally have 696 rows and 20 columns.

### Visualization of Categorical Variables

![Distribution of Gender](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20Gender.png)

![Distribution of Ethnicity](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribution%20of%20Ethnicity.png) 

![Distribution of Jaundice](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribution%20of%20jaundice.png)

![Distribution of Autism](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribution%20of%20Autism.png)

![Distribution of Used_app_before](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20app%20usedpng.png)

![Distribution of Relation](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20relation.png)

![Distribution of Class/ASD](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20Autism.png)

![Distribution of Country of Residence](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20country%20of%20residence.png)

From the plots, we can see the distributions of different categorical variables.

Now, let's analyze the dependent variable plot:

As we can see that there around 200 people with Autism and remaining people doesn't have autism. This is unbalanced.
We will handle this in the later part.

### Visualization of Numerical Variables

![Distribution of Age](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribution%20of%20Age.png)

![Distribution of Result](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20Result.png)

In this result, if the score is above 6, the individual is mostly predicted to have autism.

![Distribution of A1-A10 score](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Count%20plot%20of%20scores.png)

From this count plot, we can see that for different question the scores are distributed and used to predict the autism of the individual.

![Categorizing Age groups](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Age%20group%20plot.png)

This plot shows age group 21-40 has the highest autism count.

After visualizing all the columns,we are dropping result and age_category column.

```python
df1.drop(['result','age_category'],axis = 1, inplace = True)
```

Here, we drop the result column because this directly predicts the y variable which is not very good to use in the model because this will have direct correlation to the y variable.
After this, we'll have 19 columns and 696 rows.

### Correlation Matrix

![Heatmap of columns](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Correlation%20matrix.png)

From the correlation matrix, A9 score,A6 score and A5 score has highly correlated with the dependent variable.

# Encoding Categorical Variables

```python
label_encoder = LabelEncoder()
df1['gender'] = label_encoder.fit_transform(df1['gender'])
df1['jaundice'] = label_encoder.fit_transform(df1['jaundice'])
df1['autism'] = label_encoder.fit_transform(df1['autism'])
df1['used_app_before'] = label_encoder.fit_transform(df1['used_app_before'])
df1['Class/ASD'] = label_encoder.fit_transform(df1['Class/ASD'])
```
We have label encoded the above five variables because they have only 2 values.

```python
df1 = pd.get_dummies(df1, columns=['ethnicity', 'relation','contry_of_res'])
```
One-hot encoding these three variables because they have 3 or more categories.

# Train Test Split

```python
X = df1.drop('Class/ASD', axis = 1)
y = df1['Class/ASD']

X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```

Separating the independent variables and assiging it to X and y.

Splitting the data: 80% of the data will be used for training and 20% of the data will be used for testing the data.

In this way, the model can have best understanding of the predicting variable.

# Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

StandardScaler - A Standard Scaler is a technique used in data preprocessing, particularly in machine learning, to standardize features by removing the mean and scaling to unit variance. This means that it transforms the data in such a way that the mean of each feature becomes 0 and the standard deviation becomes 1.

Here, we scaled both training and testing data.


# Principal Component Analysis

Principal Component Analysis is done to is a dimensionality reduction technique commonly used in data analysis and machine learning.

We will perform PCA and reduce the dimensionality in our data.

We are considering the first 20 Principal Components which explains 38% of the variance.

![Scree Plot](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/PCA%20Scree%20plot.png)

# SMOTE

```python
from imblearn.pipeline import Pipeline

smote = SMOTE(random_state=42)
```

SMOTE (Synthetic Minority Over-sampling Technique) is used to address class imbalance in datasets, where one class is underrepresented compared to others. This imbalance can lead to biased models that perform poorly on the minority class.

In our dataset, one of the class is underrepresented so we are using the smote to balance the data.

























             

