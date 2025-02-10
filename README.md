# Credit Scoring using Machine Learning
## Description
This project is a requirements for completing the junior data science program by Pacman. It focuses on building a credit scoring model using machine learning algorithms like KNN, logistic regression, and XGBoost to predict probability of a person (default borrower) missing the installment payments for 90 days or more beyond the due date within a two-year period.

Workflow of credit scoring model include :
1. Data Pipeline
2. Exploratory Data Analysis (EDA)
3. Data pPreprocessing
4. Model
5. Evaluation

## Dataset 
Source of the dataset from Kaggle, the url link is https://www.kaggle.com/c/GiveMeSomeCredit/overview/description. Dataset contains the 150.000 person with 10 independent variables and 1 dependent variable. Target of the variabel is *SeriousDlqin2yrs*

The description of dataset column :
1. SeriousDlqin2yrs : Person experienced 90 days past due delinquency or worse.
2. RevolvingUtilizationOfUnsecuredLines : Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits.
3. Age : Age of borrower in years
4. NumberOfTime30-59DaysPastDueNotWorse : Number of times borrower has been 30-59 days past due but no worse in the last 2 years.
5. DebtRatio : Monthly debt payments, alimony,living costs divided by monthy gross income.
6. MonthlyIncome : Monthly income.
7. NumberOfOpenCreditLinesAndLoans : Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards).
8. NumberOfTimes90DaysLate : Number of times borrower has been 90 days or more past due.
9. NumberRealEstateLoansOrLines : Number of mortgage and real estate loans including home equity lines of credit.
10. NumberOfTime60-89DaysPastDueNotWorse : Number of times borrower has been 60-89 days past due but no worse in the last 2 years.
11. NumberOfDependents : Number of dependents in family excluding themselves (spouse, children etc.).

## Data Pipeline
Load the dataset from the path file name then read the statistical summary of data for initial stage. dependent or target variabel was defined in this step before data splitting into training, validataion and testing data. the percentage of data splitting are 60% training data, 20% validation data, and 20% testing data. To provide the lightweight pipelining in this project, all the result was saved in pkl format by joblib.

## Exploratory Data Analysis (EDA)
Eda notebook include summary of the data for each column to support actionable decision in data preprocessing. Exploratory data analysis shows data types, outliers, missing values, and visualize distribution of data.

Conclusion :
1. Missing values
`MonthlyIncome` and `NumberofDependents` column have a missing values.

2. Outliers
Visualize distribution data shows outliers for the features, `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`,  `NumberOfTime90DaysLate`, and `RevolvingUtilizationOfUnsecuredLines` needs treatment to handle outliers.

## Data Preprocessing
Handle the missing values in `MonthlyIncome` haven't normal distribution using median and `NumberofDependents` replace by zero, with people haven't dependents assumed.
Drop values in `NumberOfTimes90DaysLate` bigger than 96 to provide the normal distribution.
Drop values bigger than 1.35 (value of upper base) using winsorizing method to remove outliers in `RevolvingUtilizationOfUnsecuredLines`.

Standardize data using StandardScaler() to make the values of data can be read by model, standardizing ensures all values are on the same scale.

Handle the imbalance data using Downsampling (only for data training)

## Model
Build Machine Learning model with different algorithms and parameters :

1. KNN (n_neighbors: 50, 100, 200)
2. Logistic Regression (penalty: 'L1', 'L2'; C: 0.01, 0.1; max_iter: 100, 300, 500)
3. XGBoost (n_estimators: 5, 10, 25, 50)

## Evaluation
Metrics evaluation model using ROC-AUC (Receiver Operating Characteristic-Area Under the Curve) got the best score  :
Best Model : XGBClassifier
Metric Score : 0.86
Best model params : 'n_estimators' = 10 
