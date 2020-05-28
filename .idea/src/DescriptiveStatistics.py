import pandas as pd
from feature_engine.discretisers import EqualFrequencyDiscretiser
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

# read file
file = pd.read_csv(
    'C://Users/jarda/OneDrive - Vysoká škola ekonomická v Praze/2. semestr ZWT/4IZ450 - Dobývání znalostí z databází'
    '/DruhaSemestralka/BankingData.csv')

data = pd.DataFrame(file)

print(data.head(10))
print(list(data.columns))

# Derivation of new variables
data['UseBranch'] = data['Branch_Transactions'].map(lambda x: 0 if x == 0 else 1)
data['UseATM'] = data['ATM_Transactions'].map(lambda x: 0 if x == 0 else 1)
data['UsePhone'] = data['Phone_Transactions'].map(lambda x: 0 if x == 0 else 1)
data['UseInternet'] = data['Phone_Transactions'].map(lambda x: 0 if x == 0 else 1)
data['UseStandingOrder'] = data['Standing_Orders'].map(lambda x: 0 if x == 0 else 1)

# Discretisation
discretizer = EqualFrequencyDiscretiser(q=4, variables=['Age', 'Tenure'])
discretizer.fit(data)
data = discretizer.transform(data)

# Dichotomisation
data_dichotomized = pd.get_dummies(data[['Age', 'Tenure', 'Gender']].astype(str))
data = pd.concat([data, data_dichotomized], axis=1)
print(data.head(10))
print(list(data.columns))

print(data[['Age', 'Tenure']].head(10))

X_train, X_test, y_train, y_test = train_test_split(data[['Age_0', 'Age_1', 'Age_2',
                                                          'Current_Amount', 'Home_Loan_Amount', 'Gender_M',
                                                          'Saving_Amount', 'Stocks_Amount',
                                                          'Tenure_0', 'Tenure_1', 'Tenure_2',
                                                          'UseATM', 'UseBranch', 'UseInternet',
                                                          'UsePhone', 'UseStandingOrder']],
                                                    data['New_Credit_Card_Flag'],
                                                    test_size=0.2, random_state=42)


# Balancing training dataset with SMOTE method using oversampling
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
print(sorted(Counter(y_train).items()))

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

prediction_lr = lr.predict(X_test)
print(accuracy_score(prediction_lr, y_test))
print(confusion_matrix(prediction_lr, y_test, labels=[0, 1]))
