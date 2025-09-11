#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,f1_score,accuracy_score,recall_score,precision_score,classification_report,confusion_matrix,root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

#1.Load the Data and merge the data
Calories_data=pd.read_csv("Calories.csv")
Exercise_data=pd.read_csv("Excersise.csv")
Data=pd.merge(Calories_data,Exercise_data,on='User_ID',how='inner')
print(type(Data))

#Check the data 
print(Data.head())
print(Data.describe())
print(Data.info())

#Information about columns of data
Info=['User_ID','Calories','Gender:Feamle: 0 , Male:1 ','Age (yrs)','Height (cm)','Weight(kg)','Duration (mins)','Heart_Rate ','Body_Temp (C degree)']
for i in range (len(Info)):
   print(Data.columns[i]+"\t"+Info[i])

#Feature engineering
Data['BMI']=Data['Weight']/((Data['Height']/100)**2)
print("New Data of BMI is :",Data['BMI'].head(5))

#Separate the gender
Male_Data= Data[Data['Gender']==1]
Female_Data=Data[Data['Gender']==0]

# Visualizations
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.boxplot(data=Data,x='Gender',y='BMI')
plt.title('BMI by Gender (0:Female, 1:Male)')
plt.show()

sns.boxplot(data=Data,x='Gender',y='Body_Temp')
plt.title('Body_Temp by Gender (0:Female, 1:Male)')
plt.show()

sns.boxplot(data=Data ,x='Gender', y='Calories')
plt.title('Heart Rate by Gender (0:Female, 1:Male)')
plt.show()

sns.boxplot(data=Data ,x='Gender', y='Height')
plt.title('Height by Gender (0:Female, 1:Male)')
plt.show()


sns.boxplot(data=Data ,x='Gender', y='Weight')
plt.title('Weight by Gender (0:Female, 1:Male)')
plt.show()


sns.boxplot(data=Data ,x='Gender', y='Duration')
plt.title('Duration by Gender (0:Female, 1:Male)')
plt.show()


sns.boxplot(data=Data ,x='Gender', y='Age')
plt.title('Age by Gender (0:Female, 1:Male)')
plt.show()

# Descriptive Statistics and Head of both gender
print("Male Descriptive Statistics \n:", Male_Data.describe())
print("Male Descriptive Statistics \n:", Female_Data.describe())
print("Info about male data: \n", Male_Data.head())
print("Info for female data \n", Female_Data.head())

#Predictive Modeling (Predicting Heart Rate) in Randon Forest Regrssion Model

#Male Data for Heart Rate 
X_male=Male_Data.drop(columns=['Heart_Rate','User_ID','Gender','Body_Temp'],axis =1)
Y_male=Male_Data['Heart_Rate']
X_train_male, X_test_male, Y_train_male, Y_test_male = train_test_split (X_male, Y_male, test_size=0.2, random_state=42)
Male_Model=RandomForestRegressor()
Male_Model.fit(X_train_male,Y_train_male)

#Male Model Prediction
Male_Model_predict=Male_Model.predict(X_test_male)

#Model Evaluation
#Male Mean Square Error
Male_mse=mean_squared_error(Y_test_male, Male_Model_predict)
print("Male mean_squared_error for Heart Rate in Regression model: ", Male_mse)
Male_r2=r2_score(Y_test_male, Male_Model_predict)
#Male R2 SCORE
print("Male R2 Score for Heart Rate in Regression model: ", Male_r2)
#Male Predict
print("Male prediction for Heart Rate in Regression model: :\n",Male_Model_predict.flatten())
Male_rmse = root_mean_squared_error(Y_test_male,Male_Model_predict)
#Male Root mean square error
print("Male root_mean_squared_error for Heart Rate in Regression model",Male_rmse)

# Female Data on Heart Rate
X_female=Female_Data.drop(columns=['Heart_Rate','User_ID','Gender','Body_Temp'],axis =1)
Y_female=Female_Data['Heart_Rate']
X_train_female, X_test_female, Y_train_female, Y_test_female = train_test_split (X_female, Y_female, test_size=0.2, random_state=42)
Female_Model=RandomForestClassifier()
Female_Model.fit(X_train_female,Y_train_female)

#Female Model Prediction
Female_Model_predict=Female_Model.predict(X_test_female)

#Model Evaluation
#Female Mean Square Error
Female_mse=mean_squared_error(Y_test_female, Female_Model_predict)
print("Female mean_squared_error for Heart Rate in Regression Model", Female_mse)
#R2 SCORE 
Female_r2=r2_score(Y_test_female, Female_Model_predict)
print("Female R2 Score for Heart Rate in Regression Model", Female_r2)
#Female Model Predict
print("Female prediction for Heart Rate in Regression Model:\n",Female_Model_predict.flatten())
#Female Root Mean Square Error
Female_rmse=root_mean_squared_error(Y_test_female,Female_Model_predict)
print("Female root_mean_squared_error for Heart Rate in Regression Model", Female_rmse)

#PREDICTION SYSTEM 
Inputdata              = ("Enter your : Calories, Age, Height, Weight, Duration of walkout , BMI")
Inputdata_numpy        = np.asarray(Inputdata)
Inputdata_reshaped     =Inputdata_numpy.reshape(1,-1)
Predicition_Male       = Male_Model.predict(Inputdata_reshaped)
Predicition_Female     = Female_Model.predict(Inputdata_reshaped)

#Output will be
print("The patient is Male then  Heart Rate will be :", Predicition_Male)
print("The patient is Female then  Heart Rate will be :", Predicition_Female)




