#LOAN APPROVAL PREDICTION MODEL
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from sklearn.metrics import r2_score
df=pd.read_csv('loan_approval_dataset2.csv')
print(df)
# print(df.head(5))
df['education'] = df['education'].str.strip().replace({'Not Graduate':0,'Graduate':1})
df['loan_status'] = df['loan_status'].str.strip().replace({'Rejected':0,'Approved':1})
df['self_employed'] = df['self_employed'].str.strip().replace({'No':0,'Yes':1})
#subplot 1
# x=df['education']
# y=df['loan_status']
# plt.subplot(2,3,1)
# plt.bar(x,y,color=['violet','green'])
# front1={
#                   'color':'blue',
#                   'size':10,
#                   'family':'Arial'
# }
# front2={
#                   'color':'red',
#                   'size':15,
#                   'family':'Arial'
# }
# front3={
#                   'color':'violet',
#                   'size':15,
#                   'family':'Arial'
# }
# plt.title("education vs Loan Status",fontdict=front1)
# #plot2
# x=df['self_employed']
# y=df['loan_status']
# plt.subplot(2,3,2)
# plt.hist(df['self_employed'],color='red')
# front1={
#                   'color':'blue',
#                   'size':10,
#                   'family':'Arial'
# }
# front2={
#                   'color':'red',
#                   'size':15,
#                   'family':'Arial'
# }
# front3={
#                   'color':'violet',
#                   'size':15,
#                   'family':'Arial'
# }
# plt.title("Self Employed vs Loan Status",fontdict=front1)
# #plot3
# x=df['commercial_assets_value']
# y=df['loan_status']
# plt.subplot(2,3,3)
# plt.hist(df['commercial_assets_value'])
# front1={
#                   'color':'blue',
#                   'size':10,
#                   'family':'Arial'
# }
# front2={
#                   'color':'red',
#                   'size':15,
#                   'family':'Arial'
# }
# front3={
#                   'color':'violet',
#                   'size':15,
#                   'family':'Arial'
# }
# plt.title("Commercial Assets Value vs Loan Status",fontdict=front1)
# #plot4
# x=df['luxury_assets_value']
# y=df['loan_status']
# plt.subplot(2,3,4)
# plt.scatter(x,y)
# plt.grid()
# front1={
#                   'color':'blue',
#                   'size':10,
#                   'family':'Arial'
# }
# front2={
#                   'color':'red',
#                   'size':15,
#                   'family':'Arial'
# }
# front3={
#                   'color':'violet',
#                   'size':15,
#                   'family':'Arial'
# }
# plt.title("Luxury Assets Value vs Loan Status",fontdict=front1)
# #plot5
# x=df['bank_asset_value']
# y=df['loan_status']
# plt.subplot(2,3,5)
# plt.hist(df['bank_asset_value'])
# front1={
#                   'color':'blue',
#                   'size':10,
#                   'family':'Arial'
# }
# front2={
#                   'color':'red',
#                   'size':15,
#                   'family':'Arial'
# }
# front3={
#                   'color':'violet',
#                   'size':15,
#                   'family':'Arial'
# }
# plt.title("Bank Assets Value vs Loan Status",fontdict=front1)
# plt.show()
#model building
X=df[['education','self_employed','commercial_assets_value','luxury_assets_value','bank_asset_value','income_annum','loan_amount','loan_term','cibil_score','residential_assets_value']]
Y=df['loan_status']
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=1000)
model.fit(X_train,Y_train)
#prediction check
predictions=model.predict(X_test)
print("Prediction values are :",predictions)
#accuracy check
accuracy=accuracy_score(Y_test,predictions)
print("Accuracy is=============>",accuracy)
education=int(input("Enter education (0 for Not Graduate, 1 for Graduate): "))
self_employed=int(input("Enter self employed (0 for No, 1 for Yes): "))
commercial_assets_value=float(input("Enter commercial assets value: "))
luxury_assets_value=float(input("Enter luxury assets value: "))
bank_asset_value=float(input("Enter bank assets value: "))
income_annum=float(input("Enter income per annum: "))
loan_amount=float(input("Enter loan amount: "))
loan_term=int(input("Enter loan term (in months): "))
cibil_score=int(input("Enter cibil score: "))
residential_assets_value=float(input("Enter residential assets value: "))
loan_status=model.predict([[education,self_employed,commercial_assets_value,luxury_assets_value,bank_asset_value,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value]])
print("Loan status is:",loan_status)
if loan_status==0:
         print("========Loan Rejected========")
else:
                  print("========Loan Approved========")
with open ('Loan_model.pkl', "wb") as file:
                  pickle.dump(model,file)                  

# X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
# model=LogisticRegression()
# model.fit(X_train,Y_train)
# #prediction check
# predictions=model.predict(X_test)
# print("Prediction values are :",predictions)
# #accuracy check
# accuracy=accuracy_score(Y_test,predictions)
# print("Accuracy is=============>",accuracy)
# education=int(input("Enter education (0 for Not Graduate, 1 for Graduate): "))
# self_employed=int(input("Enter self employed (0 for No, 1 for Yes): "))
# commercial_assets_value=float(input("Enter commercial assets value: "))
# luxury_assets_value=float(input("Enter luxury assets value: "))
# bank_asset_value=float(input("Enter bank assets value: "))
# loan_status=model.predict([[education,self_employed,commercial_assets_value,luxury_assets_value,bank_asset_value]])
# print("Loan status is:",loan_status)
# if loan_status==0:
#          print("========Loan Rejected========")
# else:
#                   print("========Loan Approved========")

# X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
# model=DecisionTreeClassifier()
# model.fit(X_train,Y_train)
# #prediction check
# predictions=model.predict(X_test)
# print("Prediction values are :",predictions)
# #accuracy check
# accuracy=accuracy_score(Y_test,predictions)
# print("Accuracy is=============>",accuracy)

# X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
# model=SVC(C=1.0)
# model.fit(X_train,Y_train)
# #prediction check
# predictions=model.predict(X_test)
# print("Prediction values are :",predictions)
# #accuracy check
# accuracy=accuracy_score(Y_test,predictions)
# print("Accuracy is=============>",accuracy)
