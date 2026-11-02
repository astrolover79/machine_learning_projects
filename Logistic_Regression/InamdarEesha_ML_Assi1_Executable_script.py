# Reading the csv data file using pandas
import pandas as pd
import numpy as np 
train_data = pd.read_csv("week2.csv",skiprows=1,header=None)
train_data.columns = ["X1","X2","y"] # Give titles to the columns
print(train_data.head())
# print(df.shape)
X1=train_data.iloc [: ,0]
X2=train_data.iloc [: ,1]
y=train_data.iloc [: ,2]

%pip install seaborn

#Part a) i) Data visualized through a scatterplot using seaborn and matplotlib libraries
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=train_data,
    x=X1[y==1],
    y=X2[y==1],
    marker="+",
    label="+1"
)

sns.scatterplot(
    data=train_data,
    x=X1[y==-1],
    y=X2[y==-1],
    marker="o",
    label="-1"
)

plt.xlabel('Feature x_1')
plt.ylabel('Feature x_2')
plt.legend(
    loc='upper left',
    bbox_to_anchor = (1.03,1),
    title = "Actual 'y' values"
)
plt.show()

# Check for any missing/null values in the data
train_data.isnull().sum()

# # Logistic Regression Classifier model

# Defining the feature and target variables
X_train = train_data.drop(['y'],axis=1) # feature variables dataframe
y_train = train_data['y'] # target variable dataframe

X_train.head() # Added the print statment for verification

# Train the logistic regression classfier model using the training data
from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression(max_iter=500)
log_reg_model.fit(X_train,y_train)

# Used the above trained model for predicting the target values 
y_pred = log_reg_model.predict(X_train)
#print("Predicted values: ",y_pred)

# Output the predicted values into a new csv file
output = pd.DataFrame({
    'Feature x1': X_train['X1'],
    'Feature x2': X_train['X2'],
    'Actual target value': y_train,
    'Predicted target value': y_pred
})

output.to_csv("LogisticRegression.csv",index=False)
print("Output file for Logistic Regression model generated successfully")

# # Added the predicted values to the 2D plot generated in part(i)

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    x=X1[y_train==1],
    y=X2[y_train==1],
    marker="+",
    color="blue",
    s=100,
    label="actual: +1"
)

sns.scatterplot(
    x=X1[y_train==-1],
    y=X2[y_train==-1],
    marker="x",
    color="orange",
    s=60,
    label="actual: -1"
)

# Predicted values
sns.scatterplot(
    x=X1[y_pred==1],
    y=X2[y_pred==1],
    marker="^",
    color="blue",
    s=40,
    label="predicted: +1"
)

sns.scatterplot(
    x=X1[y_pred==-1],
    y=X2[y_pred==-1],
    marker="s",
    color="orange",
    s=25,
    label="predicted: -1"
)

# Decision boundary line
# Step (1): Defining and accessing parameter values of the trained model
θo = log_reg_model.intercept_
θ1 = log_reg_model.coef_[0][0]
θ2 = log_reg_model.coef_[0][1]

# Step (2): x1 values for the decision boundary line
x1_line = np.linspace(X1.min(),X1.max(),200)

# Step (3): Decision boundary line to be plotted where y=0
x2_line = -(θo + θ1*x1_line)/θ2

# Step (4): Plotting the line
plt.plot(
    x1_line,
    x2_line,
    linestyle = '--',
    color = 'black',
    linewidth = 3,
    label = 'Decision boundary'
)

plt.xlabel('Feature x_1')
plt.ylabel('Feature x_2')
plt.legend(
    loc='upper left',
    bbox_to_anchor = (1.03,1),
    title = "Target 'y' values"
)

plt.show()

# Parameter values of the above trained model
print("Intercept: ", θo)
print("Coefficients: ",θ1, θ2)

# # Accuracy score of the Logistic Regression model

# Accuracy score between the predictions and the training data
from sklearn.metrics import accuracy_score
score = accuracy_score(y,y_pred)
print(score)

# # Part(2) Linear SVC

# Train a linear SVM classifier model using the same training data
from sklearn.svm import LinearSVC
C_val = [0.001,1.0,100] # Different values of penalty parameter C

for C in C_val: 
    svm_model = LinearSVC(C=C)
    svm_model.fit(X_train,y_train)

    # Predicting target values
    y_pred_svm = svm_model.predict(X_train)
    print("C:",C)
    print("Intercept: ",svm_model.intercept_)
    print("Coefficients: ",svm_model.coef_)
    #print("Predictions: ",y_pred_svm)

    plt.figure() # To plot 3 different plots

    sns.scatterplot(
    x=X1[y_train==1],
    y=X2[y_train==1],
    marker="+",
    color="blue",
    s=100,
    label="actual: +1"
    )

    sns.scatterplot(
    x=X1[y_train==-1],
    y=X2[y_train==-1],
    marker="x",
    color="orange",
    s=60,
    label="actual: -1"
    )

    # Predicted values using LinearSVC
    sns.scatterplot(
    x=X1[y_pred_svm==1],
    y=X2[y_pred_svm==1],
    marker="^",
    color="blue",
    s=40,
    label="predicted: +1"
    )

    sns.scatterplot(
    x=X1[y_pred_svm==-1],
    y=X2[y_pred_svm==-1],
    marker="s",
    color="orange",
    s=25,
    label="predicted: -1"
    )

    score_svm = accuracy_score(y,y_pred_svm)
    print("Accuracy score: ", score_svm)

    # Decision boundary line
    # Step (1): Defining and accessing parameter values of the trained model
    θo = svm_model.intercept_[0]
    θ1 = svm_model.coef_[0][0]
    θ2 = svm_model.coef_[0][1]

    # Step (2): x1 values for the decision boundary line
    x1_line = np.linspace(X1.min(),X1.max(),50)

    # Step (3): Decision boundary line to be plotted where y=0
    x2_line = -(θo + θ1*x1_line)/θ2

    # Step (4): Plotting the line
    plt.plot(
    x1_line,
    x2_line,
    linestyle = '--',
    color = 'black',
    linewidth = 3,
    label = 'Decision boundary'
    )

    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title(f"Linear SVM, C={C}")
    #plt.tight_layout()
    plt.show()
    

# # Train a quadratic logistic regression classifier model on top of the initial part(a) logistic regression classifier model

# Extended Logistic Regression model (added 2 additional squared input features)
X_train_poly=np.column_stack((X_train,X_train**2))
#print(X_train_poly)
extended_log_reg_model = LogisticRegression(max_iter=100).fit(X_train_poly,y_train)
y_pred_extended = extended_log_reg_model.predict(X_train_poly)

# print(y_predicted_extended)
print(extended_log_reg_model.intercept_)
print(extended_log_reg_model.coef_)

# Plotting the predicted and actual target values
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    x=X1[y_train==1],
    y=X2[y_train==1],
    marker="+",
    color="#A14646",
    s=100,
    label="actual: +1"
)

sns.scatterplot(
    x=X1[y_train==-1],
    y=X2[y_train==-1],
    marker="x",
    color="#4F5B2A",
    s=60,
    label="actual: -1"
)

# Predicted values
sns.scatterplot(
    x=X1[y_pred_extended==1],
    y=X2[y_pred_extended==1],
    marker="^",
    color="#A14646",
    s=40,
    label="predicted: +1"
)

sns.scatterplot(
    x=X1[y_pred_extended==-1],
    y=X2[y_pred_extended==-1],
    marker="s",
    color="#4F5B2A",
    s=25,
    label="predicted: -1"
)

# Decision boundary curve
# Step (1): Defining and accessing parameter values of the trained model
θo = extended_log_reg_model.intercept_
θ1 = extended_log_reg_model.coef_[0][0]
θ2 = extended_log_reg_model.coef_[0][1]
θ3 = extended_log_reg_model.coef_[0][2]
θ4 = extended_log_reg_model.coef_[0][3]

# Decision boundary curve
# Step (2): Forming a grid (two 2D arrays) of all the X1 and X2 data points
x1_line_grid = np.linspace(X_train.iloc[:, 0].min(),X_train.iloc[:, 0].max(),200)
x2_line_grid = np.linspace(X_train.iloc[:, 1].min(),X_train.iloc[:, 1].max(),200)

x1_grid,x2_grid = np.meshgrid(x1_line_grid,x2_line_grid)

# Step (3): Flatten the above created 2D grids into single long 1D array for each of the inputs for calculating the linear score of the model
poly_grid = np.column_stack((
    x1_grid.ravel(), # ravel converts the 2D grid into a single long 1D array
    x2_grid.ravel(),
    x1_grid.ravel()**2,
    x2_grid.ravel()**2
))

# Step (4): Calculate the extended model's linear score
linear_score = extended_log_reg_model.decision_function(poly_grid)

# Step (5): Convert the linear score values back into a 2D grid for plotting the decision boundary curve
linear_score = linear_score.reshape(x1_grid.shape)

# Step (6): Plot the decision boundary curve where the model score is 0
plt.contour(
    x1_grid,
    x2_grid,
    linear_score,
    levels = [0]
)

plt.xlabel('Feature x_1')
plt.ylabel('Feature x_2')
plt.legend(
    loc='upper left',
    bbox_to_anchor = (1.03,1),
    title = "Target 'y' values"
)

plt.show()

# # Accuracy score of the extended Logistic Regression classifier model

score_extended = accuracy_score(y,y_pred_extended)
print(score_extended)

# Baseline accuracy score obtained using ZeroR classifier method
from collections import Counter

count = Counter(y)
majority_class = count.most_common()[0][0] #Tells us which target value class appears the most no. of times
baseline_score = np.mean(y==majority_class)

print("Majority class: ",majority_class)
print("Baseline predictor score: ",baseline_score)

