# Polynomial Linear Regression

# Assumptions of linear regression:
# Ensure all these assumptions are TRUE!
# Linearity
# Homoscedasticity
# Multivariate normality
# Independence of errors
# Lack of multicollinearity

# Dummy variables:
# Find all the different categorical data within the dataset
# Turn String values into binary

# Dummy variables traps

# Pvalues! (probability value)
# Statistical significance:
# 1) Flip a coin (two possible outcomes)

# 5 Methods of building models:
# 1) All-in (step wise regression) = Use of all variables

# 2) Backward elimination (step wise regression) =
# Step 1: select a significance level to stay in the model (e.g SL = 0.05)
# Step 2: Fit the full model with all possible predictors
# Step 3: Consider the predictor with the highest P-Value. If P > SL (significance level), go to step 4, otherwise go to FIN (model is ready)
# Step 4: Remove the predictor
# Step 5: Fit model without this variable
# Return back to Step 3

# 3) Forward selection (step wise regression)
# Step 1: Select a significance level to enter the model (e.g SL = 0.05)
# Step 2: Fit all simple regression models 'Y~Xn' select the one with the lowest P-value
# Step 3: Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have
# Step 4: Consider the predictor with the lowest P-value. If P<SL, go to Step 3, otherwise go to FIN

# 4) Bidirectional elimination (step wise regression)
# Step 1: Select a significance level to enter and to stay in the model
# Step 2: Perform the next step of Forward selection
# Step 3: Perform ALL steps of Backward elimination
# Step 4: No new variables can enter and no old variables can exit

# 5) Score comparison
# Step 1: Select a criterion of goodness of fit
# Step 2: Construct All possible Regression Models
# Step 3: Select the one with the best criterion

# BACKWARD ELIMINATION will be used for this project

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Importing the libraries

# 'np' is the numpy shortcut!
# 'plt' is the matplotlib shortcut!
# 'pd' is the pandas shortcut!

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Importing the dataset

# Data set is creating the data frame of the '50_Startups.csv' file
# Features (independent variables) = The columns the predict the dependent variable
# Dependent variable = The last column
# 'X' = The matrix of features (country, age, salary)
# 'Y' = Dependent variable vector (purchased (last column))
# '.iloc' = locate indexes[rows, columns]
# ':' = all rows (all range)
# ':-1' = Take all the columns except the last one
# '.values' = taking all the values

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # This will take the values from the second column (index 1) to the second from last one.
#print(X)
y = dataset.iloc[:, -1].values # NOTICE! .iloc[all the rows, only the last column]
#print(y)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Training the Linear Regression model on the whole dataset

from sklearn.linear_model import LinearRegression # sklearn library contains the linearRegression
lin_reg = LinearRegression() # An object of the linearRegression class above
lin_reg.fit(X, y) # Training method (training on the whole dataset.) 'X' and 'y' are the matrix variables

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Polynomial Regression

# Training the Polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures # Import the PolynomialFeatures.
poly_reg = PolynomialFeatures(degree = 4) # Training the model with higher degrees. (Power of 4!)
X_poly = poly_reg.fit_transform(X) # The 'fit_transform' feature transforms the matrix into x1 + x2 features.
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Polynomial Regression

# Visualising the LINEAR REGRESSION results

# Reminder 'plt' is shorthand for MatPlotLib.

plt.scatter(X, y, color = 'green') # Coordinates for the scatter plot.
plt.plot(X, lin_reg.predict(X), color = 'blue') # Plotting the regression line in blue
plt.title('Truth or Bluff (LINEAR REGRESSION)') # Title.
plt.xlabel('Position Level') # X axis label.(HORIZONTAL)
plt.ylabel('Salary') # Y axis label (VERTICAL)
plt.show() # Show the graph using the 'show' function.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Polynomial Regression

# Visualising the POLYNOMIAL REGRESSION results

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue') # Notice that this line is what seperates this code from the 'linear regression' model.
plt.title('Truth or Bluff (POLYNOMIAL REGRESSION)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Polynomial Regression

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.1) # '0.1' is the steps at which the lines connect. Lower the number the smoother the curve.
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
fontsize = 9
plt.title('Truth or Bluff (POLYNOMIAL REGRESSION HIGH RESOLUTION!)',fontsize=fontsize)
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Polynomial Regression

# Predicting a new result with LINEAR REGRESSION (The result to this is not accurate!)

# double bracket means an array [[]].
# The first dimension [] = rows.
# second dimension [[]] = columns.

lin_reg.predict([[6.5]])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Polynomial Regression

# Predicting a new result with POLYNOMIAL REGRESSION

# Notice the 2D array!

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))