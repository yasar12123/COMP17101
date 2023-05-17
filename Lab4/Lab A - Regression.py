import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
print(data.DESCR)

X = data.data
Y = data.target
predictors = data.feature_names

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Setup models
models = [linear_model.LinearRegression(),
          linear_model.Ridge(alpha=1),
          linear_model.Ridge(alpha=1000),
          linear_model.Lasso(alpha=1),
          linear_model.Lasso(alpha=0.05),
          linear_model.ElasticNet(alpha=0.1), ]
names = ['OLS', 'Ridge (alpha=1.0)', 'Ridge (alpha=1000)', 'Lasso (alpha=1)', 'Lasso (alpha=0.1)',
         'ElasticNet (alpha=0.1, rho=0.5)']

# Training and prediction
Y_pred = {}
for i, model in enumerate(models):
    model.fit(X_train, Y_train)
    Y_pred[names[i]] = model.predict(X_test)

# Visualize Results
fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.5 * len(models)))
y_max = 6
for i, (model, name) in enumerate(zip(models, names)):
    r2 = r2_score(Y_test, Y_pred[name])
    mse = mean_squared_error(Y_test, Y_pred[name])

    axes[i, 0].set_title('%s - R2=%.2f, MSE=%.2f' % (name, r2, mse))
    axes[i, 0].scatter(Y_test, Y_pred[names[i]], color='black', s=10, alpha=0.02)
    axes[i, 0].plot([0, y_max], [0, y_max], color='blue', linewidth=2)
    axes[i, 0].set_xlabel('actual (y)')
    axes[i, 0].set_ylabel('predited (f(x))')

    axes[i, 1].set_title('Coefficients %s ' % name)
    axes[i, 1].bar(predictors, model.coef_)
    axes[i, 1].tick_params(axis='x', labelrotation=90)

plt.subplots_adjust(left=None, bottom=0, right=None,
                    top=None, wspace=None, hspace=0.6)
plt.show()

print('Dropping %i instances before running regression' % sum(data.target > 5))
X = data.data[data.target <= 5]
Y = data.target[data.target <= 5]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Setup models
models = [linear_model.LinearRegression(),
          linear_model.Ridge(alpha=1),
          linear_model.Ridge(alpha=1000),
          linear_model.Lasso(alpha=1),
          linear_model.Lasso(alpha=0.1),
          linear_model.ElasticNet(alpha=0.1), ]
names = ['OLS', 'Ridge (alpha=1.0)', 'Ridge (alpha=1000)', 'Lasso (alpha=1)', 'Lasso (alpha=0.1)',
         'ElasticNet (alpha=0.1, rho=0.5)']

# Training and prediction
Y_pred = {}
for i, model in enumerate(models):
    model.fit(X, Y)
    Y_pred[names[i]] = model.predict(X_test)

# Visualize Results
fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.5 * len(models)))
y_max = 6
for i, (model, name) in enumerate(zip(models, names)):
    r2 = r2_score(Y_test, Y_pred[name])
    mse = mean_squared_error(Y_test, Y_pred[name])

    axes[i, 0].set_title('%s - R2=%.2f, MSE=%.2f' % (name, r2, mse))
    axes[i, 0].scatter(Y_test, Y_pred[names[i]], color='black', s=10, alpha=0.02)
    axes[i, 0].plot([0, y_max], [0, y_max], color='blue', linewidth=2)
    axes[i, 0].set_xlabel('actual (y)')
    axes[i, 0].set_ylabel('predited (f(x))')

    axes[i, 1].set_title('Coefficients %s ' % name)
    axes[i, 1].bar(predictors, model.coef_)
    axes[i, 1].tick_params(axis='x', labelrotation=90)

plt.subplots_adjust(left=None, bottom=0, right=None,
                    top=None, wspace=None, hspace=0.6)
#plt.show()

print(X_train[0])
#print(type(Y_train))