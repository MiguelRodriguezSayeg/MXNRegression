#día -10941: 19/04/1954


#->día 1: 21/04/1995

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from date_formatter import get_days_from_beginning


data = pd.read_csv('historico.csv')
X = pd.DataFrame(data['Days'])
Y = pd.DataFrame(data['Relation'])

poly_feat = PolynomialFeatures(degree=1)
x_poly = poly_feat.fit_transform(X)

model = LinearRegression()


x_train, x_test, y_train, y_test = train_test_split(x_poly, Y, test_size=0.19, shuffle=False)

model.fit(x_train, y_train)



predicciones = [pred[0] for pred in model.predict(x_test)]


print("Squared mean error: {}".format(mean_squared_error(y_test, predicciones)))
print("R2 score: {}".format(r2_score(Y, model.predict(x_poly))))

'''
slope = model.coef_[0][1]
squared = model.coef_[0][2]
c = model.intercept_[0]
print("y ={}x^2 + {}x +{}".format(squared, slope, c))
'''
plt.scatter(X, Y, color="black")
predicciones2 = [pred[0] for pred in model.predict(x_poly)]


today = get_days_from_beginning()
print(today)
today = poly_feat.fit_transform([[today]])
print("Last prediction, today: {}".format(model.predict(today)[0][0]))
plt.plot(X, predicciones2, color="blue", linewidth=3)
plt.xticks()
plt.yticks()
plt.show()



