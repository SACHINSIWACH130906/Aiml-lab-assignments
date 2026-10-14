from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

X = [[1], [2], [3], [4], [5], [6]]
y = [42, 48, 55, 63, 72, 80]

model = Ridge(alpha=1.0)
model.fit(X, y)

prediction = model.predict([[7]])
print("Predicted Marks:", prediction[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='orange')
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Ridge Regression with Regularization")
plt.show()