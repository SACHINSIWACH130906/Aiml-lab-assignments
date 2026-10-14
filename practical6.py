from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = [[1], [2], [3], [4], [5], [6]]
y = [40, 48, 55, 63, 70, 78]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[7]])
print("Predicted Marks for 7 hours:", prediction[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.show()