from sklearn.svm import SVR
import matplotlib.pyplot as plt

X = [[1], [2], [3], [4], [5], [6]]
y = [42, 48, 55, 63, 72, 80]

model = SVR(kernel='linear')
model.fit(X, y)

prediction = model.predict([[7]])
print("Predicted Marks:", prediction[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='green')
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Support Vector Regression")
plt.show()