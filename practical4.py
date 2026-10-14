from sklearn.naive_bayes import GaussianNB

X = [[1, 20], [2, 25], [3, 30], [8, 70], [9, 75], [10, 80]]
y = ["Low", "Low", "Low", "High", "High", "High"]

model = GaussianNB()
model.fit(X, y)

prediction = model.predict([[4, 35]])
print("Predicted Class:", prediction[0])