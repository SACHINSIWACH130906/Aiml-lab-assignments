import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

texts = [
    "Government launches new education policy",
    "Aliens found secretly in city",
    "New healthcare scheme announced",
    "Moon will disappear tomorrow"
]
labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(texts)

image_features = np.array([
    [120, 130, 125],
    [220, 50, 40],
    [110, 120, 115],
    [230, 60, 30]
])

features = hstack([text_features, image_features])

model = LogisticRegression()
model.fit(features, labels)

new_text = ["Government starts a new health program"]
new_text_feature = vectorizer.transform(new_text)
new_image_feature = np.array([[115, 125, 120]])

new_features = hstack([new_text_feature, new_image_feature])
prediction = model.predict(new_features)

if prediction[0] == 1:
    print("Prediction: Real News")
else:
    print("Prediction: Fake News")