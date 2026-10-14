import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6],
    "Marks": [40, 48, 55, 63, 72, 80]
}

df = pd.DataFrame(data)
print(df)

correlation = df["Study_Hours"].corr(df["Marks"])
print("\nCorrelation:", correlation)

if correlation > 0:
    print("There is a positive association between Study Hours and Marks.")
elif correlation < 0:
    print("There is a negative association between Study Hours and Marks.")
else:
    print("There is no association between the variables.")

plt.scatter(df["Study_Hours"], df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()