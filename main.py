import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

df.info()
df.head()
df.describe()

#women rates
women = df.loc[df.Sex == 'female']['Survived']
women_rate = sum(women)/len(women)
print(f"{women_rate}% of women survived.")

#men rates
men = df.loc[df.Sex == 'male']['Survived']
men_rate = sum(men)/len(men)
print(f"{men_rate}% of men survived.")

#forest random model
y = df['Survived']
features = ['Pclass', 'Sex', 'SibSp', 'Parch']

X = pd.get_dummies(df[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, random_state=1, max_depth=5)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerID': test.PassengerId, 'Survived': predictions})
output = output.to_csv('submissions.csv', index=False)

print("Your submissions are succesfully saved.")


