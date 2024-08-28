from classmodpick import ClassificationModelPicker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("example_data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=14)

#Scaler if required
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


classifier_models = ClassificationModelPicker()

decision_tree = classifier_models.decision_tree(X_train, X_test, y_train, y_test,criterion="entropy")
random_forest = classifier_models.random_forest(X_train, X_test, y_train, y_test,criterion="entropy", n_estimators=10)
naive_bayes = classifier_models.naive_bayes(X_train, X_test, y_train, y_test)
k_nearest = classifier_models.k_nearest(X_train, X_test, y_train, y_test, n_neighbors=10, p=2)
logistic_reg = classifier_models.logistic_regression(X_train, X_test, y_train, y_test)
svm = classifier_models.svm(X_train, X_test, y_train, y_test, kernel="rbf")


print(decision_tree)
print(random_forest)
print(naive_bayes)
print(k_nearest)
print(logistic_reg)
print(svm)

classifier_models.basic_classifier_model_picker(X_train, X_test, y_train, y_test)
