import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class ClassificationModelPicker:
    def __init__(self):
        self.decision_tree_model = None
        self.random_forest_model = None
        self.naive_bayes_model = None
        self.k_nearest_model = None
        self.svm_model = None
        self.logistic_reg_model = None

    def decision_tree(self, x_train, x_test, y_train, y_test, criterion="entropy"):
        self.decision_tree_model = DecisionTreeClassifier(criterion=criterion).fit(x_train, y_train)
        y_pred = self.decision_tree_model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        return self.decision_tree_model, cm, acc_score

    def random_forest(self, x_train, x_test, y_train, y_test, criterion="entropy", n_estimators=10):
        self.random_forest_model = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators).fit(x_train,
                                                                                                              y_train)
        y_pred = self.random_forest_model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        return self.random_forest_model, cm, acc_score

    def naive_bayes(self, x_train, x_test, y_train, y_test):
        self.naive_bayes_model = GaussianNB().fit(x_train, y_train)
        y_pred = self.naive_bayes_model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        return self.naive_bayes_model, cm, acc_score

    def k_nearest(self, x_train, x_test, y_train, y_test, n_neighbors=10, p=2):
        self.k_nearest_model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p).fit(x_train, y_train)
        y_pred = self.k_nearest_model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        return self.k_nearest_model, cm, acc_score

    def logistic_regression(self, x_train, x_test, y_train, y_test):
        self.logistic_reg_model = LogisticRegression().fit(x_train, y_train)
        y_pred = self.logistic_reg_model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        return self.logistic_reg_model, cm, acc_score

    def svm(self, x_train, x_test, y_train, y_test, kernel="rbf"):
        self.svm_model = SVC(kernel=kernel).fit(x_train, y_train)
        y_pred = self.svm_model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        return self.svm_model, cm, acc_score

    def basic_classifier_model_picker(self, x_train, x_test, y_train, y_test):
        dec_tree = self.decision_tree(x_train, x_test, y_train, y_test)[2]
        rand_frst = self.random_forest(x_train, x_test, y_train, y_test)[2]
        naive_byes = self.naive_bayes(x_train, x_test, y_train, y_test)[2]
        k_near = self.k_nearest(x_train, x_test, y_train, y_test)[2]
        logic_reg = self.logistic_regression(x_train, x_test, y_train, y_test)[2]
        svm = self.svm(x_train, x_test, y_train, y_test)[2]

        dic = {"Decision Tree Accuracy Score": dec_tree,
               "Random Forest Accuracy Score": rand_frst,
               "Naive_Byes Accuracy Score": naive_byes,
               "K-Nearest Neighbor Accuracy Score": k_near,
               "Logistic Regression Accuracy Score": logic_reg,
               "Support Vector Machine Accuracy Score": svm}

        print(dic)
