import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix



class ClassificationModelPicker:
    def __init__(self):
        self.decision_tree_model = None
        self.k_nearest_model = None
        self.svm_model = None
        self.Naive_Bayes = None
        self.logistic_reg_model = None
        self.random_forest_model = None

    def decision_tree(self,x_train,x_test,y_train,y_test,criterion="entropy"):
        self.decision_tree_model = DecisionTreeClassifier(criterion=criterion).fit(x_train,y_train)
        y_pred = self.decision_tree_model.predict(x_test)
        cm = confusion_matrix(y_test,y_pred)
        acc_score = accuracy_score(y_test,y_pred)
        return self.decision_tree_model,cm,acc_score

