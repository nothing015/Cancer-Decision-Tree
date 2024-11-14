from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

X, y = datasets.load_breast_cancer(return_X_y=True)  

print("There are", X.shape[0], "instances described by", X.shape[1], "features.") 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state = 42) 


clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=6)  
clf = clf.fit(X_train, y_train)  


predC = clf.predict(X_test) 
print('The accuracy of the classifier is', accuracy_score(y_test, predC)) 
_ = tree.plot_tree(clf, filled=True, fontsize= 12)  


trainAccuracy = []   
testAccuracy = []
depthOptions = range(1, 16) #(1 point) 
for depth in depthOptions: #(1 point) 

    cltree = tree.DecisionTreeClassifier(criterion="entropy",max_depth=depth, min_samples_split=6)

    cltree = cltree.fit(X_train, y_train) 

    y_predTrain = cltree.predict(X_train) 

    y_predTest = cltree.predict(X_test) 

    trainAccuracy.append(accuracy_score(y_train, y_predTrain)) 

    testAccuracy.append(accuracy_score(y_test, y_predTest)) 


plt.plot(depthOptions, trainAccuracy, marker='x', color='r', label='Training Accuracy')
plt.plot(depthOptions, testAccuracy, marker='x', color='b', label='Test Accuracy') 
plt.legend(['Training Accuracy','Test Accuracy']) 
plt.xlabel('Tree Depth')  
plt.ylabel('Classifier Accuracy') 

""" 
According to the test error, the best model to select is when the maximum depth is equal to 3, approximately. 
But, we should not use select the hyperparameters of our model using the test data, because it can lead to overfitting as the model may not generalize well to unseen data.
"""

parameters = {'max_depth':range(1, 20), 'min_samples_split':range(1,10)} 
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), parameters, cv = 5) 
clf.fit(X_train, y_train) 
tree_model = clf.best_estimator_ 
print("The maximum depth of the tree sis", tree_model.get_depth(), 
      'and the minimum number of samples required to split a node is', tree_model.get_params()['min_samples_split']) #(6 points)
_ = tree.plot_tree(tree_model,filled=True, fontsize=12) 
""" 
This method for tuning the hyperparameters of our model is acceptable, because it systematically explores multiple combinations of hyperparameters and identifies the best-performing model based on cross-validated performance, reducing the risk of overfitting and ensuring generalizability. . 
"""

"""
Tenfold stratified cross-validation is a technique used to assess the performance of a machine learning model.
In this method, the dataset is divided into ten equal-sized folds, ensuring that each fold contains a representative distribution of the target classes.
This is particularly important for imbalanced datasets, as it prevents skewing the evaluation metrics.
The model is trained on nine of the folds and validated on the remaining fold, and this process is repeated ten times,
each time using a different fold as the validation set.
The results from all iterations are then averaged to provide a more reliable estimate of the model's performance.

"""
