from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle, check_random_state
import matplotlib.pyplot as plt

print("Loading MNIST Dataset ...")
X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)

print("Shuffling ...")
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
Y = Y[permutation]

X = X[0:20000]
Y = Y[0:20000]

Y = [ int(y) for y in Y]


#X = X.reshape((X.shape[0], -1))

print("Separating Train and Test Data ...")
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
Y = label_binarize(Y, classes= classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3,
                                                    random_state=0)
#
print("Classifying ...")
svm_classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

Y_score = svm_classifier.fit(X_train, Y_train).decision_function(X_test)

Y_test = label_binarize(Y_test, classes= classes)

print("Computing ROC ...")
# Compute ROC curve and ROC area for each class
false_positive_rate = dict()
true_positive_rate = dict()
roc_auc = dict()
n_classes = 10


for class_cntr in range(n_classes):
    false_positive_rate[class_cntr], true_positive_rate[class_cntr], _ = roc_curve(Y_test[:, class_cntr], Y_score[:, class_cntr])
    roc_auc[class_cntr] = auc(false_positive_rate[class_cntr], true_positive_rate[class_cntr])

# Compute micro-average ROC curve and ROC area
false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

print(roc_auc["micro"])

plt.figure()
lw = 2
plt.plot(false_positive_rate[2], true_positive_rate[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()