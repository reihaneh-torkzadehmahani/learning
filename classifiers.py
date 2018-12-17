from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle, check_random_state
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier


def compute_fpr_tpr_roc(Y_test, Y_score):
    n_classes = Y_score.shape[1]
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    for class_cntr in range(n_classes):
        false_positive_rate[class_cntr], true_positive_rate[class_cntr], _ = roc_curve(Y_test[:, class_cntr],
                                                                                       Y_score[:, class_cntr])
        roc_auc[class_cntr] = auc(false_positive_rate[class_cntr], true_positive_rate[class_cntr])

    # Compute micro-average ROC curve and ROC area
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    return false_positive_rate, true_positive_rate, roc_auc


def classify(X_train, Y_train, X_test, classiferName, random_state_value=0):
    if classiferName == "svm":
        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state_value))
    elif classiferName == "dt":
        classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=random_state_value))
    elif classiferName == "lr":
        classifier = OneVsRestClassifier(
            LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=random_state_value))
    elif classiferName == "rf":
        classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state_value))
    elif classiferName == "gnb":
        classifier = OneVsRestClassifier(GaussianNB(random_state=random_state_value))
    elif classiferName == "bnb":
        classifier = OneVsRestClassifier(BernoulliNB(alpha=.01, random_state=random_state_value))
    elif classiferName == "ab":
        classifier = OneVsRestClassifier(AdaBoostClassifier(random_state=random_state_value))
    elif classiferName == "mlp":
        classifier = OneVsRestClassifier(MLPClassifier(random_state=random_state_value, alpha=1))
    else:
        print("Classifier not in the list!")
        exit()

    Y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)
    return Y_score


print("Loading MNIST Dataset ...")

X_train, Y_train = loadlocal_mnist(
    images_path='../mnist_dataset/train-images.idx3-ubyte',
    labels_path='../mnist_dataset/train-labels.idx1-ubyte')

X_test, Y_test = loadlocal_mnist(
    images_path='../mnist_dataset/t10k-images.idx3-ubyte',
    labels_path='../mnist_dataset/t10k-labels.idx1-ubyte')

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

X_train = X_train[0:1000]
Y_train = Y_train[0:1000]

X_test = X_test[0:300]
Y_test = Y_test[0:300]

Y_train = [int(y) for y in Y_train]
Y_test = [int(y) for y in Y_test]

# X = X.reshape((X.shape[0], -1))

print("Binarizing the labels ...")
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y_train = label_binarize(Y_train, classes=classes)
Y_test = label_binarize(Y_test, classes=classes)

#
print("Classifying ...")

Y_score = classify(X_train, Y_train, X_test, "mlp", random_state_value=30)

print(Y_score)

print("Computing ROC ...")

false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)

print(roc_auc["micro"])

plt.figure()
lw = 2
plt.plot(false_positive_rate["micro"], true_positive_rate["micro"], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()