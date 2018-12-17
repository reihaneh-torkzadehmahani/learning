from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle, check_random_state
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

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

Y_train = [ int(y) for y in Y_train]
Y_test = [ int(y) for y in Y_test]


#X = X.reshape((X.shape[0], -1))

print("Binarizing the labels ...")
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
Y_train = label_binarize(Y_train, classes= classes)
Y_test = label_binarize(Y_test, classes= classes)



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