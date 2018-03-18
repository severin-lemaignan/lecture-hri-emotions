import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from util import plot_embedding, plot_confusion_matrix

emotions=["fear","happiness","anger","surprise","sadness","disgust"]

csv = genfromtxt('../emotions_action_units_training.csv', delimiter=',', skip_header=1)
training = csv[:,1:-1]
categories = csv[:,0]

csv = genfromtxt('../emotions_action_units_test.csv', delimiter=',', skip_header=1)

testing = csv[:,1:-1]
testing_categories = csv[:,0]


#####################################################################################
#####################################################################################
## kNNs
from sklearn import neighbors

print("\n### kNN")

for k in range(1,10):

    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(training, categories)


    predictions = clf.predict(testing)
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("kNearestNeighbours, k=%d: %.1f%% successful prediction out of %d test faces" % (k, correct_prediction * 100./len(predictions), len(predictions)))


#####################################################################################
#####################################################################################
## SVM

from sklearn import svm

print("\n### SVM")

C=1.0
gamma=0.7
degree=4

for kernel in ['rbf','linear','poly']:

    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
    clf.fit(training, categories)


    predictions = clf.predict(testing)
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("SVM, kernel: %s: %.1f%% successful prediction out of %d test faces" % (kernel, correct_prediction * 100./len(predictions), len(predictions)))
#####################################################################################
#####################################################################################
## kNNs + PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(training)

training_pca = pca.transform(training)
testing_pca = pca.transform(testing)

plot_embedding(training_pca, categories,
               "PCA projection (training)",
               xlabel="1st eigenvector", ylabel="2nd eigenvector")


plot_embedding(testing_pca, testing_categories,
               "PCA projection (testing)",
               xlabel="1st eigenvector", ylabel="2nd eigenvector")


source_eigenvector1 = pca.inverse_transform([1,0,0,0,0])
source_eigenvector2 = pca.inverse_transform([0,1,0,0,0])
emotions_au_centroids = [training[categories == i].mean(axis=0) for i in range(len(emotions))]
au_keys = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU12","AU14","AU15","AU17","AU20","AU23","AU25","AU26","AU45"]

print("\n### PCA <-> Action units")
print("Source of first eigenvectors + mean AUs for the different emotions:\n")
print("  AU   ev 1   ev 2   " + " ".join([e.rjust(9) for e in emotions]))
for i, au in enumerate(au_keys):
    print("%s   %.02f   %.02f  "%(au, source_eigenvector1[i], source_eigenvector2[i]) +\
           ("      %.02f" * len(emotions)) % tuple(emotions_au_centroids[j][i] for j in range(len(emotions))))

#####################################################################################
#####################################################################################

print("\n### PCA + kNN")
for k in range(1,10):

    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(training_pca, categories)


    predictions = clf.predict(testing_pca)
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("kNearestNeighbours + PCA, k=%d: %.1f%% successful prediction out of %d test faces" % (k, correct_prediction * 100./len(predictions), len(predictions)))

#####################################################################################
#####################################################################################
## SVM + PCA

print("\n### PCA + SVM")

for kernel in ['rbf','linear','poly']:

    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
    clf.fit(training_pca, categories)


    predictions = clf.predict(testing_pca)
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("SVM + PCA, kernel: %s: %.1f%% successful prediction out of %d test faces" % (kernel, correct_prediction * 100./len(predictions), len(predictions)))


#####################################################################################
#####################################################################################
## kNNs + LDA

print("\n### LDA")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

NB_COMPONENTS=2

lda = LinearDiscriminantAnalysis(n_components=NB_COMPONENTS)
#lda = LinearDiscriminantAnalysis(n_components=NB_COMPONENTS,solver="eigen",shrinkage="auto")
lda.fit(training, categories)

training_lda = lda.transform(training)
testing_lda = lda.transform(testing)

plot_embedding(training_lda, categories,
               "Linear Discriminant projection (training)",
               xlabel="1st dimension", ylabel="2nd dimension")


plot_embedding(testing_lda, testing_categories,
               "Linear Discriminant projection (testing)",
               xlabel="1st dimension", ylabel="2nd dimension")


# Percentage of variance explained for each components
print('explained variance ratio (first %d components): %s' % (NB_COMPONENTS, str(lda.explained_variance_ratio_)))

print("\n### LDA + kNN")

for k in range(1,10):

    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(training_lda, categories)


    predictions = clf.predict(testing_lda)

    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("kNearestNeighbours + LDA, k=%d: %.1f%% successful prediction out of %d test faces" % (k, correct_prediction * 100./len(predictions), len(predictions)))

#####################################################################################
#####################################################################################
## SVM + LDA

print("\n### LDA + SVM")

for kernel in ['rbf','linear','poly']:

    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
    clf.fit(training_lda, categories)


    predictions = clf.predict(testing_lda)
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("SVM + LDA, kernel: %s: %.1f%% successful prediction out of %d test faces" % (kernel, correct_prediction * 100./len(predictions), len(predictions)))

#####################################################################################
#####################################################################################
## Confusion matrix

from sklearn.metrics import confusion_matrix


clf = svm.SVC(kernel='rbf', gamma=gamma, C=C, degree=degree)
clf.fit(training_lda, categories)
predictions = clf.predict(testing_lda)

cnf_matrix = confusion_matrix(testing_categories, predictions)
plot_confusion_matrix(cnf_matrix, classes=emotions, normalize=True,
                      title='Confusion matrix for a RBF SVM, after performing a LDA on AUs')


plt.show()
