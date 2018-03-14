from numpy import genfromtxt
import matplotlib.pyplot as plt
from util import plot_embedding

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

pca = PCA(n_components=10)
pca.fit(training)

training_pca = pca.transform(training)
testing_pca = pca.transform(testing)

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

NB_COMPONENTS=5

lda = LinearDiscriminantAnalysis(n_components=NB_COMPONENTS)
lda.fit(training, categories)

training_lda = lda.transform(training)
testing_lda = lda.transform(testing)

plot_embedding(training_lda, categories,
               "Linear Discriminant projection (training dataset)")


plot_embedding(testing_lda, testing_categories,
               "Linear Discriminant projection (test dataset)")

plt.show()

# Percentage of variance explained for each components
print('explained variance ratio (first %d components): %s' % (NB_COMPONENTS, str(lda.explained_variance_ratio_)))

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

for kernel in ['rbf','linear','poly']:

    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
    clf.fit(training_lda, categories)


    predictions = clf.predict(testing_lda)
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("SVM + LDA, kernel: %s: %.1f%% successful prediction out of %d test faces" % (kernel, correct_prediction * 100./len(predictions), len(predictions)))



