#! /usr/bin/env python

import matplotlib.pyplot as plt
import csv

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from util import read_image, plot_gallery, plot_embedding, plot_confusion_matrix

PREFIX="../" # to the root of the dataset

training_images = []
training_categories = []
with open("../emotions_action_units_training.csv", 'r') as csvfile:
    details = csv.DictReader(csvfile, skipinitialspace=True)
    for row in details:
        training_images.append(read_image(PREFIX + row["face"]))
        training_categories.append(row["emotion"])


testing_images = []
testing_categories = []
with open("../emotions_action_units_test.csv", 'r') as csvfile:
    details = csv.DictReader(csvfile, skipinitialspace=True)
    for row in details:
        testing_images.append(read_image(PREFIX + row["face"]))
        testing_categories.append(row["emotion"])


#####################################################################################
#####################################################################################
## PCA

pca = PCA(n_components=20)
pca.fit(training_images)

plot_embedding(pca.transform(training_images), training_categories,
               "Principal Components projection (training)",
               xlabel="1st eigenvector", ylabel="2nd eigenvector")

plot_embedding(pca.transform(testing_images), testing_categories,
               "Principal Components projection (testing)",
               xlabel="1st eigenvector", ylabel="2nd eigenvector")


#####################################################################################
#####################################################################################
## LDA


#lda = LinearDiscriminantAnalysis(n_components=10,solver='lsqr',shrinkage='auto')
lda = LinearDiscriminantAnalysis(n_components=10)

lda.fit(training_images, training_categories)

plot_embedding(lda.transform(training_images), training_categories,
               "Linear Discriminant projection (training)",
               xlabel="1st dimension", ylabel="2nd dimension")

plot_embedding(lda.transform(testing_images), testing_categories,
               "Linear Discriminant projection (testing)",
               xlabel="1st dimension", ylabel="2nd dimension")


#####################################################################################
#####################################################################################
## PCA + LDA


lda = LinearDiscriminantAnalysis(n_components=10)

lda.fit(pca.transform(training_images), training_categories)

lda_pca_training_images = lda.transform(pca.transform(training_images))
lda_pca_testing_images = lda.transform(pca.transform(testing_images))

plot_embedding(lda_pca_training_images, training_categories,
               "PCA + Linear Discriminant projection (training)",
               xlabel="1st dimension", ylabel="2nd dimension")

plot_embedding(lda_pca_testing_images, testing_categories,
               "PCA + Linear Discriminant projection (testing)",
               xlabel="1st dimension", ylabel="2nd dimension")


#####################################################################################
#####################################################################################
## Classification with kNNs


from sklearn import neighbors
for k in range(1,10):

    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(pca.transform(training_images), training_categories)


    predictions = clf.predict(pca.transform(testing_images))
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("PCA kNearestNeighbours, k=%d: %.1f%% successful prediction out of %d test faces" % (k, correct_prediction * 100./len(predictions), len(predictions)))

for k in range(1,10):

    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(lda_pca_training_images, training_categories)


    predictions = clf.predict(lda_pca_testing_images)
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("PCA+LDA kNearestNeighbours, k=%d: %.1f%% successful prediction out of %d test faces" % (k, correct_prediction * 100./len(predictions), len(predictions)))

#####################################################################################
#####################################################################################
## Classification with SVM

from sklearn import svm


C=1.0
gamma=0.7
degree=4

for kernel in ['rbf']:

    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
    clf.fit(pca.transform(training_images), training_categories)


    predictions = clf.predict(pca.transform(testing_images))
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("PCA SVM, kernel: %s: %.1f%% successful prediction out of %d test faces" % (kernel, correct_prediction * 100./len(predictions), len(predictions)))



for kernel in ['rbf','linear','poly']:

    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
    clf.fit(lda_pca_training_images, training_categories)


    predictions = clf.predict(lda_pca_testing_images)
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("PCA+LDA SVM, kernel: %s: %.1f%% successful prediction out of %d test faces" % (kernel, correct_prediction * 100./len(predictions), len(predictions)))

#####################################################################################
#####################################################################################
## Confusion matrix

from sklearn.metrics import confusion_matrix
emotions=["fear","happiness","anger","surprise","sadness","disgust"]


clf = svm.SVC(kernel='rbf', gamma=gamma, C=C, degree=degree)
clf.fit(lda_pca_training_images, training_categories)
predictions = clf.predict(lda_pca_testing_images)

cnf_matrix = confusion_matrix(testing_categories, predictions)
plot_confusion_matrix(cnf_matrix, classes=emotions, normalize=True,
                      title='Confusion matrix for a RBF SVM, after performing a PCA + LDA on faces')



plt.show()
