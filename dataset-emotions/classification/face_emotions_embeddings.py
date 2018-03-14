#! /usr/bin/env python

import matplotlib.pyplot as plt
import csv

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from util import read_image, plot_gallery, plot_embedding

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

pca = PCA(n_components=10)
pca.fit(training_images)

plot_embedding(pca.transform(training_images), training_categories,
               "Principal Components projection")

#####################################################################################
#####################################################################################
## LDA


lda = LinearDiscriminantAnalysis(n_components=10)
lda.fit(training_images, training_categories)

plot_embedding(lda.transform(training_images), training_categories,
               "Linear Discriminant projection")

plot_embedding(lda.transform(testing_images), testing_categories,
               "Linear Discriminant projection (test dataset)")


#####################################################################################
#####################################################################################
## Classification with kNNs


from sklearn import neighbors
for k in range(1,10):

    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(lda.transform(training_images), training_categories)


    predictions = clf.predict(lda.transform(testing_images))
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("kNearestNeighbours, k=%d: %.1f%% successful prediction out of %d test faces" % (k, correct_prediction * 100./len(predictions), len(predictions)))

#####################################################################################
#####################################################################################
## Classification with SVM

from sklearn import svm


C=1.0
gamma=0.7
degree=4

for kernel in ['rbf','linear','poly']:

    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
    clf.fit(lda.transform(training_images), training_categories)


    predictions = clf.predict(lda.transform(testing_images))
    correct_prediction = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == int(testing_categories[i]):
            correct_prediction += 1
    print("SVM, kernel: %s: %.1f%% successful prediction out of %d test faces" % (kernel, correct_prediction * 100./len(predictions), len(predictions)))



plt.show()
