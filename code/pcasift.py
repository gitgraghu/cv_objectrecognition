import cv2
import os
import numpy as np
from collections import defaultdict
from scipy.cluster.vq import kmeans,vq,kmeans2, whiten

def genSIFTdata(datatype, sift_detector, SIFTmean=None):
    Ylist = []
    kpcounts = []
    imglist = []
    SIFTdata = np.array([0]*128)

    # For every image generate SIFT features and append to SIFT matrix
    for i, category in enumerate(categories):
        trainimages = [os.path.join(imagesdir,category,datatype,image) for image in os.listdir(os.path.join(imagesdir, category, datatype)) if image.endswith(".jpg")]
        for imagefile in trainimages:
            imglist.append(imagefile)
            img = cv2.imread(imagefile)
            kp, features = sift_detector.detectAndCompute(img, None)
            SIFTdata = np.vstack((SIFTdata, features))
            Ylist.append(i)
            kpcounts.append(len(kp))
    SIFTdata = SIFTdata[1:, :]

    # Mean center SiFT data
    if(SIFTmean is None):
        SIFTmean = np.mean(SIFTdata, axis=0)
    SIFTdata -= SIFTmean

    Y = np.array(Ylist)

    return SIFTdata, Y, kpcounts, SIFTmean, imglist

def genPCAcomponents(SIFTdata, limit):
    # Calculate Covariance matrix
    C = np.cov(SIFTdata, rowvar=False)
    # Calculate eigenvals and eigenvectors
    evals, evecs = np.linalg.eig(C)
    # Sort eigenvals in decreasing order
    idx  = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    # Take top eigen vectorsk
    evecs = evecs[:, :limit]
    return evecs

def genTrainCodeWords(PCAdata, kpcounts, K):
    whitened = whiten(PCAdata)
    # Perform Kmean clustering on PCA data
    centroids, _ = kmeans2(whitened, k=K, minit='points', iter=40)
    # Find codewords for all image PCA data
    idx, _ = vq(whitened, centroids)
    current = 0
    Xlist = []
    # Create hist of codewords per image
    for kpcount in kpcounts:
        xidx = idx[current:current+kpcount]
        x, _ = np.histogram(xidx, bins=K, density=True)
        Xlist.append(x)
        current = current + kpcount

    X = np.array(Xlist)
    print X.shape
    return X, centroids

def genTestCodeWords(PCAdata, centroids, kpcounts, K):
    # Find codewords for all image PCA data
    idx, _ = vq(whiten(PCAdata), centroids)
    current = 0
    Xlist = []
    # Create hist of codewords per image
    for kpcount in kpcounts:
        xidx = idx[current:current+kpcount]
        x, _ = np.histogram(xidx, bins=K, density=True)
        Xlist.append(x)
        current = current + kpcount

    X = np.array(Xlist)
    return X

if __name__ == '__main__':
    # Get list of categories based on directories
    imagesdir = 'images'
    categories = [category for category in os.listdir(imagesdir) if os.path.isdir(os.path.join(imagesdir, category))]
    print 'Image Categories :' , categories

    # Get SIFT features matrix of all training images
    print 'Detecting SIFT features for all training images..'
    sift_detector  = cv2.xfeatures2d.SIFT_create()
    SIFTtrain, Ytrain, kpcounts1, SIFTmean, trainimglist = genSIFTdata('train', sift_detector)

    # Get PCA components of SIFT training features
    print 'Generating PCA Components..'
    evecs = genPCAcomponents(SIFTtrain, 25)

    # Transform SIFT features to PCA SIFT features
    print 'Transforming SIFT features to PCA SIFT..'
    PCASIFTtrain = np.dot(evecs.T, SIFTtrain.T).T
    print PCASIFTtrain.shape

    # Perform Kmeans clustering and find codeword hist for training images
    print 'Started Kmeans Clustering on training PCA-SIFT features..'
    Xtr, centroids = genTrainCodeWords(PCASIFTtrain, kpcounts1, 72);

    # Get SIFT features matrix of all test images
    print 'Detecting SIFT features for all test images..'
    SIFTtest, Ytest, kpcounts2, _, testimglist = genSIFTdata('test', sift_detector, SIFTmean)

    # Transform SIFT features to PCA SIFT features
    print 'Transforming SIFT features to PCA SIFT..'
    PCASIFTtest = np.dot(evecs.T, SIFTtest.T).T

    # Find codeword hist for test images
    print 'Generating codewords histogram for test images..'
    X = genTestCodeWords(PCASIFTtest, centroids, kpcounts2, 72);

    # Perform Nearest Neighbor classification
    print 'Starting Nearest Neighbor classification..'
    incorrect = defaultdict(int)
    predictions =  [[0]*len(categories) for i in range(len(categories))]
    for i in range(X.shape[0]):
        distances = np.linalg.norm(Xtr - X[i,:], axis=1)
        idx = np.argsort(distances)[:6]
        preds = Ytrain[idx]
        preddist = distances[idx]
        preddistweights = (1.0 - (preddist/np.max(preddist)))
        weightedcounts = np.zeros(len(categories))
        for j in range(preds.shape[0]):
            weightedcounts[preds[j]] += preddistweights[j]

        predcategoryid = np.argmax(weightedcounts)
        actualcategoryid = Ytest[i]

        predictions[actualcategoryid][predcategoryid]+=1

        if(actualcategoryid != predcategoryid):
            incorrect[categories[actualcategoryid]]+=1

        print testimglist[i], categories[predcategoryid]

    # Print out Num of incorrect preds
    print 'Incorrect Prediction counts:'
    for category in categories:
        if category in incorrect:
            print category, incorrect[category]
        else:
            print category, str(0)

    print 'Total Incorrect:', sum(incorrect.values())

    # Print out confusion matrix
    print 'Confusion Matrix:'
    confusionmat = np.array(predictions)
    print confusionmat
