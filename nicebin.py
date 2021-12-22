# Author: Claudio Quevedo Gallardo, Civil Engineer in Bioinformatics <claudio.quev92@gmail.com>
#https://machinelearningmastery.com/clustering-algorithms-with-python/

from Bio import SeqIO
import re
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AffinityPropagation
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

def parse_fasta():

    return SeqIO.parse("LNS3_MaxBin2_conBinnacle.005.fasta", "fasta")
    #return SeqIO.parse("test.fasta", "fasta")


def bin_size():

    size = 0

    for record in parse_fasta():

        size = size + len(record.seq)

    return size


def calculate_orf_with_partial_codon(seq):

    warnings.filterwarnings("ignore")

    table = 1

    min_pro_len = 100

    orf_count = 0

    for strand, nuc in [(+1, seq), (-1, seq.reverse_complement())]:

        for frame in range(3):

            for pro in nuc[frame:].translate(table).split("*"):

                if len(pro) >= min_pro_len:

                    print ("%s...%s - length %i, strand %i, frame %i" % (pro[:30], pro[-3:], len(pro), strand, frame))

                    orf_count = orf_count + 1

    return orf_count


def calculate_orf_without_partial_codon(seq):

    warnings.filterwarnings("ignore")

    table = 1

    min_pro_len = 100

    for strand, nuc in [(+1, seq), (-1, seq.reverse_complement())]:

        for frame in range(3):

            for pro in nuc[frame:].translate(table).split("*"):

                if len(pro) >= min_pro_len:

                    if len(pro)%3==0:

                        print ("%s...%s - length %i, strand %i, frame %i" % (pro[:30], pro[-3:], len(pro), strand, frame))


def calculate_properties():
    data = []
    for record in parse_fasta():

        A = re.findall('(?i)A', str(record.seq))

        C = re.findall('(?i)C', str(record.seq))

        T = re.findall('(?i)T', str(record.seq))

        G = re.findall('(?i)G', str(record.seq))

        N = re.findall('(?i)N', str(record.seq))

        data.append([record.id,
        len(record.seq),
        mt.Tm_GC(record.seq,strict=False),
        (len(A)/len(record.seq))*100,
        (len(C)/len(record.seq))*100,
        (len(T)/len(record.seq))*100,
        (len(G)/len(record.seq))*100,
        (len(N)/len(record.seq))*100,
        ((len(G)+len(C))/len(record.seq))*100,
        calculate_orf_with_partial_codon(record.seq)])


        print("ID:", record.id)

        print("%Length:",len(record.seq))

        print("Melting Temp:", mt.Tm_GC(record.seq,strict=False))

        print("%A:",(len(A)/len(record.seq))*100)

        print("%C:",(len(C)/len(record.seq))*100)

        print("%T:",(len(T)/len(record.seq))*100)

        print("%G:",(len(G)/len(record.seq))*100)

        print("%N:",(len(N)/len(record.seq))*100)

        print("%GC:",((len(G)+len(C))/len(record.seq))*100)

        print("OFS's:",calculate_orf_with_partial_codon(record.seq))
    return data

def do_X():
    data = calculate_properties()
    df = DataFrame(data, columns=['id','contigLength','meltingTemp','A','C','T','G','N','GC','ORF'])
    print (df)
    X = np.array(df[['contigLength','meltingTemp','A','C','T','G','N','GC','ORF']])
    #X = np.array(df[['meltingTemp','C','G','N','GC','ORF']])
    standX = StandardScaler().fit_transform(X)
    return standX

def do_X2():
    data = calculate_properties()
    df = DataFrame(data, columns=['id','contigLength','meltingTemp','A','C','T','G','N','GC','ORF'])
    print (df)
    X = np.array(df[['contigLength','meltingTemp','A','C','T','G','N','GC','ORF']])
    #X = np.array(df[['meltingTemp','C','G','N','GC','ORF']])
    standX = StandardScaler().fit_transform(X)
    return standX, data


def Affinity_Propagation_algthm():
    X = do_X()
    afprop = AffinityPropagation(max_iter=350)
    afprop.fit(X)
    cluster_centers_indices = afprop.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    # Predict the cluster for all the samples
    P = afprop.predict(X)
    print(P)
    pca = PCA()
    x_pca = pca.fit_transform(X)
    print(x_pca)
    #explained_variance = pca.explained_variance_ratio_
    #print(explained_variance)
    #x_pca = pd.DataFrame(x_pca)
    #x_pca.columns = ['PC1','PC2','PC3','PC4','PC5','PC6']
    print(x_pca)
    plt.scatter(x_pca[:,1], x_pca[:,2], c="black", marker="o", picker=True)
    plt.title(f'Affinity Propagation')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

def Affinity_Propagation_algthm2():
    from numpy import unique
    from numpy import where
    X = do_X()
    # define the model
    model = AffinityPropagation(damping=0.9)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    plt.show()


def t_SNE_algthm():
    X, data = do_X2()
    Y = TSNE(n_components=2,init='pca', random_state=0, perplexity=40,learning_rate=600,n_iter=1200).fit_transform(X)
    df_tsne = pd.DataFrame(Y, columns=['PC1', 'PC2'])
    #pca = PCA()
    #x_pca = pca.fit_transform(Y)
    #print(x_pca)
    #print(X)
    print(data[0])
    print(df_tsne)


    plt.scatter(Y[:, 0], Y[:, 1], c="blue")
    plt.title(f't-distributed Stochastic Neighbor Embedding')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()





#
#
#
#
#
#
#
#
#
def Gaussian_Mixture_algthm():
    from numpy import unique
    from numpy import where
    X = do_X()
    model = GaussianMixture(n_components=2, covariance_type='full')
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    print(yhat)
    flag = 0
    for binary in yhat:
        if binary == 1:
            flag = flag +1
    print("flag: ",flag)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        #print(row_ix)
        #print("0 length: ",X[row_ix, 0])
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
        plt.title(f'Gaussian Mixture')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    plt.show()

def Gaussian_Mixture_algthm2():
    from sklearn import mixture
    import itertools
    import matplotlib as mpl
    from scipy import linalg
    # Number of samples per component
    n_samples = 300
    X = do_X()

    lowest_bic = np.infty
    bic = []
    n_components_range = range(2, 3)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        print(X[Y_ == i, 0])
        flag = 0
        for binary in X[Y_ == i, 0]:
            flag = flag +1
        print("flag: ",flag)
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(f'Selected GMM: {best_gmm.covariance_type} model, '
              f'{best_gmm.n_components} components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()

def Spectral_Clustering_algthm():
    from numpy import unique
    from numpy import where
    from sklearn.cluster import SpectralClustering
    X = do_X()
    model = SpectralClustering(n_clusters=2)
    yhat = model.fit_predict(X)
    flag = 0
    for binary in yhat:
        if binary == 1:
            flag = flag +1
    print("flag: ",flag)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()

def Agglomerative_Clustering_algthm():
    from numpy import unique
    from numpy import where
    from sklearn.datasets import make_classification
    from sklearn.cluster import AgglomerativeClustering
    X = do_X()
    # define the model
    model = AgglomerativeClustering(n_clusters=2)
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    flag = 0
    for binary in yhat:
        if binary == 1:
            flag = flag +1
    print("flag: ",flag)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    plt.show()

def BIRCH_algthm():
    from numpy import unique
    from numpy import where
    from sklearn.cluster import Birch
    X = do_X()
    # define the model
    model = Birch(threshold=0.01, n_clusters=2)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    flag = 0
    for binary in yhat:
        if binary == 1:
            flag = flag +1
    print("flag: ",flag)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()

def DBSCAN_Clustering_algthm():
    from numpy import unique
    from numpy import where
    from sklearn.cluster import DBSCAN
    X = do_X()
    # define the model
    model = DBSCAN(eps=0.6, min_samples=300)
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    flag = 0
    for binary in yhat:
        if binary == 1:
            flag = flag +1
    print("flag: ",flag)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()

def Kmeans_algthm():
    from numpy import unique
    from numpy import where
    from sklearn.cluster import KMeans
    X = do_X()
    # define the model
    model = KMeans(n_clusters=2)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    flag = 0
    for binary in yhat:
        if binary == 1:
            flag = flag +1
    print("flag: ",flag)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()

def MeanShift_Clustering_algthm():
    from numpy import unique
    from numpy import where
    from sklearn.cluster import MeanShift
    X = do_X()
    # define the model
    model = MeanShift()
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    flag = 0
    for binary in yhat:
        if binary == 1:
            flag = flag +1
    print("flag: ",flag)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()

def OPTICS_algthm():
    from numpy import unique
    from numpy import where
    from sklearn.cluster import OPTICS
    X = do_X()
    model = OPTICS(eps=0.8, min_samples=10)
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    print(yhat)
    flag = 0
    for binary in yhat:
        if binary == -1:
            flag = flag +1
    print("flag: ",flag)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()

def main():
    #Affinity_Propagation_algthm() NO
    #Affinity_Propagation_algthm2()  NO
    t_SNE_algthm()
    #Gaussian_Mixture_algthm()  SI
    #Gaussian_Mixture_algthm2()  SI
    #Spectral_Clustering_algthm()  NO
    #Agglomerative_Clustering_algthm() PUEDE QUE SI
    #BIRCH_algthm() PUEDE QUE SI
    #DBSCAN_Clustering_algthm()  NO
    #Kmeans_algthm() PUEDE QUE SI
    #MeanShift_Clustering_algthm()  NO
    #OPTICS_algthm() PUEDE QUE SI










if __name__ == "__main__":

    main()
