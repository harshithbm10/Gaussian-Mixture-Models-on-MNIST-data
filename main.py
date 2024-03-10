from model import GMM,KMeans
from utils import calc_NMI,train_test_split,read_data
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.mixture import GaussianMixture

def main() -> None:
    # set hyperparameters
    kVal=5  #because it is MNIST 10 clusters for 10 digits  (0 to 9)
    bins=10
    
    train_data,test_data = read_data()
    # read data
    X_train, Y_train, X_test, Y_test = train_test_split()
    # create a model
    #if X_test.shape[0] != Y_test.shape[0]:
       # raise ValueError("Number of samples in X_test and Y_test must be the same.")    
    gmm = GMM(k=kVal)
    kmeans = KMeans(k=kVal)

    # fit the model
    gmm.fit(X_train)
    kmeans.fit(X_train)

    # evaluate the models
    #Y_test = Y_test.reshape(-1)
    #if X_test.shape[0] != Y_test.shape[0]:
        #raise ValueError("Number of samples in X_test and Y_test must be the same.")
    #gmm_nmi = calc_NMI(gmm.predict(X_test),Y_test,bins)
    kmeans_nmi = calc_NMI(kmeans.predict(X_test),Y_test,bins)
    gmm_nmi = normalized_mutual_info_score(gmm.predict(X_test), Y_test)
    #kmeans_nmi = normalized_mutual_info_score(kmeans.predict(X_test), Y_test)
    print(f"Kmeans nmi={kmeans_nmi}, GMM nmi={gmm_nmi}")
    gmm_log_likelihood = log_likelihood(gmm, X_test)
    print(f"GMM Log-Likelihood: {gmm_log_likelihood:.4e}")

    sklearn_kmeans = SklearnKMeans(n_clusters=kVal, random_state=0)
    sklearn_gmm = GaussianMixture(n_components=kVal, random_state=0)  
    sklearn_kmeans.fit(X_train)
    sklearn_gmm.fit(X_train)
    sklearn_kmeans_nmi = normalized_mutual_info_score(sklearn_kmeans.predict(X_test), Y_test)
    sklearn_gmm_nmi = normalized_mutual_info_score(sklearn_gmm.predict(X_test), Y_test)
    print(f"Kmeans NMI (Scikit): {sklearn_kmeans_nmi},GMM NMI (Scikit): {sklearn_gmm_nmi}")
    sklearn_gmm_log_likelihood = sklearn_gmm.score(X_test)
    print(f"GMM Log-Likelihood (Scikit): {sklearn_gmm_log_likelihood:.4e}")
    #print("GMM Log-Likelihood (Scikit):",sklearn_gmm_log_likelihood)


def log_likelihood(model, X):
    responsibilities=model.predict_proba(X)
    log_likelihood=np.sum(np.log(np.sum(responsibilities,axis=1)))
    return log_likelihood

if __name__ == '__main__':
    main()
