from model import GMM,KMeans
from utils import calc_NMI,train_test_split,read_data
from sklearn.metrics import normalized_mutual_info_score

def main() -> None:
    # set hyperparameters
    kVal=3
    bins=10
    
    train_data,test_data = read_data()
    # read data
    X_train,X_test,Y_train,Y_test = train_test_split()
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
    #gmm_nmi = calc_NMI(gmm.predict(X_test),Y_test)
    #kmeans_nmi = calc_NMI(kmeans.predict(X_test),Y_test)
    gmm_nmi = normalized_mutual_info_score(gmm.predict(X_test), Y_test)
    kmeans_nmi = normalized_mutual_info_score(kmeans.predict(X_test), Y_test)

    print(f'Kmeans nmi={kmeans_nmi}, GMM nmi={gmm_nmi}')

if __name__ == '__main__':
    main()
