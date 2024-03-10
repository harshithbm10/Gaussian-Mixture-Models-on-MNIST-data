import numpy as np

def calc_NMI(X,Y,bins):
   # Complete code for the measure
   #ConfusionMatrix,probabilities(x,y and joint*),
   #pmi(xy)=pj(xiyj)log(pj(xiyj)/p(x)p(y))
   #h(x)=-sum(p(xi).log(p(xi))
   print("shape of y",Y.shape)
   print("shape of x",X.shape)
   sampleNo=len(X)
   #print(sampleNo)
   ConfusionMatrix=np.zeros((bins,bins))
   #print(ConfusionMatrix)
   for i in range(sampleNo):
      ConfusionMatrix[X[i],Y[i]]+=1
   jointP=ConfusionMatrix/sampleNo
   pX=np.sum(jointP,axis=1)
   pY=np.sum(jointP,axis=0)
   mutual=0
   #print(pX,pY)
   for i in range(bins):
      for j in range(bins):
         if jointP[i,j]>0:
            mutual= mutual+jointP[i,j]*np.log((jointP[i,j])/(pX[i]*pY[j]))
               
   X_entro=-np.sum(pX[pX>0]* np.log(pX[pX>0]))
   Y_entro=-np.sum(pY[pY>0]* np.log(pY[pY>0]))
   nmi=mutual/np.sqrt(X_entro*Y_entro)   
   return nmi

def read_data():
   train='ASSN1_Q2\MNIST\PCAMnist_test.csv'
   test='ASSN1_Q2\MNIST\PCAMnist_test.csv'

   """with open(train,'r') as f:
        train_data = np.array([line.strip().split(',') for line in f], dtype=int)
   X_train, Y_train = train_data[:, :-1], train_data[:, -1]
   with open(test,'r') as f:
        test_data = np.array([line.strip().split(',') for line in f], dtype=int)"""
   train_data = np.genfromtxt(train, delimiter=',',dtype=int)
   test_data = np.genfromtxt(test, delimiter=',',dtype=int)   
   return train_data,test_data  
  

def train_test_split():
   train_data,test_data=read_data()   
   X_train,Y_train= train_data[:, :-1], train_data[:, -1]
   X_test,Y_test= test_data[:, :-1], test_data[:, -1]   
   # Complete code for getting train test split
   return X_train,Y_train,X_test, Y_test 
  

X_train,Y_train,X_test,Y_test=train_test_split()
print("X_train shape:",X_train.shape)
print("Y_train shape:",Y_train.shape)
print("X_test shape:",X_test.shape)
print("Y_test shape:",Y_test.shape)