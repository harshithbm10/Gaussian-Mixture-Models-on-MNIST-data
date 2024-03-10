import numpy as np

class GMM:
	def __init__(self,k=3, max_iter=10):
		self.k=k
		self.max_iter=max_iter
		self.weights=None
		self.mu=None
		self.covariances=None		
		#pass
		self.responsibilities= None
	
	def initialize(self, X):
		#weights= 1/k (vector?) mu (random sample of X)  covariance=(diagonal matrix np.tilr)
		self.sampleNO=X.shape[0]		
		self.weights=np.ones(self.k)/self.k #phi
		#self.mu=X[np.random.choice(sampleNO,self.k),:]		
		self.mu= X[np.random.choice(self.sampleNO,self.k,replace=False),:]
		self.covariances= np.tile(np.diag(np.var(X, axis=0)),(self.k,1,1 ))	
		#pass	

	def e_step(self, X):
		# E-Step: update weights and phi holding mu and sigma constant		
		#responsibility= mix*multivariate
		"""responsibility=np.zeros((X.shape[0],self.k))
		for i in range(self.k):
			responsibility[:,i] =self.weights[i]*self.multivariate_normal(X,self.mu[i],self.covariances[i])
		#denominator=np.sum(responsibility,axis=1)
		denominator=np.sum(responsibility,axis=1,keepdims=True)
		self.responsibilities=(responsibility/denominator)"""
		self.responsibilities =self.predict_proba(X)
		return self.responsibilities
		
	def m_step(self, X):
		# M-Step: update mu and sigma holding phi and weights constant
		#sampleNO=X.shape[0]
		total_responsibilties=np.sum(self.responsibilities,axis=0)
		self.mu=np.dot(self.responsibilities.T ,X)/total_responsibilties[:,np.newaxis]
		for i in range(self.k):
			diff=X-self.mu[i]
			covariance = np.dot(self.responsibilities[:, i] * diff.T, diff) /total_responsibilties[i]
			covariance += 1e-6 * np.eye(X.shape[1])
			self.covariances[i] = covariance
			self.covariances[i]=np.dot(self.responsibilities[:,i]*diff.T,diff)/total_responsibilties[i]
		#pass returns nothing

	def fit(self, X):
		self.initialize(X)
		for iteration in range(self.max_iter):
			responsibilities=self.e_step(X)
			self.m_step(X)
		#returns nothing

	"""def multivariate_normal(self,x,mu,cov):
		N=X.shape[0]
		numerator =(-0.5*np.sum((x-mu)*np.linalg.solve(cov,(x-mu).T).T, axis=1))
		denominator=np.sqrt((2*np.pi)**N*np.linalg.det(cov))
		#pass
		return np.exp(numerator)/denominator"""
	
		#exponent=-0.5 *np.sum(np.dot((x - mu), inv_cov)*(x - mu),axis=1)
		#np.exp(exponent)/np.sqrt((2 * np.pi) ** n_features * det_cov)
	
	def multivariate_normal(self, x, mu, cov):		
		N=x.shape[1]
		det_cov=np.linalg.det(cov)		
		inv_cov=np.linalg.inv(cov)
		diff=x-mu
		exponent =-0.5*np.sum(np.dot(diff, inv_cov)*diff,axis=1)
		log_normalizer=-0.5*N*np.log(2*np.pi)-0.5*np.log(det_cov)
		log_prob = exponent+log_normalizer
		return log_prob

	def predict_proba(self, X):
		responsibilities = np.zeros((X.shape[0], self.k))
		for i in range(self.k):
			responsibilities[:, i] = self.weights[i] * self.multivariate_normal(X, self.mu[i], self.covariances[i])
		responsibilities = responsibilities / np.sum(responsibilities, axis=1, keepdims=True)
		return responsibilities

	def predict(self, X):
		weights = self.predict_proba(X)
		return np.argmax(weights ,axis=1)
	
class KMeans:
	def __init__(self,k=3, max_iter=10):
		self.k=k
		self.max_iter=max_iter
		self.centroids=None
		self.clusterName=None	
			
	def initialize(self, X):
		#self.centroids=X[np.random.choice(X.shape[0],self.k)] #assign k random centroids 
		self.centroids=X[np.random.choice(X.shape[0],self.k,replace=False)]
			
	def fit(self, X):
		self.initialize(X)
		for iteration in range(self.max_iter):
			clusterName=self.predict(X)
			#self.predict(X)
			for i in range(self.k):
				self.centroids[i]=np.mean(X[clusterName==i],axis=0)
				#self.centroids[i]=np.mean(X[self.clusterName==i],axis=0)
			#pass

	def predict(self, X):
		distances=np.linalg.norm(X[:,np.newaxis]-self.centroids,axis=2)
		#self.clusterName=np.argmin(distances,axis=1)
		clusterName=np.argmin(distances, axis=1)
		return clusterName
	



