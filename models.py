import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
import pandas as pd
import scipy

class HelperFunctions():
    
    def add_constant(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    

class OrdinaryLeastSquares(HelperFunctions):
    
    def __init__(self, addIntercept = True):
        self.coefficients = []
        self.addIntercept = True
        super().__init__()
        
    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        if self.addIntercept is True:
            X = self.add_constant(X)
            self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)


class Heckman():
    
    def __init__(self):
        pass
    
    def minFind(self,theta, w, q):
    
        z = np.dot(w, theta)
        h = norm.cdf(z) 
        return (-q * np.log(h) - (1 - q) * np.log(1 - h)).sum()
    
    
    
    def Probit(self,df, endog, exogList):
        binary = df[endog]
        
        X = df[exogList].values
        
        
        binary.loc[binary > 0] = 1.
        binary = binary.values
        
    
        w = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        q = binary
        
        
        probit_params = scipy.optimize.fmin(func=self.minFind, x0=np.transpose(np.array([-10,0,0])), args = (w,q))
        theta = probit_params
        
        return theta
    
    
    def SecondStep(self, df, theta ,endog, exog):
        
        df = pd.read_excel(r'data.xlsx')
        
        df = df.loc[df.wage > 0]
        
        fitted = theta[0] + theta[1] * df.GRE.values.reshape(-1,1) + theta[2] * df.GPA.values.reshape(-1,1)
        
        inverse_mill = norm.pdf(fitted) / norm.cdf(fitted)
        
        df.loc[:,'inverse mill'] = inverse_mill 
        df.loc[:,'intercept'] = 1.0
        
        
        step2model = sm.OLS(df.wage, df[['intercept','inverse mill','GRE']])
        step2res = step2model.fit()
        print("Second Step Parameters")
        print(step2res.params)

        



