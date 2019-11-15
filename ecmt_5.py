import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
np.set_printoptions(suppress=True)
                    
def minFind(theta, w, q):
    
    z = np.dot(w, theta)
    h = norm.cdf(z) 
    return (-q * np.log(h) - (1 - q) * np.log(1 - h)).sum()
    
if __name__ == "__main__":
    
    df = pd.read_excel(r'data.xlsx')
    

    from models import OrdinaryLeastSquares, Heckman
    
    reg =  OrdinaryLeastSquares()
    print("OLS Coefficients \n")
    reg.fit(df.GRE.values,df.wage.values)
    print(reg.coefficients)
    
    print("Compared with built in")
    
    regmodel = sm.OLS(df.wage.values.reshape(-1,1), np.concatenate((df.GRE.values.reshape(-1,1),np.ones((df.wage.values.reshape(-1,1).shape[0], 1))), axis = 1))
    model = regmodel.fit()
    print(model.summary())
    
    heckman = Heckman()
    print()
    print("Step 1 Probit Model")
    
    theta = heckman.Probit(df, 'wage', ['GRE','GPA'])
    theta = np.array([-5.1917, .0064543, .0894813])
    
    print("probit parameters")
    print(theta)
    
    print("Step 2")
    
    parameters = heckman.SecondStep(df, theta, 'wage', ['GRE'])
    print(parameters)

    

    
    