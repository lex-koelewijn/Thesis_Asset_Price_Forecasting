import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy.linalg import inv

# actual      = n-vector of actual values
# benchmark   = n-vestor of forecasts for the restricted model
# model       = n-vector of forecasts for the unresitrcted model. 
def clarkWestTest(actual, benchmark, model):
    size = actual.shape[0]
    f_hat = (actual - benchmark)**2 - ((actual-model)**2 - (benchmark-model)**2)
    f_hat = f_hat.values.reshape(size,1)
    X_f = np.ones(f_hat.shape).reshape(size,1)
    beta_f = inv(np.matmul(np.transpose(X_f),X_f))*np.matmul(np.transpose(X_f),f_hat) 
    e_f = f_hat - beta_f*X_f
    sigma2_e = np.matmul(np.transpose(e_f), e_f)/(size-1)
    cov_beta_f =  sigma2_e * inv(np.matmul(np.transpose(X_f),X_f))
    MSPE_adjusted = beta_f/np.sqrt(cov_beta_f)
    p_value = 1 - norm.cdf(MSPE_adjusted)
    return [MSPE_adjusted[0][0], p_value[0][0]]