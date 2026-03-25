import numpy as np

from stats_functions import cokurtosis, coskewness

class DataProvider():

    def __init__(self, data) -> None:
        self.data = data
        
        # First compute the period returns
        # This is a standard step in portfolio optimization
        _div_func = np.vectorize(DataProvider._divide)
        self.period_returns = _div_func(np.array(self.data)[:, 1:], np.array(self.data)[:, :-1]) - 1


    @staticmethod
    def _divide(val_1, val_2):
        if val_2 == 0:
            if val_1 == 0:
                return 1
            return np.nan
        return val_1 / val_2

    
    def get_period_return_mean_vector(self):
        self.period_return_mean = np.mean(self.period_returns, axis=1)
        return self.period_return_mean


    def get_period_return_covariance_matrix(self):
        self.period_return_cov = np.cov(self.period_returns)
        return self.period_return_cov
    
    
    def get_period_return_coskewness_tensor(self):
        self.period_return_skew = np.array(coskewness(self.period_returns))
        return self.period_return_skew
    
    
    def get_period_return_cokurtosis_tensor(self):
        self.period_return_kurt = np.array(cokurtosis(self.period_returns))
        return self.period_return_kurt