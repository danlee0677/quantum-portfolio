import numpy as np
from scipy.optimize import minimize
from pypfopt.discrete_allocation import DiscreteAllocation


class HigherMomentPortfolioOptimizer:
    
    def __init__(self, stocks, expected_returns, covariance_matrix, coskewness, cokurtosis, risk_aversion=3):
        self.stocks = stocks
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.coskewness = coskewness
        self.cokurtosis = cokurtosis
        self.risk_aversion = risk_aversion
        self.weights = None
        self.x = None
    
        
    def get_objective(self):
        def objective(w):
            mu = np.dot(w, self.expected_returns)
            variance_term = (self.risk_aversion/2) * np.einsum('ij,i,j->', self.covariance_matrix, w, w)
            skewness_term = (self.risk_aversion/6) * np.einsum("ijk,i,j,k->", self.coskewness, w, w, w)
            kurtosis_term = (self.risk_aversion/24) * np.einsum("ijkl,k,j,i,l->", self.cokurtosis, w, w, w, w)
            value = -(mu - variance_term + skewness_term - kurtosis_term)
            return value
        return objective
    

    def get_objective_unconstrained(self):
        penalizer = 1
        def objective_unconstrained(w):
            mu = np.dot(w, self.expected_returns)
            variance_term = (self.risk_aversion/2) * np.einsum('ij,i,j->', self.covariance_matrix, w, w)
            skewness_term = (self.risk_aversion/6) * np.einsum("ijk,i,j,k->", self.coskewness, w, w, w)
            kurtosis_term = (self.risk_aversion/24) * np.einsum("ijkl,k,j,i,l->", self.cokurtosis, w, w, w, w)
            value = -(mu - variance_term + skewness_term - kurtosis_term) + penalizer * (np.sum(w) - 1)**2
            return value
        return objective_unconstrained
    
    
    def optimize_portfolio_with_higher_moments(self):
        num_assets = len(self.expected_returns)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(num_assets)]
        objective = self.get_objective()
        w0 = np.ones(num_assets) / num_assets
        result = minimize(objective, w0, bounds=bounds, constraints=constraints)
        self.weights = dict(zip(self.stocks, result.x))
        self.x = result.x
        return self.weights
    

    def optimize_portfolio_with_higher_moments_unconstrained(self):
        num_assets = len(self.expected_returns)
        bounds = [(0, 1) for _ in range(num_assets)]
        objective = self.get_objective_unconstrained()
        w0 = np.ones(num_assets) / num_assets
        result = minimize(objective, w0, bounds=bounds)
        self.weights = dict(zip(self.stocks, result.x))
        self.x = result.x
        return self.weights
    

    def get_optimal_value(self):
        objective = self.get_objective()
        if self.x is None:
            w = self.optimize_portfolio_with_higher_moments()
        else:
            w = self.x
        return -objective(w)
    
    
    def get_discrete_allocation(self, weights, latest_prices, total_portfolio_value):
        allocator = DiscreteAllocation(weights, latest_prices, total_portfolio_value)
        self.discrete_allocation = allocator.lp_portfolio()
        return self.discrete_allocation
        