'''
Module For Math calculation that used in Artificial Neural Network
'''
import numpy as np

def z_score(X: np.ndarray, mean = 0, std = 0) :
    Xshape = X.shape
    if type(mean) != int and type(std) != int:
        normal = (X[:] - mean) / std
        return normal
    x_mean = [np.mean(X[:,i]) for i in range(Xshape[1])] if len(Xshape) > 1 else np.mean(X)
    x_std = [np.std(X[:,i]) for i in range(Xshape[1])] if len(Xshape) > 1 else np.std(X)
    normal = (X[:] - x_mean) / x_std
    return (x_mean, x_std, normal)


def compute_z(x: np.ndarray, w: np.ndarray, b):
    '''
    Compute result for linear equation for f(x) = w * x + b, where w and x is 1-dimension matrix

    Args :
        x    (ndarray (n, )) : Data with n features, 1 example 
        w    (ndarray (n, )) : Weight matrix, n features per unit
        b    (scalar) : bias vector  
    
    Returns
        z (scalar)  : result of the equation
    '''
    n = x.shape[0]
    z = 0
    for i in range(n) :
        # Expression below is equivalen to z = x1 * w1 + x2 * w2 +...+ xn * wn
        z += x[i] * w[i]
    z += b
    return z


def sigmoid(z) :
    '''
    Compute result for sigmoid funcion for g(z) = 1 / (1 + e^(-z)). 
    
    This function will map the value of z approaching 0 if negative, and approaching 1 if positive, and 0.5 if 0

    Args :
        z (scalar)  : number to map
    '''
    return 1 / (1 + np.exp(-z))


def ReLU(z) :
    '''
    Compute result for ReLU (Rectified Linear Unit) funcion where g(z) = z if z is positive and g(z) = 0 if z is negative.

    Args :
        z (scalar)  : number to map
    '''
    g = max(0, z)
    return g


def logistic_cost(y_predict: np.ndarray, y: np.ndarray) :
    """
    Computes logistic cost L(f_wb(x), y) = -y * log(f_wb(x)) - (1 - y) * log(1 - f_wb(x))

    Args:
        X (ndarray (m,n)) : Data, m examples with n features
        y (ndarray (m, )) : target values
        W (ndarray (n,j)) : model parameters for n features, and j neuron unit
        b (ndarray (j, )) : model parameter
      
    Returns:
        cost (scalar): cost for entire dataset
    """
    m = y.shape[0]
    m = y.shape[0]
    logloss = np.multiply(-y, np.log(y_predict)) - np.multiply(1 - y, np.log(1 - y_predict))
    cost = 1 / m * np.sum(logloss)
    return cost


def dy_dz(a_out):
    '''
    Compute Partial derivative of Y with respect to Z, or dY/dZ

    Args:
        a_out (ndarray (m, )) : Result of forward propagation, with m examples
    
    Returns:
        dy_dz (ndarray (m, )) : Result of partial derivative
    '''
    dy_dz = a_out * (1 - a_out)
    return dy_dz


def gradient_w(cost_function: callable, w, x):
    result = (cost_function(compute_z(x, w+1, 0)) - cost_function(compute_z(x, w-1, 0))) / 2
    return result

if __name__ == '__main__' :
    data = np.array([1, 2, 3, 4, 5])
    _, _, normal = z_score(data)
    print(normal)
    print(ReLU(10), ReLU(-3))