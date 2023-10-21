from utils.neural_network import train
from utils.math_calculation import z_score
import numpy as np

def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
    X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))
    
    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1,1))



if  __name__ == '__main__' :
    X,Y = load_coffee_data();
    x_mean, x_std, x_norm = z_score(X)

    parameters = train(x_norm, Y, [5, 5])

    print("Write results to file...")
    with open('data/parameters.txt', 'w') as file :
        file.write(f"x_mean = {x_mean}\n")
        file.write(f"x_std = {x_std}\n")
        file.write(f"W = {parameters['W']}\n")
        file.write(f"b = {parameters['b']}\n")
    print("Result saved to file data/parameters.txt")

    
    