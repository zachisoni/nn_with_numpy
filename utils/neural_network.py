import csv
import utils.math_calculation as mc
import numpy as np

def init_params(n_feature, n_y, n_layer, n_unit: list):
    '''
    Initialize random parameters w and b, that used as first w and b

    Args:
        n_feature (scalar) : number of features (column) in dataset
        n_y       (scalar) : number of ouput layer, 1 if binary classification, number of class if multiclass classification
        n_layer   (scalar) : number of hidden layers
        n_unit    (list)   : number of unit in each hidden layer, where n_unit length is n_layer
    
    Returns:
        W (list) : weight parameter with n_layer length
        b (list) : bias parameter with n_layer length
    '''
    W = []
    b = []
    # 0 to number of hidden layer + output layer
    for i in range(n_layer + 1):
        # use previous layer's unit as number of input if current layer is not the first layer
        inputs = n_unit[i - 1] if i != 0 else n_feature
        # use current layer's unit as number of output if current layer is not output layer
        outputs = n_unit[i] if i < n_layer else n_y
        W.append(np.random.randn(inputs, outputs) * 0.01)
        b.append(np.random.randn(1, outputs))
    #return a dictionary
    return {'W' : W, 'b' : b}


def dense_layer(x_in: np.ndarray, W: np.ndarray, b: np.ndarray, activation: callable = lambda z : z):
    '''
    Create fully connected layer. A layer which every unit take and process all input

    Args:
        x_in       (ndarray(1,n)) : Input from one row with n feature
        W          (ndarray(n,j)) : model parameters for n features and j neuron unit
        b          (ndarray(1,j)) : model parameters for n features and j neuron unit
        activation (callable)     : activation function for ouput of this layer
    
    Returns:
    '''
    units = W.shape[1]
    a_out = np.zeros((units, 1))
    for j in range(units):
        #for each unit, compute the output
        z = mc.compute_z(x_in, W[:, j], b[0, j])
        g = activation(z)
        a_out[j] = g
    # return the falttened output
    return (a_out.flatten())


def forward_prop(X: np.ndarray, W: list, b: list):
    rows = X.shape[0]
    layers = len(W)
    p = np.zeros((rows, 1))
    activation = mc.ReLU
    # array of outputs
    a = [np.zeros((rows, w.shape[1])) for w in W]
    for i in range(rows) :
        a_in = X[i]
        for j in range(layers) :
            if j == (layers - 1):
                activation = mc.sigmoid
            # compute output for each layer,for each row of input
            z = dense_layer(a_in, W[j], b[j], activation)
            if j < (layers - 1):
                a[j][i] = z.flatten()
            a_in = z
        p[i] = a_in
    # return the output of output layer and output for each layer
    return (p, a)
    

def back_prop(X: np.ndarray, Y: np.ndarray, y_pred: np.ndarray, a: list, W: list):
    m = X.shape[0]
    layers = len(W)

    dJ_dz = [[] for _ in range(layers)]
    dJ_dw = [[] for _ in range(layers)]
    dJ_db = [[] for _ in range(layers)]
    # compute gradient in reverse order
    for i in range(layers-1, -1, -1):
        # use X as input if its the first layer
        a_in = X if i == 0 else a[i-1]
        if i == (layers - 1) :
            # this will compute for the output layer
            dJ_dz[i] = y_pred - Y
        else :
            # expression below is same as dJ/dz1 = dJ/dz2 * w2 * a1 * (1 - a1)
            dJ_dz[i] = np.dot(dJ_dz[i+1], W[i+1].T) * mc.dy_dz(a[i])
        dJ_dw[i] = 1/ m * np.dot(a_in.T, dJ_dz[i])
        dJ_db[i] = 1/ m * np.sum(dJ_dz[i], axis= 0) 
    
    return {'dW': dJ_dw, 'db' : dJ_db}


def update_params(W: list, b: list, gradients: dict, learning_rate = 1.2):
    layers = len(W)
    dW = gradients['dW']
    db = gradients['db']
    new_W = []
    new_b = []
    for i in range(layers):
        w_i = W[i] - learning_rate * dW[i]
        b_i = b[i] - learning_rate * db[i]
        new_W.append(w_i)
        new_b.append(b_i)
    return {'W': new_W, 'b' : new_b}


def create_network( layer_num = 1, unit_num = [3, 1]):
    pass


def train(X: np.ndarray, y: np.ndarray, n_unit: list, num_iteration=100, learning_rate=1.2, print_cost = True):
    '''
    Train the Network to predict y with input X, this will return W and b parameters that ready to use in prediction

    Args:
        X       (ndarray(m,n)) : input data with m rows, and n features
        y        (ndarray(m,)) : label of dataset with m rows
        n_layer       (scalar) : number of hidden layers
        n_unit          (list) : number of unit in each hidden layer
        num_iteration (scalar) : number of iteration in train process
        learning_rate (scalar) : (alpha) used when updating parameters, try adjust this if cost in training is still high
        print_cost   (boolean) : determine to print the cost for each iteration or not
    
    Returns:
        W (list) : trained weight parameter
        b (list) : trained bias parameter
    '''
    initial_lr = learning_rate
    n_layer = len(n_unit)
    # get x features number and the unique value of y
    n_x, y_uniq = X.shape[1], np.unique(y)
    # if there just 2 unique value, then it's binary classification with one unit in output layer
    n_y = 1 if len(y_uniq) == 2 else len(y_uniq)
    # initialize paramter W and b
    params = init_params(n_x, n_y, n_layer, n_unit)

    for i  in range(num_iteration):
        # do forward propagation to get output of network and ouput of each layer
        y_pred, a = forward_prop(X, params['W'], params['b'])
        # compute cost
        cost = mc.logistic_cost(y_pred, y)
        grads = back_prop(X, y, y_pred, a, params['W'])

        params = update_params(params['W'], params['b'], grads, learning_rate)

        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost), ' -  lr =', learning_rate)

    print("Trainig complete!")
    # return parameters W and b that the result of training
    return params



def predict(X: np.ndarray, W: list, b: list):
    y_pred = forward_prop(X, W, b)[0]
    rows = X.shape[0]
    prediction = np.zeros(rows)
    for i in range(rows) :
        prediction[i] = 1 if y_pred[i] >= 0.5 else 0
        if prediction[i] == 1:
            print('This one is good roasted coffee!!!') 
        else: 
            print('This is not so good coffee..')
    return prediction


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

    # W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
    # b1_tmp = np.array( [-9.82, -9.28,  0.96] )
    # W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
    # b2_tmp = np.array( [15.41] )

    x_mean, x_std, x_norm = mc.z_score(X)

    # W = [W1_tmp, W2_tmp]
    # b = [b1_tmp, b2_tmp]

    # X_tst = np.array([
    #     [200,13.9],  # postive example
    #     [200,17]])   # negative example
    # X_tstn = mc.z_score(X_tst, x_mean, x_std)
    # # print('x_test_normal = ',  X_tstn, 'dim = ', X_tstn.shape)  # remember to normalize
    # predictions = predict(X_tstn, W, b)
    # yhat = np.zeros_like(predictions)
    # for i in range(len(predictions)):
    #     if predictions[i] >= 0.5:
    #         yhat[i] = 1
    #     else:
    #         yhat[i] = 0
    # print(f"decisions = \n{predictions}")
    # print("w1 = ", W[0], ", shape = ", W[0].shape)

    parameters = train(x_norm, Y, 2, [5, 5], num_iteration=1000, learning_rate=1.1)
    # for i in range(3) :
    #     print("W", i+1, " = \n" + str(parameters["W"][i]))
    #     print("b", i+1," = \n" + str(parameters["b"][i]))
    X_tst = np.array([
        [200,13.9],  # postive example
        [200,17]])   # negative example
    X_tstn = mc.z_score(X_tst, x_mean, x_std)
    # print('x_test_normal = ',  X_tstn, 'dim = ', X_tstn.shape)  # remember to normalize
    predictions = predict(X_tstn, parameters['W'], parameters['b'])