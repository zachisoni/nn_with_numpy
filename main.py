import utils.data as data
from utils.neural_network import predict
from numpy import array
from utils.math_calculation import z_score

command = data.retrieve_param()
exec(command)
x_mean_np = array(x_mean)
x_std_np = array(x_std)

def main():
    temp = int(input("Input temperature to roast (Celcius) : "))
    minute = int(input("Input time to roast coffee (minutes) : "))
    inputs = array([[temp, minute]])
    input_norm = z_score(inputs, x_mean_np, x_std_np)

    predict(input_norm, W, b)

main()