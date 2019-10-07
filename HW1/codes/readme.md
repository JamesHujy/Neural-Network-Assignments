# MLP for MNIST

Hyperparameters configurations 

We implement a simple MLP neural network and train it on MNIST dataset. 

## Running
Open the directory where the codes belong and run the run_mlp.py. Hyperparameters can be set in the command line. Following command set the hyper parameters for each experiment. They can be copied directly.

#### 1-layer relu+euclidean
`python3 run_mlp.py --hidden_layer 1 --learning_rate 0.01 --weight_decay 1e-4 --loss euclidean --max_epoch 30 --batch_size 30`  
  
#### sigmoid + softmax 
 `python3 run_mlp.py --learning_rate 0.02 --weight_decay 0.0001 --loss softmax --activation sigmoid --max_epoch 200 --momentum 0.95 --batch_size 200`  

#### sigmoid + euclidean
`python3 run_mlp.py --learning_rate 0.02 --weight_decay 0.0001 --loss euclidean --activation sigmoid --max_epoch 200 --momentum 0.95 --batch_size 200`  96.

#### 2-layer Relu + eulidean
`python3 run_mlp.py --hidden_layer 2 --learning_rate 0.01 --weight_decay 1e-4 --loss euclidean --max_epoch 30 --batch_size 30` 98.43%

#### 2-layer sigmoid + softmax
`python3 run_mlp.py --learning_rate 0.035 --weight_decay 0.0001 --loss softmax --activation sigmoid --max_epoch 300 --momentum 0.95 --hidden_layer 2 --batch_size 300`

#### 2-layer sigmoid + euclidean
`python3 run_mlp.py --activation sigmoid --loss euclidean --learning_rate 0.03 --weight_decay 0.0001 --momentum 0.95 --max_epoch 300 --hidden_layer 2 --batch_size 300`

## Other Instructions
Layers.py and loss.py implements the network model and loss, including activation functions, loss functions and linear network's forward and backward function. solve_net.py and run_mlp.py implement training part. The loss and accuracy data generated during training will be saved in the npy directory. draw.py implement the plot function for the graph.