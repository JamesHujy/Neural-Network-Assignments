# CNN model for Cifar-10

It implements the CNN model for cifar-10 classification problem based on tensorflow.     

CNN model has two convolutional layers and two max pooling layers, using Relu as activation function. The accuracy can reach above 79%. 

Considering slow speed to train CNN model, we use GPU to train the model and corresponding code is implemented in the main.py.

### Run
Type `python main.py` can run the model. User can adjust batch size, dropout rate, epochs, GPU index and whether to use batch normalization in command line. Parameters of the best performance have been set as default. Users are expected to adjust to GPU index according to the GPU condition as following, for example.

`python main.py --device 4`

### Instruction
`model.py` defines the structure of the model. `main.py` implements the training of the model.