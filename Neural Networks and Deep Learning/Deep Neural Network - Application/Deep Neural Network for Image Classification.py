import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils import *

# GRADED FUNCTION: two_layer_model
def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x,n_h,n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x,n_h,n_y)
    ### END CODE HERE ###

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Loop (gradient descent)
    for i in range(0,num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1,cache1 = linear_activation_forward(X,W1,b1,'relu')
        A2,cache2 = linear_activation_forward(A1,W2,b2,'sigmoid')
        ### END CODE HERE ###

        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2,Y)
        ### END CODE HERE ###

        # Initializing backward propagation
        dA2 = - (np.divide(Y,A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2,db2 = linear_activation_backward(dA2,cache2,'sigmoid')
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,'relu')
        ### END CODE HERE ###

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters,grads,learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i,np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

# GRADED FUNCTION: n_layer_model
def L_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)
    costs = [] # keep track of cost
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop(gradient descent)
    for i in range(0,num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL,caches = L_model_forward(X,parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL,Y)
        ### END CODE HERE ###

        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL,Y,caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters,grads,learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


    return

if __name__ == '__main__':

    # matplotlib inline
    plt.rcParams['figure.figsize'] = (5.0,4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # load_ext autoreload
    # autoreload 2

    np.random.seed(1)

    """
    - a training set of m_train images labelled as cat (1) or non-cat (0)
    - a test set of m_test images labelled as cat and non-cat
    - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
    """
    train_x_orig,train_y,test_x_orig,test_y,classes = load_data()

    """
    The following code will show you an image in the dataset. 
    Feel free to change the index and re-run the cell multiple times to see other images.
    """

    # Example of a picture
    index = 10
    plt.imshow(train_x_orig[index])
    plt.show()
    print('y = '+str(train_y[0,index]) + ".  It's a "+classes[train_y[0,index]].decode("utf-8") + "picture.")

    # Explore your dataset
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))

    # Reshape the training and test examples
    print(train_x_orig.shape)
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T
    print(train_x_flatten.shape)
    print(test_x_flatten.shape)

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    ### CONSTANTS DEFINING THE MODEL ####
    n_x = 12288
    n_h = 7
    n_y = 1
    layers_dim = (n_x,n_h,n_y)
    parameters = two_layer_model(train_x,train_y,layers_dims = (n_x,n_h,n_y),num_iterations=2500,print_cost=True)

    # Now, you can use the trained parameters to classify images from the dataset. To see your predictions on the
    # training and test sets, run the cell below.
    print('Accuracy of train:')
    predictions_train = predict(train_x,train_y,parameters)

    print('Accuracy of test:')
    predictions_test = predict(test_x,test_y,parameters)

    #  L-layer Neural Network
    layers_dims = [12288,20,7,5,1]
    parameters = L_layer_model(train_x,train_y,layers_dims,num_iterations=1600,print_cost=True)

    print('Accuracy of train:')
    pred_train = predict(train_x, train_y, parameters)

    print('Accuracy of test:')
    pred_test = predict(test_x, test_y, parameters)

    #First, let's take a look at some images the L-layer model labeled incorrectly.
    # This will show a few mislabeled images
    # print_mislabeled_images(classes,test_x,test_y,pred_test)

    #  Test with your own image (optional/ungraded exercise)
    ## START CODE HERE ##
    my_image = "my_image.jpg" # change this to the name of your image file
    my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
    ## END CODE HERE ##

    fname = "image/" + my_image
    img = Image.open(fname)
    # convert to RGB
    image = np.array(img.convert("RGB"))
    # image = np.array(ndimage.imread(fname,flatten=False))
    # print(image.shape)
    my_image = scipy.misc.imresize(image, size=(num_px, num_px,3))
    # print(my_image.shape)
    my_image = my_image.reshape((num_px * num_px * 3, 1))
    my_predicted_image = predict(my_image,my_label_y,parameters)
    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")