from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': False}
        self.reset(parameters)

        self.Xtrain_Zero_Mean = None
        self.Xtrain_Zero_Var = None
        self.Xtrain_One_Mean = None
        self.Xtrain_One_Var = None


    def reset(self, parameters):
        self.resetparams(parameters)
    def learn(self, Xtrain, ytrain):

        if self.params ['usecolumnones'] == False:
            Xtrain = Xtrain[:, :-1]

        dataset = np.insert(Xtrain, Xtrain.shape[1], ytrain, axis=1)

        Xtrain_Zero = dataset[dataset[:, (len(dataset[0]) - 1)] == 0]
        Xtrain_one = dataset[dataset[:, (len(dataset[0]) - 1)] == 1]

        self.Xtrain_Zero_Mean = Xtrain_Zero.mean(axis=0)
        self.Xtrain_Zero_Std = Xtrain_Zero.std(axis=0)
        self.Xtrain_One_Mean = Xtrain_one.mean(axis=0)
        self.Xtrain_One_Std = Xtrain_one.std(axis=0)

    def predict(self, Xtest):
        if self.params['usecolumnones'] == False:
            Xtest = Xtest[:, :-1]
        ytest = []
        for row in Xtest:
            p_zero = 1
            p_one = 1
            for datapoint_index in range(row.shape[0]):
                p_zero = p_zero * (utils.calculateprob(row[datapoint_index], self.Xtrain_Zero_Mean[datapoint_index], self.Xtrain_Zero_Std[datapoint_index]))
                p_one = p_one * (utils.calculateprob(row[datapoint_index], self.Xtrain_One_Mean[datapoint_index], self.Xtrain_One_Std[datapoint_index]))

            if p_one >= p_zero:
                ytest.append(1)
            else:
                ytest.append(0)

        return ytest

class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        #if self.params['regularizer'] is 'l1':
        #    self.regularizer = (utils.l1, utils.dl1)
        #elif self.params['regularizer'] is 'l2':
        #    self.regularizer = (utils.l2, utils.dl2)
        #else:
        #    self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def learn(self, Xtrain, ytrain):

        weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),ytrain)
        initial_error = np.linalg.norm(np.subtract(ytrain,utils.sigmoid(np.dot(Xtrain, weights))))

        tolerance = 0.01
        gra_err = 0
        while abs(initial_error - gra_err) > tolerance:
            initial_error = gra_err

            p = utils.sigmoid(np.dot(Xtrain, weights))
            # Calculating the diagonal matrix
            diagonal_matrix = np.diag(utils.sigmoid(np.dot(Xtrain, weights)))
            identity_matrix = np.identity(diagonal_matrix.shape[0])
            hessian_value = np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T, diagonal_matrix),np.subtract(identity_matrix,diagonal_matrix)),Xtrain))

            if self.params['regularizer'] is 'l1':
                gradient_value = np.dot(Xtrain.T, np.subtract(ytrain, p)) - (self.params['regwgt'] * utils.dl1(weights))
            elif self.params['regularizer'] is 'l2':
                gradient_value = np.dot(Xtrain.T, np.subtract(ytrain, p)) - (2 * self.params['regwgt'] * utils.dl2(weights))
            elif self.params['regularizer'] is 'ElasticNet':
                gradient_value = np.dot(Xtrain.T, np.subtract(ytrain, p)) - ((self.params['regwgt'] * utils.dl1(weights)) + (2 * self.params['regwgt'] * utils.dl2(weights)))
            else:
                gradient_value = np.dot(Xtrain.T, np.subtract(ytrain, p))

            weights = weights + (np.dot(hessian_value,gradient_value))
            gra_err = np.linalg.norm(np.subtract(ytrain, utils.sigmoid(np.dot(Xtrain, weights))))

        self.weights = weights

    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(ytest)
        return ytest
           
class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.01,
                        'epochs': 10}
        self.reset(parameters)        

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None

    def learn(self, Xtrain, ytrain):

        self.weights = [np.random.randn(5000, 100)]
        self.weights.append(np.random.randn(100, 1))
        self.biases = [np.random.randn(size, 1) for size in [5000, 100]]
        for reps in range(self.params['epochs']):
            index = 0
            for row in Xtrain:
                z1 = 0
                z2 = 0
                hidden_z = []
                output_z = []

                new_bias_hidden = []
                new_bias_output = []

                # --------Forward propagation--------
                for no_of_hidden in range(100):
                    for datapoint_index in range(row.shape[0]):
                        z1 = z1 + (self.weights[0][datapoint_index][no_of_hidden] * row[datapoint_index]) + self.biases[0][datapoint_index][0]
                    hidden_z.append(self.sigmoid(z1)) #TO DO: Write sigmoid function


                for val_index in range(len(hidden_z)):
                    z2 = z2 + (self.weights[1][val_index][0] * hidden_z[val_index]) + self.biases[1][val_index][0]
                output_z.append(self.sigmoid(z2))


                # --------Back propagation--------
                # For hidden layer
                diff = output_z[0] - ytrain[index]
                delta = diff * self.desigmoid(output_z[0])
                new_bias_output.append(delta)
                for val_index in range(len(hidden_z)):
                    self.biases[1][val_index][0] = self.biases[1][val_index][0] - (new_bias_output[0]*(0.01/5000))
                    x = (delta * hidden_z[val_index]) * (0.01/5000)
                    self.weights[1][val_index][0]=self.weights[1][val_index][0] - x

                # For input layer
                for val_index in range(len(hidden_z)):
                    new_bias_hidden.append(delta * self.weights[1][val_index][0] * self.desigmoid(hidden_z[val_index]))

                for no_of_hidden in range(100):
                    for datapoint_index in range(row.shape[0]):
                        self.biases[0][datapoint_index][0] = self.biases[0][datapoint_index][0] - (new_bias_hidden[no_of_hidden]*(0.01/5000))
                        x = (self.weights[0][datapoint_index][no_of_hidden] * delta) *(0.01/5000)
                        self.weights[0][datapoint_index][no_of_hidden] = self.weights[0][datapoint_index][no_of_hidden] - x
                index = index + 1

    def predict(self, Xtest):
        samples = Xtest.shape[0]
        ytest = np.empty(samples)
        i = 0
        for row in Xtest:
            z1 = 0
            z2 = 0

            hidden_z = []

            # Forward propagation
            for no_of_hidden in range(100):
                for datapoint_index in range(row.shape[0]):
                    z1 = z1 + (self.weights[0][datapoint_index][no_of_hidden] * row[datapoint_index]) + \
                         self.biases[0][datapoint_index][0]
                hidden_z.append(self.sigmoid(z1))

            for val_index in range(len(hidden_z)):
                z2 = z2 + (self.weights[1][val_index][0] * hidden_z[val_index]) + self.biases[1][val_index][0]

            if z2 >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0
            i = i+1
        return ytest

    def sigmoid(self, z):
        vecsig = 1.0 / (1.0 + np.exp(np.negative(z)))
        return vecsig

    def desigmoid(self, z):
        vecsig = self.sigmoid(z)
        return vecsig * (1 - vecsig)

class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)
        p = 0.5 * (1 + np.divide(np.dot(Xtrain, weights), np.sqrt(1 + np.square(np.dot(Xtrain, weights)))))

        initial_error = np.linalg.norm(np.subtract(ytrain, p))

        gra_err = 0
        alpha = 0.0001
        old_weight = 0
        while abs(initial_error - gra_err) > 0.01:
            initial_error = gra_err
            old_weight = weights
            grad = np.dot(Xtrain.T,(((1-2*ytrain)/np.sqrt(1+np.square(np.dot(Xtrain,weights))))+(np.dot(Xtrain,weights)/np.sqrt(1+np.square(np.dot(Xtrain,weights))))))

            weights = weights - (alpha*grad)

            p = 0.5 * (1 + np.divide(np.dot(Xtrain, weights), np.sqrt(1 + np.square(np.dot(Xtrain, weights)))))
            gra_err = np.linalg.norm(np.subtract(ytrain, p))

        self.weights = old_weight


    def predict(self, Xtest):
        ytest = 0.5 * (1 + np.divide(np.dot(Xtest, self.weights), np.sqrt(1 + np.square(np.dot(Xtest, self.weights)))))
        ytest = utils.threshold_probs(ytest)
        return ytest
        


