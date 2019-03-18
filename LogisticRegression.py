import numpy as np

#Let's create logistic regression

#create sigmoid function
def sigmoid(S):
    """
    S: an numpy array
    return sigmoid function of each element of S
    """
    return 1/(1 + np.exp(-S))

def prob(X, w):
    """
    X: a numpy array of shape(N, d)-N data point with d size
    w: a numpy array of shape (d, 1)
    return the sigmoid of X*w
    """
    return sigmoid(X.dot(w))

def loss(w, X, y, lam):
    """
        Calculating the loss function constructed via
        Maximum Likehood Estimation approach
    X, w as in prob
    y: a numpy array of shape (N) which is expected to contain
        label. Each element = 0 or 1
    lam: weight decay as a regularization strategy to avoid over-fitting
    """
    z = prob(X, w)
    return - np.mean(y*np.log(z) + (1-y)*np.log(1-z)) + lam*0.5/X.shape[0]*np.sum(w*w)

def logistic_regression(X, y, lam = 0.001, lr = 0.1, nepoches = 2000):
    """
        Finding parameter for logistic regression model
        via Gradient descent optimization
    lr: learning rate
    nepoches: number of epoches
    X, y, described in previous function
    """
    N = X.shape[0]
    X = np.concatenate((X, np.ones((N, 1))), axis = 1)
    w = w_old = w_init = np.random.randn(X.shape[1], 1)
    d = X.shape[1]
    
    loss_hist = [loss(w, X, y, lam)]
    ep = 0
    while ep < nepoches:
        ep += 1
        z = prob(X, w)
        w = w - lr*(X.T.dot((y - z)) + lam*w)
        loss_hist.append(loss(w, X, y, lam))
        if np.linalg.norm(w - w_old)/d < 1e-6:
            break
        w_old = w
        return w, loss_hist

def predict(X, y, threshold = 0.5):
    """
    predict output of each row of X
    X: a numpy array of shape (N, d)
    threshold: a threshold between 0 and 1
    return a numpy array, each element is 0 or 1
    """
    res = np.zeros(X.shape[0])
    res[np.where(prob(X, logistic_regression(X, y, lam = 0.001, lr = 0.1, nepoches = 2000)[0])) > threshold[0]] = 1
    return res

#source: https://machinelearningcoban.com/2017/01/27/logisticregression/

if __name__ == '__main__':
    X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
             2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
    y = np.array([[0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]]).T
    w, loss_hist = logistic_regression(X, y)
    print('Weight = {} \nBias = {}'.format(w[0][0],w[-1][0]))
    print('Loss after the run = {}'.format(loss_hist[-1]))
