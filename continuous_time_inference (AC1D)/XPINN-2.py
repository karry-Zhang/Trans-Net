import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys

sys.path.insert(0, '../Utilities/')

import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class XPINN:
    # Initialize the class
    def __init__(self, X_0, u0, X_lb1, X_ub1, X_lb2, X_ub2, X_f1_train, X_f2_train, X_i1_train, layers1, layers2):
        self.i = 1

        self.x0 = X_0[:, 0:1]
        self.t0 = X_0[:, 1:2]
        self.u0 = u0
        self.x_lb1 = X_lb1[:, 0:1]
        self.t_lb1 = X_lb1[:, 1:2]
        self.x_ub1 = X_ub1[:, 0:1]
        self.t_ub1 = X_ub1[:, 1:2]

        self.x_lb2 = X_lb2[:, 0:1]
        self.t_lb2 = X_lb2[:, 1:2]
        self.x_ub2 = X_ub2[:, 0:1]
        self.t_ub2 = X_ub2[:, 1:2]

        self.x_f1 = X_f1_train[:, 0:1]
        self.t_f1 = X_f1_train[:, 1:2]
        self.x_f2 = X_f2_train[:, 0:1]
        self.t_f2 = X_f2_train[:, 1:2]

        self.x_i1 = X_i1_train[:, 0:1]
        self.t_i1 = X_i1_train[:, 1:2]

        self.layers1 = layers1
        self.layers2 = layers2

        self.weights1, self.biases1, self.A1 = self.initialize_NN(layers1)
        self.weights2, self.biases2, self.A2 = self.initialize_NN(layers2)

        Config = tf.ConfigProto(allow_soft_placement=True)
        Config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=Config)

        self.x0_tf = tf.placeholder(tf.float64, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float64, shape=[None, self.t0.shape[1]])

        self.x_lb1_tf = tf.placeholder(tf.float64, shape=[None, self.x_lb1.shape[1]])
        self.t_lb1_tf = tf.placeholder(tf.float64, shape=[None, self.t_lb1.shape[1]])
        self.x_ub1_tf = tf.placeholder(tf.float64, shape=[None, self.x_ub1.shape[1]])
        self.t_ub1_tf = tf.placeholder(tf.float64, shape=[None, self.t_ub1.shape[1]])

        self.x_lb2_tf = tf.placeholder(tf.float64, shape=[None, self.x_lb2.shape[1]])
        self.t_lb2_tf = tf.placeholder(tf.float64, shape=[None, self.t_lb2.shape[1]])
        self.x_ub2_tf = tf.placeholder(tf.float64, shape=[None, self.x_ub2.shape[1]])
        self.t_ub2_tf = tf.placeholder(tf.float64, shape=[None, self.t_ub2.shape[1]])

        self.x_f1_tf = tf.placeholder(tf.float64, shape=[None, self.x_f1.shape[1]])
        self.t_f1_tf = tf.placeholder(tf.float64, shape=[None, self.t_f1.shape[1]])
        self.x_f2_tf = tf.placeholder(tf.float64, shape=[None, self.x_f2.shape[1]])
        self.t_f2_tf = tf.placeholder(tf.float64, shape=[None, self.t_f2.shape[1]])

        self.x_i1_tf = tf.placeholder(tf.float64, shape=[None, self.x_i1.shape[1]])
        self.t_i1_tf = tf.placeholder(tf.float64, shape=[None, self.t_i1.shape[1]])

        self.ub1_pred = self.net_u1(self.x_f1_tf, self.t_f1_tf)
        self.ub2_pred = self.net_u2(self.x_f2_tf, self.t_f2_tf)

        self.u1_0_pred = self.net_u1(self.x0_tf, self.t0_tf)

        self.u1_lb_pred, self.u1_lb_x_pred  = self.net_u1_uv(self.x_lb1_tf, self.t_lb1_tf)
        self.u1_ub_pred, self.u1_ub_x_pred  = self.net_u1_uv(self.x_ub1_tf, self.t_ub1_tf)

        self.u2_lb_pred, self.u2_lb_x_pred = self.net_u2_uv(self.x_lb2_tf, self.t_lb2_tf)
        self.u2_ub_pred, self.u2_ub_x_pred = self.net_u2_uv(self.x_ub2_tf, self.t_ub2_tf)

        self.f1_pred, self.f2_pred, self.fi1_pred,\
            self.uavgi1_pred,\
            self.u1i1_pred, self.u2i1_pred \
            = self.net_f(self.x_f1_tf, self.t_f1_tf, self.x_f2_tf, self.t_f2_tf,
                   self.x_i1_tf, self.t_i1_tf)

        self.loss1 = 20 * tf.reduce_mean(tf.square(self.u1_0_pred - self.u0)) \
                     + tf.reduce_mean(tf.square(self.u1_lb_pred - self.u1_ub_pred)) \
                     + tf.reduce_mean(tf.square(self.u1_lb_x_pred - self.u1_ub_x_pred)) \
                     + tf.reduce_mean(tf.square(self.f1_pred)) \
                     + 20 * tf.reduce_mean(tf.square(self.fi1_pred)) \
                     + 20 * tf.reduce_mean(tf.square(self.u1i1_pred - self.uavgi1_pred))

        self.loss2 = tf.reduce_mean(tf.square(self.u2_lb_pred - self.u2_ub_pred)) \
                     + tf.reduce_mean(tf.square(self.u2_lb_x_pred - self.u2_ub_x_pred)) \
                     + tf.reduce_mean(tf.square(self.f2_pred)) \
                     + 20 * tf.reduce_mean(tf.square(self.fi1_pred)) \
                     + 20 * tf.reduce_mean(tf.square(self.u2i1_pred - self.uavgi1_pred))

        self.loss = self.loss1 + self.loss2
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(0.0008)
        self.train_op_Adam1 = self.optimizer_Adam.minimize(self.loss1)
        self.train_op_Adam2 = self.optimizer_Adam.minimize(self.loss2)

        init = tf.global_variables_initializer()
        self.sess.run(init)


    def initialize_NN(self, layers):
        weights = []
        biases = []
        A = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            a = tf.Variable(0.05, dtype=tf.float64)
            weights.append(W)
            biases.append(b)
            A.append(a)

        return weights, biases, A

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.to_double(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)), dtype=tf.float64)

    def neural_net_tanh(self, X, weights, biases, A):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(20 * A[l] * tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_sin(self, X, weights, biases, A):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(20 * A[l] * tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_cos(self, X, weights, biases, A):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.cos(20 * A[l] * tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u1(self, x, t):
        u = self.neural_net_tanh(tf.concat([x, t], 1), self.weights1, self.biases1, self.A1)
        return u

    def net_u2(self, x, t):
        u = self.neural_net_sin(tf.concat([x, t], 1), self.weights2, self.biases2, self.A2)
        return u

    def net_u1_uv(self, x, t):
        uv = self.net_u1(x, t)
        u = uv[:, 0:1]
        u_x = tf.gradients(u, x)[0]
        return u, u_x

    def net_u2_uv(self, x, t):
        uv = self.net_u2(x, t)
        u = uv[:, 0:1]
        u_x = tf.gradients(u, x)[0]
        return u, u_x

    def net_f(self, x1, t1, x2, t2, xi1, ti1):
        # Sub-Net1
        u1 = self.net_u1(x1, t1)
        u1_x = tf.gradients(u1, x1)[0]
        u1_t = tf.gradients(u1, t1)[0]
        u1_xx = tf.gradients(u1_x, x1)[0]

        # Sub-Net2
        u2 = self.net_u2(x2, t2)
        u2_x = tf.gradients(u2, x2)[0]
        u2_t = tf.gradients(u2, t2)[0]
        u2_xx = tf.gradients(u2_x, x2)[0]

        # Sub-Net1, Interface 1
        u1i1 = self.net_u1(xi1, ti1)
        u1i1_x = tf.gradients(u1i1, xi1)[0]
        u1i1_t = tf.gradients(u1i1, ti1)[0]
        u1i1_xx = tf.gradients(u1i1_x, xi1)[0]

        # Sub-Net2, Interface 1
        u2i1 = self.net_u2(xi1, ti1)
        u2i1_x = tf.gradients(u2i1, xi1)[0]
        u2i1_t = tf.gradients(u2i1, ti1)[0]
        u2i1_xx = tf.gradients(u2i1_x, xi1)[0]

        # Average value
        uavgi1 = (u1i1 + u2i1) / 2

        # Residuals
        f1 = u1_t - 0.0001*u1_xx + 5*u1*(u1-1)*(u1+1)
        f2 = u2_t - 0.0001*u2_xx + 5*u2*(u2-1)*(u2+1)

        fi1 = (u1i1_t - 0.0001*u1i1_xx + 5*u1i1*(u1i1-1)*(u1i1+1)) - (u2i1_t - 0.0001*u2i1_xx + 5*u2i1*(u2i1-1)*(u2i1+1))

        return f1, f2, fi1, uavgi1, u1i1, u2i1

    def callback(self, loss):
        print('Iter:', self.i, 'Loss:', loss)
        self.i += 1

    def train(self, nIter, X_star1, X_star2, u_exact1, u_exact2):
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.x_lb1_tf: self.x_lb1, self.t_lb1_tf: self.t_lb1,
                   self.x_ub1_tf: self.x_ub1, self.t_ub1_tf: self.t_ub1,
                   self.x_lb2_tf: self.x_lb2, self.t_lb2_tf: self.t_lb2,
                   self.x_ub2_tf: self.x_ub2, self.t_ub2_tf: self.t_ub2,
                   self.x_f1_tf: self.x_f1, self.t_f1_tf: self.t_f1,
                   self.x_f2_tf: self.x_f2, self.t_f2_tf: self.t_f2,
                   self.x_i1_tf: self.x_i1, self.t_i1_tf: self.t_i1}

        MSE_history1 = []
        MSE_history2 = []

        l2_err1 = []
        l2_err2 = []

        for it in range(nIter):
            self.sess.run(self.train_op_Adam1, tf_dict)
            self.sess.run(self.train_op_Adam2, tf_dict)

            if it % 10 == 0:
                # elapsed = time.time() - start_time
                loss1_value = self.sess.run(self.loss1, tf_dict)
                loss2_value = self.sess.run(self.loss2, tf_dict)

                # Predicted solution
                u_pred1, u_pred2 = model.predict(X_star1, X_star2)

                l2_error1 = np.linalg.norm(u_exact1 - u_pred1, 2) / np.linalg.norm(u_exact1, 2)
                l2_error2 = np.linalg.norm(u_exact2 - u_pred2, 2) / np.linalg.norm(u_exact2, 2)

                print('It: %d, Loss1: %.3e, Loss2: %.3e, L2_err1: %.3e, L2_err2: %.3e' %
                      (it, loss1_value, loss2_value, l2_error1, l2_error2))

                MSE_history1.append(loss1_value)
                MSE_history2.append(loss2_value)

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

        return MSE_history1, MSE_history2, l2_err1, l2_err2

    def predict(self, X_star1, X_star2):

        u_star1 = self.sess.run(self.ub1_pred, {self.x_f1_tf: X_star1[:, 0:1], self.t_f1_tf: X_star1[:, 1:2]})
        u_star2 = self.sess.run(self.ub2_pred, {self.x_f2_tf: X_star2[:, 0:1], self.t_f2_tf: X_star2[:, 1:2]})

        return u_star1, u_star2

if __name__ == '__main__':
    N_0 = 200
    N_b1 = 50
    N_b2 = 50

    # Residual points in three subdomains
    N_f1 = 5000
    N_f2 = 5000

    # Interface points along the two interfaces
    N_I1 = 256

    # NN architecture in each subdomain
    layers1 = [2, 20, 20, 20, 20, 20, 1]
    layers2 = [2, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('../Data/AC.mat')

    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['uu']).T
    X, T = np.meshgrid(x, t)

    X_0 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    u0 = Exact[0:1, :].T
    idx = np.random.choice(X_0.shape[0], N_0, replace=False)
    X_0 = X_0[idx, :]
    u0 = u0[idx, :]

    X_star1 = np.hstack((X[0:101, :].flatten()[:, None], T[0:101, :].flatten()[:, None]))
    u_exact1 = Exact[0:101, :].flatten()[:, None]

    X_star2 = np.hstack((X[100:201, :].flatten()[:, None], T[100:201, :].flatten()[:, None]))
    u_exact2 = Exact[100:201, :].flatten()[:, None]

    X_lb1 = np.hstack((X[0: 101, 0:1], T[0: 101, 0:1]))
    X_ub1 = np.hstack((X[0: 101, -1:], T[0: 101, -1:]))
    idx = np.random.choice(X_lb1.shape[0], int(N_b1 / 2), replace=False)
    X_lb1 = X_lb1[idx, :]
    X_ub1 = X_ub1[idx, :]
    X_lb2 = np.hstack((X[100:201, 0:1], T[100: 201, 0:1]))
    X_ub2 = np.hstack((X[100: 201, -1:], T[100: 201, -1:]))
    idx = np.random.choice(X_lb2.shape[0], int(N_b2 / 2), replace=False)
    X_lb2 = X_lb2[idx, :]
    X_ub2 = X_ub2[idx, :]

    lb1 = np.array([-1, 0])
    ub1 = np.array([1, 0.5])
    lb2 = np.array([-1, 0.5])
    ub2 = np.array([1, 1])
    X_f1_train = lb1 + (ub1 - lb1) * lhs(2, N_f1)
    X_f2_train = lb2 + (ub2 - lb2) * lhs(2, N_f2)

    ti1 = np.zeros_like(x) + 0.5
    X_i1_train = np.hstack((x, ti1))
    idx = np.random.choice(X_i1_train.shape[0], N_I1, replace=False)
    X_i1_train = X_i1_train[idx, :]


    model = XPINN(X_0, u0, X_lb1, X_ub1, X_lb2, X_ub2, X_f1_train, X_f2_train, X_i1_train, layers1, layers2)

    Max_iter = 10000
    start_time = time.time()
    MSE_history1, MSE_history2, l2_err1, l2_err2 = model.train(Max_iter, X_star1, X_star2, u_exact1, u_exact2)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    # Solution prediction
    u_pred1, u_pred2 = model.predict(X_star1, X_star2)

    u_pred = np.vstack((u_pred1, u_pred2))
    u_exact = np.vstack((u_exact1, u_exact2))

    error_u = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    print("All domain Error:")
    print('Error u: %e' % (error_u))