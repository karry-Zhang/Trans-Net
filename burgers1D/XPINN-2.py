import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import time


np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u1, u1, X_u2, u2, X_f1, X_f2, X_i1, layers1, layers2, lb1, ub1, lb2, ub2, nu):
        self.i = 1

        self.lb1 = lb1
        self.ub1 = ub1
        self.lb2 = lb2
        self.ub2 = ub2

        self.x_u1 = X_u1[:, 0:1]
        self.t_u1 = X_u1[:, 1:2]
        self.x_u2 = X_u2[:, 0:1]
        self.t_u2 = X_u2[:, 1:2]

        self.x_f1 = X_f1[:, 0:1]
        self.t_f1 = X_f1[:, 1:2]
        self.x_f2 = X_f2[:, 0:1]
        self.t_f2 = X_f2[:, 1:2]

        self.x_i1 = X_i1[:, 0:1]
        self.t_i1 = X_i1[:, 1:2]

        self.u1 = u1
        self.u2 = u2

        self.layers1 = layers1
        self.layers2 = layers2
        self.nu = nu

        # Initialize NNs
        self.weights1, self.biases1 = self.initialize_NN(layers1)
        self.weights2, self.biases2 = self.initialize_NN(layers2)

        Config = tf.ConfigProto(allow_soft_placement=True)
        Config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=Config)

        self.x_u1_tf = tf.placeholder(tf.float32, shape=[None, self.x_u1.shape[1]])
        self.t_u1_tf = tf.placeholder(tf.float32, shape=[None, self.t_u1.shape[1]])
        self.u1_tf = tf.placeholder(tf.float32, shape=[None, self.u1.shape[1]])

        self.x_u2_tf = tf.placeholder(tf.float32, shape=[None, self.x_u2.shape[1]])
        self.t_u2_tf = tf.placeholder(tf.float32, shape=[None, self.t_u2.shape[1]])
        self.u2_tf = tf.placeholder(tf.float32, shape=[None, self.u2.shape[1]])

        self.x_f1_tf = tf.placeholder(tf.float32, shape=[None, self.x_f1.shape[1]])
        self.t_f1_tf = tf.placeholder(tf.float32, shape=[None, self.t_f1.shape[1]])

        self.x_f2_tf = tf.placeholder(tf.float32, shape=[None, self.x_f2.shape[1]])
        self.t_f2_tf = tf.placeholder(tf.float32, shape=[None, self.t_f2.shape[1]])

        self.x_i1_tf = tf.placeholder(tf.float32, shape=[None, self.x_i1.shape[1]])
        self.t_i1_tf = tf.placeholder(tf.float32, shape=[None, self.t_i1.shape[1]])

        self.u_pred1 = self.net_u1(self.x_u1_tf, self.t_u1_tf)
        self.u_pred2 = self.net_u2(self.x_u2_tf, self.t_u2_tf)

        self.f_pred1, self.f_pred2, self.fi_pred1, self.uavgi1, self.u1i1_pred, self.u2i1_pred = \
            self.net_f(self.x_f1_tf, self.t_f1_tf,
                       self.x_f2_tf, self.t_f2_tf,
                       self.x_i1_tf, self.t_i1_tf)

        self.loss1 = 20 * tf.reduce_mean(tf.square(self.u1_tf - self.u_pred1)) + \
                     tf.reduce_mean(tf.square(self.f_pred1)) \
                     + 20 * tf.reduce_mean(tf.square(self.fi_pred1)) \
                     + 20 * tf.reduce_mean(tf.square(self.u1i1_pred - self.uavgi1))

        self.loss2 = 1 * tf.reduce_mean(tf.square(self.u2_tf - self.u_pred2)) + \
                     tf.reduce_mean(tf.square(self.f_pred2)) \
                     + 20 * tf.reduce_mean(tf.square(self.fi_pred1)) \
                     + 20 * tf.reduce_mean(tf.square(self.u2i1_pred - self.uavgi1))


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
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases, lb, ub):
        num_layers = len(weights) + 1

        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        # H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u1(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights1, self.biases1, self.lb1, self.ub1)
        return u

    def net_u2(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights2, self.biases2, self.lb2, self.ub2)
        return u

    def net_f(self, x1, t1, x2, t2, xi1, ti1):
        # sub1
        u1 = self.net_u1(x1, t1)
        u1_t1 = tf.gradients(u1, t1)[0]
        u1_x1 = tf.gradients(u1, x1)[0]
        u1_x1x1= tf.gradients(u1_x1, x1)[0]
        f1 = u1_t1 + u1 * u1_x1- self.nu * u1_x1x1

        # sub2
        u2 = self.net_u2(x2, t2)
        u2_t2 = tf.gradients(u2, t2)[0]
        u2_x2 = tf.gradients(u2, x2)[0]
        u2_x2x2 = tf.gradients(u2_x2, x2)[0]
        f2 = u2_t2 + u2 * u2_x2 - self.nu * u2_x2x2

        # sub1 i1
        u1i1 = self.net_u1(xi1, ti1)
        u1i1_ti1 = tf.gradients(u1i1, ti1)[0]
        u1i1_xi1 = tf.gradients(u1i1, xi1)[0]
        u1i1_xi1xi1 = tf.gradients(u1i1_xi1, xi1)[0]
        f1i1 = u1i1_ti1 + u1i1 * u1i1_xi1 - self.nu * u1i1_xi1xi1

        # sub2 i1
        u2i1 = self.net_u2(xi1, ti1)
        u2i1_ti1 = tf.gradients(u2i1, ti1)[0]
        u2i1_xi1 = tf.gradients(u2i1, xi1)[0]
        u2i1_xi1xi1 = tf.gradients(u2i1_xi1, xi1)[0]
        f2i1 = u2i1_ti1 + u2i1 * u2i1_xi1 - self.nu * u2i1_xi1xi1

        # ave
        uavgi1 = (u1i1 + u2i1) / 2

        # Residuals

        fi1 = f1i1 - f2i1


        return f1, f2, fi1, uavgi1, u1i1, u2i1

    def callback(self, loss):
        print('Iter:', self.i, 'Loss:', loss)
        self.i += 1

    def train(self, nIter):
        tf_dict = {self.x_u1_tf: self.x_u1, self.t_u1_tf: self.t_u1, self.u1_tf: self.u1,
                   self.x_u2_tf: self.x_u2, self.t_u2_tf: self.t_u2, self.u2_tf: self.u2,
                   self.x_i1_tf: self.x_i1, self.t_i1_tf: self.t_i1,
                   self.x_f1_tf: self.x_f1, self.t_f1_tf: self.t_f1,
                   self.x_f2_tf: self.x_f2, self.t_f2_tf: self.t_f2}

        for it in range(nIter):
            self.sess.run(self.train_op_Adam1, tf_dict)
            self.sess.run(self.train_op_Adam2, tf_dict)


            if it % 1 == 0:
                loss1_value = self.sess.run(self.loss1, tf_dict)
                loss2_value = self.sess.run(self.loss2, tf_dict)
                print('Iter:', it, 'Loss1:', loss1_value, 'Loss2:', loss2_value)


        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        u_star = self.sess.run(self.u_pred1, {self.x_u1_tf: X_star[:, 0:1], self.t_u1_tf: X_star[:, 1:2]})

        return u_star

if __name__ == "__main__":
    nu = 0.01 / np.pi

    N_u1 = 100
    N_u2 = 100
    N_f1 = 5000
    N_f2 = 5000

    N_i1 = 256

    layers1 = [2, 20, 20, 20, 20, 20, 1]
    layers2 = [2, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)

    X_star1 = np.hstack((X[0:51, :].flatten()[:, None], T[0:51, :].flatten()[:, None]))
    u_star1 = Exact[0:51, :].flatten()[:, None]
    X_star2 = np.hstack((X[50:100, :].flatten()[:, None], T[50:100, :].flatten()[:, None]))
    u_star2 = Exact[50:100, :].flatten()[:, None]

    lb1 = X_star1.min(0)
    ub1 = X_star1.max(0)
    lb2 = X_star2.min(0)
    ub2 = X_star2.max(0)

    xx0_1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu0_1 = Exact[0:1, :].T
    xxlb_1 = np.hstack((X[0:51, 0:1], T[0:51, 0:1]))
    uulb_1 = Exact[0:51, 0:1]
    xxub_1 = np.hstack((X[0:51, -1:], T[0:51, -1:]))
    uuub_1 = Exact[0:51, -1:]

    xxlb_2 = np.hstack((X[50:100, 0:1], T[50:100, 0:1]))
    uulb_2 = Exact[50:100, 0:1]
    xxub_2 = np.hstack((X[50:100, -1:], T[50:100, -1:]))
    uuub_2 = Exact[50:100, -1:]

    X_u_train1 = np.vstack([xx0_1, xxlb_1, xxub_1])
    X_f_train1 = lb1 + (ub1 - lb1) * lhs(2, N_f1)
    X_f_train1 = np.vstack((X_f_train1, X_u_train1))
    u_train1 = np.vstack([uu0_1, uulb_1, uuub_1])

    X_u_train2 = np.vstack([xxlb_2, xxub_2])
    X_f_train2 = lb2 + (ub2 - lb2) * lhs(2, N_f2)
    X_f_train2 = np.vstack((X_f_train2, X_u_train2))
    u_train2 = np.vstack([uulb_2, uuub_2])

    idx = np.random.choice(X_u_train1.shape[0], N_u1, replace=False)
    X_u_train1 = X_u_train1[idx, :]
    u_train1 = u_train1[idx, :]

    idx = np.random.choice(X_u_train2.shape[0], N_u2, replace=False)
    X_u_train2 = X_u_train2[idx, :]
    u_train2 = u_train2[idx, :]

    ti1 = np.zeros_like(x) + 0.5
    X_i1_train = np.hstack((x, ti1))
    idx = np.random.choice(X_i1_train.shape[0], N_i1, replace=False)
    X_i1_train = X_i1_train[idx, :]

    model = PhysicsInformedNN(X_u_train1, u_train1, X_u_train2, u_train2, X_f_train1, X_f_train2, X_i1_train,
                              layers1, layers2, lb1, ub1, lb2, ub2, nu)

    Max_iter = 10000
    start_time = time.time()
    model.train(Max_iter)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_pred1 = model.predict(X_star1)
    error_u1 = np.linalg.norm(u_star1 - u_pred1, 2) / np.linalg.norm(u_star1, 2)
    print('Error u1: %e' % (error_u1))

    u_pred2 = model.predict(X_star2)
    error_u2 = np.linalg.norm(u_star2 - u_pred2, 2) / np.linalg.norm(u_star2, 2)
    print('Error u2: %e' % (error_u2))

    # all domain
    u_pred = np.vstack((u_pred1, u_pred2))
    u_exact = np.vstack((u_star1, u_star2))
    error_u = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    print("All domain Error:")
    print('Error u: %e' % (error_u))

