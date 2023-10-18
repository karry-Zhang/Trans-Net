import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import sys
# sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
# from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, x1, layers, dt, lb, ub, q):

        self.lb = lb
        self.ub = ub

        self.x0 = x0
        self.x1 = x1

        self.u0 = u0

        self.layers = layers
        self.dt = dt  # dt = [0.8]
        self.q = max(q, 1)

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # Load IRK weights
        tmp = np.float32(np.loadtxt('../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2))
        self.IRK_weights = np.reshape(tmp[0:q ** 2 + q], (q + 1, q))
        self.IRK_times = tmp[q ** 2 + q:]

        Config = tf.ConfigProto(allow_soft_placement=True)
        Config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=Config)

        self.x0_tf = tf.placeholder(tf.float32, shape=(None, self.x0.shape[1]))
        self.x1_tf = tf.placeholder(tf.float32, shape=(None, self.x1.shape[1]))
        self.u0_tf = tf.placeholder(tf.float32, shape=(None, self.u0.shape[1]))
        self.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, self.q))  # dummy variable for fwd_gradients
        self.dummy_x1_tf = tf.placeholder(tf.float32, shape=(None, self.q + 1))  # dummy variable for fwd_gradients

        self.U0_pred = self.net_U0(self.x0_tf)  # N x (q+1)
        self.U1_pred, self.U1_x_pred = self.net_U1(self.x1_tf)  # N1 x (q+1)

        self.loss = tf.reduce_sum(tf.square(self.u0_tf - self.U0_pred)) + \
                    tf.reduce_sum(tf.square(self.U1_pred[0, :] - self.U1_pred[1, :])) + \
                    tf.reduce_sum(tf.square(self.U1_x_pred[0, :] - self.U1_x_pred[1, :]))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

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

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def fwd_gradients_0(self, U, x):
        g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]
        return tf.gradients(g, self.dummy_x0_tf)[0]

    def fwd_gradients_1(self, U, x):
        g = tf.gradients(U, x, grad_ys=self.dummy_x1_tf)[0]
        return tf.gradients(g, self.dummy_x1_tf)[0]

    def net_U0(self, x):
        U1 = self.neural_net(x, self.weights, self.biases)
        U = U1[:, :-1]
        U_x = self.fwd_gradients_0(U, x)
        U_xx = self.fwd_gradients_0(U_x, x)
        F = 5.0 * U - 5.0 * U ** 3 + 0.0001 * U_xx
        U0 = U1 - self.dt * tf.matmul(F, self.IRK_weights.T)
        return U0

    def net_U1(self, x):
        U1 = self.neural_net(x, self.weights, self.biases)
        U1_x = self.fwd_gradients_1(U1, x)
        return U1, U1_x  # N x (q+1)

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter):
        tf_dict = {self.x0_tf: self.x0, self.u0_tf: self.u0, self.x1_tf: self.x1,
                   self.dummy_x0_tf: np.ones((self.x0.shape[0], self.q)),
                   self.dummy_x1_tf: np.ones((self.x1.shape[0], self.q + 1))}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 1 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, x_star):

        U1_star = self.sess.run(self.U1_pred, {self.x1_tf: x_star})

        return U1_star

    def save_mode(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params/point.ckpt')

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params/point.ckpt')

def load():
    data = scipy.io.loadmat('../Data/AC.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['uu']).T
    return t, x, Exact

def main(subdomain, q, N, layers, t_idx, u_boundary = None):
    t, x, Exact = load()
    idx_t0 = t_idx[0]
    idx_t1 = t_idx[1]
    dt = t[idx_t1] - t[idx_t0]

    lb = np.array([-1.0])
    ub = np.array([1.0])

    if u_boundary is None:
        idx_x = np.random.choice(Exact.shape[1], N, replace=False)
        x0 = x[idx_x, :]
        u0 = Exact[idx_t0:idx_t0 + 1, idx_x].T
    else:
        idx_x = np.random.choice(Exact.shape[1], N, replace=False)
        x0 = x[idx_x, :]
        u0 = u_boundary[idx_x, :]

    # Boudanry data
    x1 = np.vstack((lb, ub))

    # Test data
    x_star = x
    """设置网络"""
    model = PhysicsInformedNN(x0, u0, x1, layers, dt, lb, ub, q)
    if subdomain != 1:
        model.load()
    model.train(10000)

    U1_pred = model.predict(x_star)

    error = np.linalg.norm(U1_pred[:, -1] - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
    print('Error: %e' % (error))

    model.save_mode()
    tf.keras.backend.clear_session()
    return U1_pred[:, -1].reshape(-1, 1)

if __name__ == "__main__":
    q = 100
    N = 200
    layers = [1, 200, 200, 200, 200, q + 1]

    t_idx = [20, 60]
    u_boundary = main(1, q, N, layers, t_idx)

    t_idx = [60, 100]
    u_boundary = main(2, q, N, layers, t_idx, u_boundary)

    t_idx = [100, 140]
    u_boundary = main(3, q, N, layers, t_idx, u_boundary)

    t_idx = [140, 180]
    u_boundary = main(4, q, N, layers, t_idx, u_boundary)