import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        self.i = 1
        self.loss_data = []
        self.ite = []

        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u = u

        self.layers = layers
        self.nu = nu

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        Config = tf.ConfigProto(allow_soft_placement=True)
        Config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=Config)

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

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

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u * u_x - self.nu * u_xx

        return f

    def callback(self, loss):
        print('Loss:', loss)
        self.ite.append(self.i)
        self.loss_data.append(loss)
        self.i += 1

    def train(self):

        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star

    def save_mode(self):
        print("################ save #####################")
        saver = tf.train.Saver()
        saver.save(self.sess, './params/point.ckpt')

    def load(self):
        print("################ load #####################")
        saver = tf.train.Saver()
        saver.restore(self.sess, './params/point.ckpt')

    def save_loss(self, subdomain):
        np.savetxt('./loss_data/' + str(subdomain) + '/ite.txt', self.ite)
        np.savetxt('./loss_data/' + str(subdomain) + '/loss.txt', self.loss_data)


def load_data():
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    return t, x, Exact


def get_boundary(X, T, Exact, t_idx, xt_boundary, u_boundary, N_0, N_b):
    if xt_boundary is None:
        xx1 = np.hstack((X[t_idx[0]:t_idx[0] + 1, :].T, T[t_idx[0]:t_idx[0] + 1, :].T))
        uu1 = Exact[t_idx[0]:t_idx[0] + 1, :].T
    else:
        xx1 = xt_boundary
        uu1 = u_boundary

    xx2 = np.hstack((X[t_idx[0]: t_idx[-1] + 1, 0:1], T[t_idx[0]: t_idx[-1] + 1, 0:1]))
    uu2 = Exact[t_idx[0]: t_idx[-1] + 1, 0:1]

    xx3 = np.hstack((X[t_idx[0]: t_idx[-1] + 1, -1:], T[t_idx[0]: t_idx[-1] + 1, -1:]))
    uu3 = Exact[t_idx[0]: t_idx[-1] + 1, -1:]

    idx = np.random.choice(xx1.shape[0], N_0, replace=False)
    xx1 = xx1[idx, :]
    uu1 = uu1[idx, :]

    idx = np.random.choice(xx2.shape[0], int(N_b / 2), replace=False)
    xx2 = xx2[idx, :]
    uu2 = uu2[idx, :]

    idx = np.random.choice(xx3.shape[0], int(N_b / 2), replace=False)
    xx3 = xx3[idx, :]
    uu3 = uu3[idx, :]

    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])

    return X_u_train, u_train


def main(subdomain, N_0, N_b, N_f, layers, t_idx, xt_boundary=None, u_boundary=None):
    nu = 0.01 / np.pi
    t, x, Exact = load_data()

    X, T = np.meshgrid(x, t)

    X_star = np.hstack(
        (X[t_idx[0]:t_idx[-1] + 1, :].flatten()[:, None], T[t_idx[0]:t_idx[-1] + 1, :].flatten()[:, None]))
    u_star = Exact[t_idx[0]:t_idx[-1] + 1, :].flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    X_u_train, u_train = get_boundary(X, T, Exact, t_idx, xt_boundary, u_boundary, N_0, N_b)

    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))

    end_t = t[-1][0]
    t_boundary = np.zeros((x.shape[0], 1)) + end_t
    xt_boundary = np.hstack((x, t_boundary))
    X_f_train = np.vstack((X_f_train, xt_boundary))

    lb = np.array([-1, 0])
    ub = np.array([1, 0.99])
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)

    if subdomain != 1:
        model.load()
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    # 预测
    if subdomain != 1:
        X_star = np.hstack(
            (X[t_idx[0] + 1:t_idx[-1] + 1, :].flatten()[:, None], T[t_idx[0] + 1:t_idx[-1] + 1, :].flatten()[:, None]))
        u_star = Exact[t_idx[0] + 1:t_idx[-1] + 1, :].flatten()[:, None]
    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))

    end_t = t[t_idx[-1]][0]
    t_boundary = np.zeros((x.shape[0], 1)) + end_t
    xt_boundary = np.hstack((x, t_boundary))
    u_boundary, f_boundary = model.predict(xt_boundary)

    model.save_loss(subdomain)

    model.save_mode()
    tf.keras.backend.clear_session()

    return xt_boundary, u_boundary, u_pred, u_star


if __name__ == "__main__":
    layers = [2, 20, 20, 20, 20, 20, 1]

    N_0 = 90
    N_b = 10
    N_f = 5000
    t_idx = [0, 50]
    xt_boundary, u_boundary, u1_pred, u1_star = main(1, N_0, N_b, N_f, layers, t_idx)

    N_0 = 256
    t_idx = [50, 99]
    xt_boundary, u_boundary, u2_pred, u2_star = main(2, N_0, N_b, N_f, layers, t_idx, xt_boundary, u_boundary)

    u_pred = np.vstack((u1_pred, u2_pred))
    u_star = np.vstack((u1_star, u2_star))
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print("All domain Error:")
    print('Error u: %e' % (error_u))