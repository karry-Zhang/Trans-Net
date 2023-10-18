import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X0, u0, tb, X_f, layers, lb, ub):
        self.i = 1
        self.loss_data = []
        self.ite = []

          # (x0, 0)
        X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]

        self.x_lb = X_lb[:, 0:1]
        self.t_lb = X_lb[:, 1:2]

        self.x_ub = X_ub[:, 0:1]
        self.t_ub = X_ub[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u0 = u0
        self.layers = layers
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        # tf placeholders and graph
        Config = tf.ConfigProto(allow_soft_placement=True)
        Config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=Config)

        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])

        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])

        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u0_pred, _ = self.net_u(self.x0_tf, self.t0_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        self.u_lb_pred, self.u_x_lb_pred = self.net_u(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.u_x_ub_pred = self.net_u(self.x_ub_tf, self.t_ub_tf)

        self.loss = tf.reduce_mean(tf.square(self.f_pred)) + tf.reduce_mean(
            tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred))

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
        xavier_stddev = (np.random.randn(in_dim, out_dim)) / np.sqrt(in_dim)
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
        u_x = tf.gradients(u, x)[0]
        return u, u_x

    def net_f(self, x, t):
        u, u_x = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]

        f = u_t - 0.0001 * u_xx + 5 * u * u * u - 5 * u
        return f

    def callback(self, loss):
        print('Iter:', self.i, 'Loss:', loss)
        self.ite.append(self.i)
        self.loss_data.append(loss)
        self.i += 1

    def train(self):
        tf_dict = {self.x0_tf: self.x0,
                   self.t0_tf: self.t0,
                   self.x_lb_tf: self.x_lb,
                   self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub,
                   self.t_ub_tf: self.t_ub,
                   self.u0_tf: self.u0,
                   self.x_f_tf: self.x_f,
                   self.t_f_tf: self.t_f}
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X, T):
        tf_dict = {self.x0_tf: X.flatten()[:, None], self.t0_tf: T.flatten()[:, None]}
        u_star = self.sess.run(self.u0_pred, tf_dict)
        return u_star

    def save_loss(self):
        np.savetxt('./loss_data/PINNs/ite.txt', self.ite)
        np.savetxt('./loss_data/PINNs/loss.txt', self.loss_data)

def plot(X_star, Exact, u_predict, x, t, X, T, time):
    fig, ax = newfig(1.0, 1.1)
    ax.axis('on')
    U_pred = griddata(X_star, u_predict.flatten(), (X, T), method='cubic')
    ax.plot(x, Exact[:, time], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[time, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('t = %.2f' % (t[time]), fontsize=10)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    # plt.savefig('./loss_data/PINNs/figure/AC1Dt=0.' + str(t[time]) + '.eps')
    plt.show()

if __name__ == "__main__":
    layers = [2, 20, 20, 20, 20, 20, 1]
    N_i = 200
    N_b = 50
    N_f = 10000
    data = scipy.io.loadmat('../Data/AC.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['uu'])
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.T.flatten()[:, None]

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    idx_x = np.random.choice(x.shape[0], N_i, replace=False)
    x0 = x[idx_x, :]
    x0 = np.concatenate((x0, 0 * x0), 1)
    u0 = Exact[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    X_f = lb + (ub - lb) * lhs(2, N_f)
    X_f = np.vstack((X_f, x0))

    model = PhysicsInformedNN(x0, u0, tb, X_f,
                              layers, lb, ub)

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_predict = model.predict(X_star[:, 0:1], X_star[:, 1:2])

    error_u = np.linalg.norm(u_star - u_predict, 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))

    model.save_loss()

    plot(X_star, Exact, u_predict, x, t, X, T, 30)
    plot(X_star, Exact, u_predict, x, t, X, T, 100)
    plot(X_star, Exact, u_predict, x, t, X, T, 150)
