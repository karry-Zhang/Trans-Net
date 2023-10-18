import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from scipy.interpolate import griddata
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

torch.cuda.empty_cache()

seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

use_gpu = torch.cuda.is_available()

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a


class Model:
    def __init__(self, net, net_transform_fun, x_label, x_labels, x_label_w, x_f, x_f_loss_fun
                 ):
        self.optimizer_LBGFS = None
        self.net = net
        self.net_transform_fun = net_transform_fun
        self.x_label = x_label
        self.x_labels = x_labels
        self.x_label_w = x_label_w

        self.x_f = x_f
        self.x_f_loss_fun = x_f_loss_fun

        self.x_label_loss_collect = []
        self.x_f_loss_collect = []

    def train_U(self, x):
        if self.net_transform_fun is None:
            return self.net(x)
        else:
            return self.net_transform_fun(x, self.net(x))

    def predict_U(self, x):
        return self.train_U(x)

    # computer backward loss
    def epoch_loss(self):
        loss_equation = torch.mean(self.x_f_loss_fun(self.x_f, self.train_U) ** 2)
        self.x_f_loss_collect.append([self.net.iter, loss_equation.item()])
        loss = loss_equation

        loss_label = torch.mean((self.train_U(self.x_label) - self.x_labels) ** 2)
        loss += self.x_label_w * loss_label
        self.x_label_loss_collect.append([self.net.iter, loss_label.item()])

        loss.backward()
        self.net.iter += 1
        return loss

    # computer backward loss
    def LBGFS_epoch_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_equation = torch.mean(self.x_f_loss_fun(self.x_f, self.train_U) ** 2)
        self.x_f_loss_collect.append([self.net.iter, loss_equation.item()])
        loss = loss_equation

        loss_label = torch.mean((self.train_U(self.x_label) - self.x_labels) ** 2)
        loss += self.x_label_w * loss_label
        self.x_label_loss_collect.append([self.net.iter, loss_label.item()])

        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item())
        return loss

    def train(self, epochs):
        optimizer_adam = torch.optim.Adam(self.net.parameters())
        # self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(),
        #                                    lr=0.1,
        #                                    max_iter=20000,
        #                                    max_eval=None,
        #                                    history_si ze=100,
        #                                    tolerance_grad=1e-5,
        #                                    tolerance_change=1.0 * np.finfo(float).eps,
        #                                    line_search_fn="strong_wolfe")  # can be "strong_wolfe"

        # start_time = time.time()
        # self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)
        # print('LBGFS done!')

        start_time = time.time()
        for it in range(epochs):
            optimizer_adam.zero_grad()
            loss = self.epoch_loss()
            optimizer_adam.step()
            print('Iter:', self.net.iter, 'Loss:', loss.item())
        print('Adam done!')

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)


def x_f_loss_fun(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    u = train_U(x)
    d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    u_t = d[0][:, 0].unsqueeze(-1)
    u_x = d[0][:, 1].unsqueeze(-1)
    u_y = d[0][:, 2].unsqueeze(-1)

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [1]]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [2]]

    f = u_t - 0.25 ** 2 * (u_xx + u_yy) + (u ** 3 - u)
    return f

def initial_boundary(x, y):
    u0 = np.zeros((x.shape[0], x.shape[1]))
    for i in range(u0.shape[0]):
        for j in range(u0.shape[1]):
            if (x[i, j] - np.pi + 1) ** 2 + (y[i, j] - np.pi) ** 2 <= 1 \
                    or (x[i, j] - np.pi - 1) ** 2 + (y[i, j] - np.pi) ** 2 <= 1:
                u0[i, j] = 1
            else:
                u0[i, j] = -1
    return u0

def plot(xy, u0, x_meshgrid, y_meshgrid, lb, ub, title):
    """
    :param xy.shape: N x 2
    :param u0.shape: N x 1
    :param x_meshgrid.shape: N x N
    :param y_meshgrid.shape: N x N
    """
    u_data_pred = griddata(xy, u0, (x_meshgrid, y_meshgrid), method='cubic')

    fig = plt.figure()
    ax = plt.subplot(111)
    h = ax.imshow(u_data_pred, interpolation='nearest', cmap='seismic',
                  extent=[lb[0], ub[0], lb[1], ub[1]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.axis('square')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))  # MultipleLocator : interval
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_title('t=%.2f' % title, fontsize=10)
    plt.tight_layout()
    # plt.savefig('./figure/AC2_' + 'title' + '.eps')
    # plt.show()


def main(lb, ub, N, dt, layers, t_test_index, L):
    t = np.linspace(lb[2], ub[2], int(float(ub[2] - lb[2]) / dt + 1))
    x = np.linspace(lb[0], ub[0], N)[:, None]
    y = np.linspace(lb[1], ub[1], N)[:, None]

    # data
    x_star, y_star = np.meshgrid(x, y)
    u0 = initial_boundary(x_star, y_star).flatten()[:, None]
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    xy0 = np.hstack((x_star, y_star))

    x_test = torch.from_numpy(xy0).float()
    t_test = torch.from_numpy(t).float()

    x_init = torch.from_numpy(xy0).float()
    x_initial = torch.cat((torch.zeros(x_init.shape[0], 1), x_init), dim=1)
    x_initial_label = torch.from_numpy(u0).float()

    x_label = x_initial
    x_labels = x_initial_label

    # residual points
    x_f = torch.cat((torch.full_like(torch.zeros(x_init.shape[0], 1), t[0]), x_init), dim=1)
    for i in range(t.shape[0] - 1):
        x_f_temp = torch.cat((torch.full_like(torch.zeros(x_init.shape[0], 1), t[i + 1]), x_init), dim=1)
        x_f = torch.cat((x_f, x_f_temp), dim=0)
    net = Net(layers)

    if use_gpu:
        net.cuda()
        x_label = x_label.cuda()
        x_labels = x_labels.cuda()
        x_f = x_f.cuda()
        x_test = x_test.cuda()
        t_test = t_test.cuda()

    model = Model(
        net=net,
        net_transform_fun=lambda x, u: u,
        x_label=x_label,
        x_labels=x_labels,
        x_label_w=1,
        x_f=x_f,
        x_f_loss_fun=x_f_loss_fun
    )

    model.train(20000)

    nn = 200
    x_plot = np.linspace(lb[0], ub[0], nn)
    y_plot = np.linspace(lb[1], ub[1], nn)
    x_plot, y_plot = np.meshgrid(x_plot, y_plot)

    for item in t_test_index:
        txy_pred = torch.cat((torch.full_like(torch.zeros(x_test.shape[0], 1), t_test[item]).cuda(), x_test), dim=1)
        U_pred = model.predict_U(txy_pred).cpu().detach().numpy()
        plot(xy0, U_pred, x_plot, y_plot, lb, ub, t[item])

if __name__ == '__main__':
    N = 81
    dt = 0.1
    L = 6
    layers = [3] + L * [50] + [1]

    lb = np.array([0, 0, 0])
    ub = np.array([6, 6, 12])

    t_test_index = [0, 1, 10, 20, 30, 40, 50, 70, 100, 120]
    main(lb, ub, N, dt, layers, t_test_index, L)












