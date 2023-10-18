import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib
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


loss_data = []
item_data = []
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
        loss_data.append(loss.item())
        item_data.append(self.net.iter)
        print('Iter:', self.net.iter, 'Loss:', loss.item())
        return loss

    def train(self):
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(),
                                           lr=0.1,
                                           max_iter=50000,
                                           max_eval=None,
                                           history_size=100,
                                           tolerance_grad=1e-5,
                                           tolerance_change=1.0 * np.finfo(float).eps,
                                           line_search_fn="strong_wolfe")  # can be "strong_wolfe"

        start_time = time.time()
        self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)
        print('LBGFS done!')

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

    x_train = x[:, 1:2]
    y_train = x[:, 2:3]

    r = torch.sqrt(x_train ** 2 + y_train ** 2)
    v_t = (1 / torch.cosh(r)) ** 2 * torch.tanh(r)
    v_tmax = 0.385
    a = -1 * (v_t * y_train) / (v_tmax * r)
    b = (v_t * x_train) / (v_tmax * r)

    f = u_t + a * u_x + b * u_y
    return f

def exact_solution(t, x, y):
    u0 = np.zeros((x.shape[0], x.shape[1]))
    v_tmax = 0.385
    for i in range(u0.shape[0]):
        for j in range(u0.shape[1]):
            r = np.sqrt(x[i, j] ** 2 + y[i, j] ** 2)
            v_t = (1 / np.cosh(r)) ** 2 * np.tanh(r)
            omega = v_t / (r * v_tmax)
            a1 = 0.5 * y[i, j] * np.cos(omega * t[i, j])
            a2 = 0.5 * x[i, j] * np.sin(omega * t[i, j])
            u0[i, j] = -1 * np.tanh(a1 - a2)
    return u0

def plot(xy, u0, x_meshgrid, y_meshgrid, lb, ub, title):
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
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_title('t=%.2f' % title, fontsize=10)
    plt.tight_layout()
    # plt.savefig(str(title) + '.eps')
    plt.show()

def plot_error(xy, u0, x_meshgrid, y_meshgrid, lb, ub, title):
    u_data_pred = griddata(xy, u0, (x_meshgrid, y_meshgrid), method='cubic')

    fig = plt.figure()
    ax = plt.subplot(111)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.035)
    h = ax.imshow(u_data_pred, interpolation='nearest', cmap='seismic',
                  extent=[lb[0], ub[0], lb[1], ub[1]],
                  origin='lower', aspect='auto', norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.axis('square')
    ax.xaxis.set_major_locator(MultipleLocator(1))  # MultipleLocator : interval
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_title('t=%.2f' % title, fontsize=10)
    plt.tight_layout()
    # plt.savefig('mix_error_' + str(title) + '.eps')
    plt.show()

def save_loss(loss_data, item_data):
    np.savetxt('ite.txt', item_data)
    np.savetxt('loss.txt', loss_data)


def main(lb, ub, N, dt, layers, t_test_index, initial_u0 = None):
    loss_data.clear()
    item_data.clear()

    t = np.linspace(lb[2], ub[2], int(float(ub[2] - lb[2]) / dt + 1))
    x = np.linspace(lb[0], ub[0], N)[:, None]
    y = np.linspace(lb[1], ub[1], N)[:, None]

    # data
    x_star, y_star = np.meshgrid(x, y)
    if initial_u0 == None:
        u0 = exact_solution(np.zeros([x_star.shape[0], x_star.shape[1]]), x_star, y_star).flatten()[:, None]
    else:
        u0 = None
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

    # test
    x_testError, y_testError, t_testError = np.meshgrid(x, y, t)
    txy_test = np.hstack((t_testError.flatten()[:, None], x_testError.flatten()[:, None], y_testError.flatten()[:, None]))
    txy_test = torch.from_numpy(txy_test).float()

    # residual points
    x_f = torch.cat((torch.full_like(torch.zeros(x_init.shape[0], 1), t[0]), x_init), dim=1)
    for i in range(t.shape[0] - 1):
        x_f_temp = torch.cat((torch.full_like(torch.zeros(x_init.shape[0], 1), t[i + 1]), x_init), dim=1)
        x_f = torch.cat((x_f, x_f_temp), dim=0)
    net = Net(layers)
    print(x_f.shape)

    if use_gpu:
        net.cuda()
        x_label = x_label.cuda()
        x_labels = x_labels.cuda()
        x_f = x_f.cuda()
        x_test = x_test.cuda()

    model = Model(
        net=net,
        net_transform_fun=lambda x, u: u,
        x_label=x_label,
        x_labels=x_labels,
        x_label_w=1,
        x_f=x_f,
        x_f_loss_fun=x_f_loss_fun
    )

    model.train()

    nn = 200
    x_plot = np.linspace(lb[0], ub[0], nn)
    y_plot = np.linspace(lb[1], ub[1], nn)
    x_plot, y_plot = np.meshgrid(x_plot, y_plot)

    for item in t_test_index:
        txy_pred = torch.cat((torch.full_like(torch.zeros(x_test.shape[0], 1), item).cuda(), x_test), dim=1)
        U_pred = model.predict_U(txy_pred).cpu().detach().numpy()
        U_exact = exact_solution(txy_pred.cpu().detach().numpy()[:, 0:1], txy_pred.cpu().detach().numpy()[:, 1:2],
                                 txy_pred.cpu().detach().numpy()[:, 2:3])
        plot(xy0, U_pred, x_plot, y_plot, lb, ub, item)
        plot_error(xy0, np.abs(U_exact - U_pred), x_plot, y_plot, lb, ub, item)

    # save loss
    # save_loss(loss_data, item_data)

if __name__ == '__main__':
    N = 120
    dt = 0.1
    lb = np.array([-4, -4, 0])
    ub = np.array([4, 4, 4])

    layers = [3, 20, 20, 20, 20, 20, 20, 1]

    t_test_index = [2, 4]
    main(lb, ub, N, dt, layers, t_test_index)