import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from torch import nn
from pyDOE import lhs
from torch.autograd import Variable

seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

use_gpu = torch.cuda.is_available()

torch.cuda.empty_cache()

training_time = []
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
    def __init__(self, net, net_transform_fun, x_label, u_label, v_label, w_label, x_label_w, x_f, x_f_loss_fun
                 ):
        self.optimizer_LBGFS = None
        self.net = net
        self.net_transform_fun = net_transform_fun
        self.x_label = x_label
        self.u_label = u_label
        self.v_label = v_label
        self.w_label = w_label
        self.x_label_w = x_label_w

        self.x_f = x_f
        self.x_f_loss_fun = x_f_loss_fun

        self.x_label_loss_collect = []
        self.x_f_loss_collect = []

    def train_U(self, x):
        if self.net_transform_fun is None:
            out = self.net(x)
            return out[:, 0:1], out[:, 1:2], out[:, 2:3]
        else:
            out = self.net_transform_fun(x, self.net(x))
            return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def predict_U(self, x):
        return self.train_U(x)

    # computer backward loss
    def LBGFS_epoch_loss(self):
        self.optimizer_LBGFS.zero_grad()
        f_u_pred, f_v_pred, f_w_pred = self.x_f_loss_fun(self.x_f, self.train_U)
        loss_equation = torch.mean(f_u_pred ** 2) + torch.mean(f_v_pred ** 2) + torch.mean(f_w_pred ** 2)

        self.x_f_loss_collect.append([self.net.iter, loss_equation.item()])
        loss = loss_equation

        u_pred, v_pred, w_pred = self.train_U(self.x_label)
        loss_label = torch.mean((u_pred - self.u_label) ** 2) + torch.mean((v_pred - self.v_label) ** 2) + torch.mean((w_pred - self.w_label) ** 2)

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
    u, v, w = train_U(x)

    epislon = 1

    u_xyzt = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    u_x = u_xyzt[0][:, 0].unsqueeze(-1)
    u_y = u_xyzt[0][:, 1].unsqueeze(-1)
    u_z = u_xyzt[0][:, 2].unsqueeze(-1)
    u_t = u_xyzt[0][:, 3].unsqueeze(-1)

    v_xyzt = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)
    v_x = v_xyzt[0][:, 0].unsqueeze(-1)
    v_y = v_xyzt[0][:, 1].unsqueeze(-1)
    v_z = v_xyzt[0][:, 2].unsqueeze(-1)
    v_t = v_xyzt[0][:, 3].unsqueeze(-1)

    w_xyzt = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)
    w_x = w_xyzt[0][:, 0].unsqueeze(-1)
    w_y = w_xyzt[0][:, 1].unsqueeze(-1)
    w_z = w_xyzt[0][:, 2].unsqueeze(-1)
    w_t = w_xyzt[0][:, 3].unsqueeze(-1)

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, [0]]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0][:, [0]]

    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [1]]
    v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, [1]]
    w_yy = torch.autograd.grad(w_y, x, grad_outputs=torch.ones_like(w_y), create_graph=True)[0][:, [1]]

    u_zz = torch.autograd.grad(u_z, x, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, [2]]
    v_zz = torch.autograd.grad(v_z, x, grad_outputs=torch.ones_like(v_z), create_graph=True)[0][:, [2]]
    w_zz = torch.autograd.grad(w_z, x, grad_outputs=torch.ones_like(w_z), create_graph=True)[0][:, [2]]

    f_u = u_t + u * u_x + v * u_y + w * u_z - epislon * (u_xx + u_yy + u_zz)
    f_v = v_t + u * v_x + v * v_y + w * v_z - epislon * (v_xx + v_yy + v_zz)
    f_w = w_t + u * w_x + v * w_y + w * w_z - epislon * (w_xx + w_yy + w_zz)

    return f_u, f_v, f_w


def draw_epoch_loss(model):
    x_label_loss_collect = np.array(model.x_label_loss_collect)
    x_f_loss_collect = np.array(model.x_f_loss_collect)
    plt.subplot(2, 1, 1)
    plt.yscale('log')
    plt.plot(x_label_loss_collect[:, 0], x_label_loss_collect[:, 1], 'b-', label='Label_loss')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Loss$')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.yscale('log')
    plt.plot(x_f_loss_collect[:, 0], x_f_loss_collect[:, 1], 'r-', label='PDE_loss')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Loss$')
    plt.legend()
    # plt.show()



def analyticalSolution(x,y,z,t):
    epislon = 1
    u = (-2) * epislon * ((1 + np.exp(-t)* np.cos(x)*np.sin(y)*np.sin(z))/(1 + x + np.exp(-t)* np.sin(x)*np.sin(y)*np.sin(z)))
    v = (-2) * epislon * ((    np.exp(-t)* np.sin(x)*np.cos(y)*np.sin(z))/(1 + x + np.exp(-t)* np.sin(x)*np.sin(y)*np.sin(z)))
    w = (-2) * epislon * ((    np.exp(-t)* np.sin(x)*np.sin(y)*np.cos(z))/(1 + x + np.exp(-t)* np.sin(x)*np.sin(y)*np.sin(z)))

    return u,v,w

def plot(X1, Y1, Z1, u_predict, t, title):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 60)
    ax.scatter(X1, Y1, Z1, c=u_predict, s=20, cmap=cm.jet)
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(u_predict)
    plt.colorbar(m)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('t=%.2f' % t, fontsize=10)
    # plt.savefig('/figure/Burgers3D_' + str(title) + '.eps')
    # plt.show()

def plot_error(X1, Y1, Z1, u_predict, t, title):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 60)
    heatmp = ax.scatter(X1, Y1, Z1, c=u_predict, s=20, cmap=cm.jet)
    m = cm.ScalarMappable(cmap=cm.jet)
    heatmp.set_clim(vmin=0, vmax=0.02)
    m.set_array(np.array([0, 0.02]))
    plt.colorbar(m)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('t=%.2f' % t, fontsize=10)
    # plt.savefig('/figure/error_Burgers3D_' + str(title) + '.eps')
    plt.show()

def initial_data(lb, ub, N_0):
    t0 = np.zeros((N_0, 1))
    x0 = np.random.uniform(lb[0], ub[0], N_0).reshape(N_0, 1)
    y0 = np.random.uniform(lb[1], ub[1], N_0).reshape(N_0, 1)
    z0 = np.random.uniform(lb[2], ub[2], N_0).reshape(N_0, 1)
    u0, v0, w0 = analyticalSolution(x0, y0, z0, t0)
    xyzt0_train = np.hstack((x0, y0, z0, t0))

    return u0, v0, w0, xyzt0_train

def boundary_data(lb, ub, N_b):
    t = np.linspace(lb[3], ub[3], N_b).reshape(N_b, 1)

    x1 = np.zeros((N_b, 1)) + lb[0]
    y1 = np.random.uniform(lb[1], ub[1], N_b).reshape(N_b, 1)
    z1 = np.random.uniform(lb[2], ub[2], N_b).reshape(N_b, 1)
    u1, v1, w1 = analyticalSolution(x1, y1, z1, t)
    v1 = np.zeros((N_b, 1))
    w1 = np.zeros((N_b, 1))
    x1_u_train = np.hstack((x1, y1, z1, t))

    x2 = np.zeros((N_b, 1)) + ub[0]
    y2 = np.random.uniform(lb[1], ub[1], N_b).reshape(N_b, 1)
    z2 = np.random.uniform(lb[2], ub[2], N_b).reshape(N_b, 1)
    u2, v2, w2 = analyticalSolution(x2, y2, z2, t)
    x2_u_train = np.hstack((x2, y2, z2, t))

    x3 = np.random.uniform(lb[0], ub[0], N_b).reshape(N_b, 1)
    y3 = np.zeros((N_b, 1)) + lb[1]
    z3 = np.random.uniform(lb[2], ub[2], N_b).reshape(N_b, 1)
    u3, v3, w3 = analyticalSolution(x3, y3, z3, t)
    w3 = np.zeros((N_b, 1))
    x3_u_train = np.hstack((x3, y3, z3, t))

    x4 = np.random.uniform(lb[0], ub[0], N_b).reshape(N_b, 1)
    y4 = np.zeros((N_b, 1)) + ub[1]
    z4 = np.random.uniform(lb[2], ub[2], N_b).reshape(N_b, 1)
    u4, v4, w4 = analyticalSolution(x4, y4, z4, t)
    x4_u_train = np.hstack((x4, y4, z4, t))

    x5 = np.random.uniform(lb[0], ub[0], N_b).reshape(N_b, 1)
    y5 = np.random.uniform(lb[1], ub[1], N_b).reshape(N_b, 1)
    z5 = np.zeros((N_b, 1)) + lb[2]
    u5, v5, w5 = analyticalSolution(x5, y5, z5, t)
    v5 = np.zeros((N_b, 1))
    x5_u_train = np.hstack((x5, y5, z5, t))

    x6 = np.random.uniform(lb[0], ub[0], N_b).reshape(N_b, 1)
    y6 = np.random.uniform(lb[1], ub[1], N_b).reshape(N_b, 1)
    z6 = np.zeros((N_b, 1)) + ub[2]
    u6, v6, w6 = analyticalSolution(x6, y6, z6, t)
    x6_u_train = np.hstack((x6, y6, z6, t))

    xyztb_train = np.vstack((x1_u_train, x2_u_train, x3_u_train, x4_u_train, x5_u_train, x6_u_train))
    u_b = np.vstack((u1, u2, u3, u4, u5, u6))
    v_b = np.vstack((v1, v2, v3, v4, v5, v6))
    w_b = np.vstack((w1, w2, w3, w4, w5, w6))

    return u_b, v_b, w_b, xyztb_train

def save_loss(loss_data, item_data, subdomain):
    np.savetxt('/subdomain' + str(subdomain) + '/loss/ite.txt', item_data)
    np.savetxt('/subdomain' + str(subdomain) + '/loss/loss.txt', loss_data)

def save_net_para(net):
    torch.save(net.state_dict(), 'params/model_parameter.pkl')

def load_net_para(net):
    net.load_state_dict(torch.load('params/model_parameter.pkl'))

def main(lb, ub, N, N_0, N_b, N_f, dt, layers, t_test_index, xyzt_pred, initial_u0, initial_v0, initial_w0, subdomain):
    loss_data.clear()
    item_data.clear()

    x = np.linspace(lb[0], ub[0], N)[:, None]
    y = np.linspace(lb[1], ub[1], N)[:, None]
    z = np.linspace(lb[2], ub[2], N)[:, None]
    t = np.linspace(lb[3], ub[3], int(float(ub[3] - lb[3]) / dt + 1))

    X, Y, Z, T = np.meshgrid(x, y, z, t)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None], T.flatten()[:, None]))

    if initial_u0 is None:
        u_0, v_0, w_0, xyzt0_point= initial_data(lb, ub, N_0)
    else:
        idx = np.random.choice(initial_u0.shape[0], N_0, replace=False)
        u_0 = initial_u0[idx, :]
        v_0 = initial_v0[idx, :]
        w_0 = initial_w0[idx, :]
        xyzt0_point = xyzt_pred[idx, :]

    u_b, v_b, w_b, xyztb_point = boundary_data(lb, ub, N_b)

    x_label = np.vstack((xyzt0_point, xyztb_point))
    u_label = np.vstack((u_0, u_b))
    v_label = np.vstack((v_0, v_b))
    w_label = np.vstack((w_0, w_b))

    x_f = lb + (ub - lb) * lhs(4, N_f)

    # change to tensor
    x_label = torch.from_numpy(x_label).float()
    u_label = torch.from_numpy(u_label).float()
    v_label = torch.from_numpy(v_label).float()
    w_label = torch.from_numpy(w_label).float()
    x_f = torch.from_numpy(x_f).float()

    net = Net(layers)
    if (subdomain != 1):
        load_net_para(net)

    if use_gpu:
        net.cuda()
        x_label = x_label.cuda()
        u_label = u_label.cuda()
        v_label = v_label.cuda()
        w_label = w_label.cuda()
        x_f = x_f.cuda()

    model = Model(
        net=net,
        net_transform_fun=lambda x, u: u,
        x_label=x_label,
        u_label=u_label,
        v_label=v_label,
        w_label=w_label,
        x_label_w=1,
        x_f=x_f,
        x_f_loss_fun=x_f_loss_fun
    )

    model.train()
    draw_epoch_loss(model)

    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    z = np.linspace(lb[2], ub[2], nn)
    X1, Y1 = np.meshgrid(x, y)
    Z1 = np.tile(z.T, (nn, 1))

    for item in t_test_index:
        T1 = np.zeros((X1.shape[0] * X1.shape[1], 1)) + item
        u_predict, v_predict, w_predict = analyticalSolution(X1.flatten()[:, None], Y1.flatten()[:, None],
                                                             Z1.flatten()[:, None], T1)
        u_predict = u_predict.reshape((nn, nn))
        plot(X1, Y1, Z1, u_predict, T1[0], str(item) + 'u_analy')

        xyzt_pred = np.hstack(
            (X1.flatten()[:, None], Y1.flatten()[:, None], Z1.flatten()[:, None], T1.flatten()[:, None]))
        u_exact, v_exact, w_exact = analyticalSolution(xyzt_pred[:, 0:1], xyzt_pred[:, 1:2], xyzt_pred[:, 2:3], xyzt_pred[:, 3:4])
        xyzt_pred = torch.from_numpy(xyzt_pred).float().cuda()
        u_predict, v_predict, w_predict = model.predict_U(xyzt_pred)

        u_predict = u_predict.cpu().detach().numpy().reshape((nn, nn))
        plot(X1, Y1, Z1, u_predict, T1[0], str(item) + 'u_pred')
        plot_error(X1, Y1, Z1, np.abs(u_predict - u_exact.reshape((nn, nn))), T1[0], str(item) + 'u_pred')

    # save loss
    # save_loss(loss_data, item_data, subdomain)

    # saving network parameters
    save_net_para(net)

    # return interface value
    T1 = np.zeros((X1.shape[0] * X1.shape[1], 1)) + ub[3]
    xyzt_pred = np.hstack((X1.flatten()[:, None], Y1.flatten()[:, None], Z1.flatten()[:, None], T1.flatten()[:, None]))
    u_predict, v_predict, w_predict = model.predict_U(torch.from_numpy(xyzt_pred).float().cuda())
    u_predict = u_predict.cpu().detach().numpy()
    v_predict = v_predict.cpu().detach().numpy()
    w_predict = w_predict.cpu().detach().numpy()

    return xyzt_pred, u_predict, v_predict, w_predict



if __name__ == '__main__':
    N = 51
    dt = 0.001
    layers = [4, 50, 50, 50, 50, 50, 50, 3]

    # subdomain1
    N_0 = 121
    N_b = 100
    N_f = 50000
    lb = np.array([0, 0, 0, 0])
    ub = np.array([1, 1, 1, 0.05])
    t_test_index = []
    initial_u0 = None
    initial_v0 = None
    initial_w0 = None
    xyzt_pred = None
    xyzt_pred, u_predict, v_predict, w_predict = main(lb, ub, N, N_0, N_b, N_f, dt, layers, t_test_index, xyzt_pred, initial_u0, initial_v0, initial_w0, 1)

    # subdomain2
    N_0 = 121
    lb = np.array([0, 0, 0, 0.05])
    ub = np.array([1, 1, 1, 0.1])
    t_test_index = [0.1]
    xyzt_pred, u_predict, v_predict, w_predict = main(lb, ub, N, N_0, N_b, N_f, dt, layers, t_test_index, xyzt_pred, u_predict, v_predict, w_predict, 2)