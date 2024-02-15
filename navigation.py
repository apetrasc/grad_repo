import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import torch
import gpytorch
import math
from matplotlib.lines import Line2D
from scipy.stats import norm
import GPy
import time
#以下はライブラリではなく、同じディレクトリのファイルからのインポート
from drive_course_2 import course 
from ai_engine import ExactGPModel, DataManager, kernel_simple
from measure import Measurement
from gp_utils import GPUtils
class CoordinateTransform:#座標変換のクラス
    def __init__(self, car_x=0, car_y=0, car_theta=0):
        self.car_x = car_x
        self.car_y = car_y
        self.car_theta = car_theta

    def to_car_frame(self, points_x, points_y):
        transform = Affine2D()
        transform.translate(-self.car_x, -self.car_y).rotate(-self.car_theta)
        transformed_points = transform.transform(np.column_stack((points_x, points_y)))
        return transformed_points[:, 0], transformed_points[:, 1]

    def to_global_frame(self, points):
        transform = Affine2D() #points input is torch, outcome is numpy
        points_x=points[:,0]
        points_y=points[:,1]
        transform.rotate(self.car_theta).translate(self.car_x, self.car_y)
        transformed_points = transform.transform(np.column_stack((points_x, points_y)))
        return torch.tensor(transformed_points)

    def update_car_position(self, car_x, car_y, car_theta):
        self.car_x = car_x
        self.car_y = car_y
        self.car_theta = car_theta       

def plot_course(ax, course, color, label):
    ax.plot(course[0,:,0], course[0,:,1], color, label=label)
    ax.plot(course[1,:,0], course[1,:,1], color) 
    #print("co",course[0,:,0].shape)
def plot_regression_function(transformer,ax, test,color,label):
    test_global_l= transformer.to_global_frame(test[0,:,:].T)
    test_global_r= transformer.to_global_frame(test[1,:,:].T)
    #print("rv",test[0,:,:].T.shape)
    time.sleep(2)
    print("gl",test_global_l.shape)
    ax.plot(test_global_l[:,0], test_global_l[:,1], color=color,label=label)
    ax.plot(test_global_r[:,0], test_global_r[:,1], color=color)


k_v = 0.01
v_0 = 0
ax1 = None
dt = 300
fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()
K_v = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
K = ['0', '0.1', '0.2', '0.4','0.6', '0.8', '1.0', '2.0']
seed = 42 
v = v_0 +1.0
cum_dev = 0
omega = 0 #初期の角速度
theta = -math.pi/4  # 注、弧度法
xdata, ydata = [-7], [7]  #初期値
square_size=0.8 #車を模した長方形のサイズ
prob_mask =0.053 #残るデータの割合
spec = 2.8 #Lidarセンサの測定可能距離
#transformer = CoordinateTransform(car_x=0, car_y=0, car_theta=0)
def init():
    line.set_data([0], [0])
    return line,
# アニメーションを更新する関数
def update(num,v, xdata, ydata, line):
    global theta,dt,seed,prob_mask,cum_dev,ax1,transformer,omega
    ax1.clear()
    ax1.set_xlim(-10, 15)
    ax1.set_ylim(-10, 15)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    manager = DataManager(seed, prob_mask)
    transformer = CoordinateTransform(car_x=xdata[-1], car_y=ydata[-1], car_theta=theta)
    gp_utils = GPUtils()
    plot_course(ax1, course, color='b', label='Course')
   
    train_l = manager.prepare_data(course[0,:,:],xdata[-1],ydata[-1],theta,max_distance=spec, min_angle=np.pi/8, max_angle=np.pi*7/8)
    train_r = manager.prepare_data(course[1,:,:],xdata[-1],ydata[-1],theta,max_distance=spec, min_angle=-np.pi*7/8, max_angle=-np.pi/8)
    train_l_global = transformer.to_global_frame(train_l)
    train_r_global = transformer.to_global_frame(train_r)
    ax1.plot(train_l_global[:, 0], train_l_global[:, 1], 'y*',label='train data (left)')
    ax1.plot(train_r_global[:, 0], train_r_global[:, 1], 'r*',label='train data (right)')
    
    model_l, likelihood_l = gp_utils.train_model(train_l[:,0], train_l[:,1]) #Gaussian Process による学習、推論を行う場所
    model_r, likelihood_r = gp_utils.train_model(train_r[:,0], train_r[:,1])
    gp_utils.set_eval_mode(model_l, likelihood_l, model_r, likelihood_r)
    test_x = torch.tensor([[v*dt*0.001]])
    mu_l, sigma_l = gp_utils.predict_with_model(model_l, likelihood_l, test_x)
    mu_r, sigma_r = gp_utils.predict_with_model(model_r, likelihood_r, test_x)
    
    v_x = v * np.cos(theta)
    v_y = v * np.sin(theta)
    next_x = xdata[-1] + v_x*dt*0.001
    next_y = ydata[-1] + v_y*dt*0.001
    xdata.append(next_x)
    ydata.append(next_y)
    line.set_data(xdata, ydata)
    for rect in ax1.patches:
        rect.remove()
    omega = (1/(dt*0.001))* math.atan((mu_l+mu_r)/(2*v*dt*0.001))
    print(omega)
    theta += omega*dt*0.001
    Affine2D().translate(-xdata[-1], -ydata[-1]).rotate(-theta) # 車中心の座標系に変換
    rot_trans = Affine2D().rotate_deg_around(next_x, next_y, theta+math.degrees(np.pi/4)) #長方形が車の正面に相当するように、表示を少しずらしている
    square = Rectangle((next_x - square_size/2, next_y - square_size/2), square_size, square_size, fill=False, edgecolor='blue')
    square.set_transform(rot_trans + ax1.transData)
    ax1.add_patch(square)
    
    
    ax1.plot(xdata, ydata, 'k', label='trajectory')
    #plot_regression_function(transformer,ax1, test,color='g',label='regression')
    ax1.legend()
    cum_dev += Measurement.cal_dev(v, dt, omega)
    #print("Average deviation from center:", cum_dev)

    return line,

line, = ax1.plot([0], [0], 'ro', markersize=1)

ani = animation.FuncAnimation(fig1, update, frames=range(100), fargs=[v, xdata, ydata, line], 
                              init_func=init, blit=False, repeat=False, interval=dt)

plt.show()