import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import torch
import math
from matplotlib.lines import Line2D
from scipy.stats import norm
import time
import json
#以下はライブラリではなく、同じディレクトリのファイルからのインポート
from drive_course_2 import course 
from ai_engine import DataManager, CoordinateTransform
from measure import Measurement
from gp_utils import GPUtils

def plot_course(ax, course, color, label):
    ax.plot(course[0,:,0], course[0,:,1], color, label=label)
    ax.plot(course[1,:,0], course[1,:,1], color) 
    #print("co",course[0,:,0].shape)
def draw_vehicle_position(ax, next_x, next_y, theta, square_size=0.8):
    for rect in ax.patches: #次に書く車のために先の車を削除
        rect.remove()
    
    square = Rectangle((next_x - square_size/2, next_y - square_size/2), square_size, square_size, fill=False, edgecolor='blue')# 長方形（車）を描画
    rot_trans = Affine2D().rotate_deg_around(next_x, next_y, np.degrees(theta))
    square.set_transform(rot_trans + ax.transData)
    ax.add_patch(square)
def plot_regression_function(transformer,ax, test,color,label):
    test_global_l= transformer.to_global_frame(test[0,:,:].T)
    test_global_r= transformer.to_global_frame(test[1,:,:].T)
    ax.plot(test_global_l[:,0], test_global_l[:,1], color=color,label=label)
    ax.plot(test_global_r[:,0], test_global_r[:,1], color=color)

with open('init.json', 'r') as f:
    config = json.load(f)

ax1 = None
dt = 300
fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()
seed = config["seed"] 
v = config["v"]
cum_dev = config["cum_dev"]
omega = config["omega"] #初期の角速度
theta = config["theta"]  # 弧度法
xdata, ydata = config["xdata"], config["ydata"]  #初期値
square_size=config["square_size"] #車を模した長方形のサイズ
prob_mask =config["prob_mask"] #残るデータの割合
spec = config["spec"] #Lidarセンサの測定可能距離
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
    model_l, likelihood_l = gp_utils.train_model(train_l[:,0], train_l[:,1]) #Gaussian Process による学習
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
    omega = (1/(dt*0.001))* math.atan((mu_l+mu_r)/(2*v*dt*0.001))
    theta += omega*dt*0.001
    
    train_l_global = transformer.to_global_frame(train_l)
    train_r_global = transformer.to_global_frame(train_r)
    ax1.plot(train_l_global[:, 0], train_l_global[:, 1], 'y*',label='train data (left)')
    ax1.plot(train_r_global[:, 0], train_r_global[:, 1], 'r*',label='train data (right)')
    draw_vehicle_position(ax1, next_x, next_y, theta)
    ax1.plot(xdata, ydata, 'k-', label='trajectory')
    #plot_regression_function(transformer,ax1, test,color='g',label='regression')
    ax1.legend()
    cum_dev += Measurement.cal_dev(v, dt, omega)
    #print("Average deviation from center:", cum_dev)

    return line,

line, = ax1.plot([0], [0], 'ro', markersize=1)

ani = animation.FuncAnimation(fig1, update, frames=range(100), fargs=[v, xdata, ydata, line], 
                              init_func=init, blit=False, repeat=False, interval=dt)

plt.show()