import numpy as np
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import math
from matplotlib.lines import Line2D

x_l_0 = np.linspace(-8, 2, 1000)
x_r_0 = np.linspace(-9, 1, 1000)
y_l_0 = -x_l_0 + 1
y_r_0 = -x_r_0 - 1
x_l_1 = np.linspace(2, 6, 1000)
x_r_1 = np.linspace(1, 7, 1000)
y_l_1 = -0.5*x_l_1
y_r_1 = -0.5*x_r_1 - 1.5
x_l_2 = np.linspace(6, 10, 1000)
x_r_2 = np.linspace(7, 11, 1000)
y_l_2 = x_l_2 - 9
y_r_2 = x_r_2 - 12
course_l_x = np.hstack((x_l_0,x_l_1,x_l_2))
course_r_x = np.hstack((x_r_0,x_r_1,x_r_2))
course_l_y = np.hstack((y_l_0,y_l_1,y_l_2))
course_r_y = np.hstack((y_r_0,y_r_1,y_r_2))


course_l = np.vstack((course_l_x, course_l_y)).T
course_r = np.vstack((course_r_x, course_r_y)).T

course = np.stack((course_l, course_r))


'''
print(course.shape)
print(course_l.shape)
print(course_l_reconstructed.shape)
'''