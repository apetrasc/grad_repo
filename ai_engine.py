import numpy as np
import torch
import gpytorch
from matplotlib.transforms import Affine2D
from scipy.stats import norm
import GPy
class ExactGPModel(gpytorch.models.ExactGP):#ガウス過程のクラス
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)
class StudentTGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, train_x, variational_distribution, learn_inducing_locations=True)
        super(StudentTGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)
class StudentTLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self, deg_free, scale):
        super().__init__()
        self.deg_free = deg_free
        self.scale = scale

    def forward(self, function_samples, **kwargs):
        return torch.distributions.StudentT(df=self.deg_free, loc=function_samples, scale=self.scale)
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
   
class DataManager:
    def __init__(self, seed, prob_mask, noise_stddev=0.01):
        self.seed = seed
        self.prob_mask = prob_mask
        self.noise_stddev = noise_stddev
        
    def filter_data_based_on_distance_and_angle(self, course, car_x, car_y, car_angle, max_distance, min_angle, max_angle):
        x = course[:, 0]
        y = course[:, 1]
        data_x = torch.tensor(x, dtype=torch.float32)
        data_y = torch.tensor(y, dtype=torch.float32)
        distances = torch.sqrt((data_x - car_x)**2 + (data_y - car_y)**2)
        relative_angles = torch.atan2(data_y - car_y, data_x - car_x) - car_angle
        relative_angles = torch.atan2(torch.sin(relative_angles), torch.cos(relative_angles))
        mask = (distances <= max_distance)  & (min_angle <= relative_angles) & (relative_angles <= max_angle)
        filtered_data_x = data_x[mask]
        filtered_data_y = data_y[mask]

        return filtered_data_x, filtered_data_y
    
    def flip_sign_randomly(data, percentage=0.1):
        num_to_flip = int(percentage * len(data))  # 符合を反転させるデータの数
        indices_to_flip = np.random.choice(len(data), num_to_flip, replace=False)  # 反転させるインデックスをランダムに選択
        data[indices_to_flip] *= -1  # 符合を反転
        return data
    
    def icm_denoise(self,data, iterations=10):
        if not isinstance(iterations, int):
            raise ValueError("iterations must be an integer")
        for _ in range(iterations):
            for i in range(1, data.size(0) - 1):
                neighbors = data[i - 1] + data[i + 1]
                data[i] = 1 if neighbors > 0 else -1
            return data
    def bayesian_denoise(self, data, flip_probability=0.01):
        for i in range(data.size(0)):
            p_correct = (1 - flip_probability) if data[i] > 0 else flip_probability
            p_flipped = flip_probability if data[i] > 0 else (1 - flip_probability)
            data[i] = data[i] if p_correct > p_flipped else -data[i]
            return data
    def remove_lowp_data(self, model, data, threshold=0.05):
        model.eval()
        with torch.no_grad():
            prob = model(data)
        data[prob>threshold]
        return data
    def prepare_data(self, course, car_x, car_y, car_angle, max_distance=6, min_angle=0, max_angle=np.pi):
        percentage=0.1
        x = course[:, 0]
        y = course[:, 1]
        filtered_x_raw, filtered_y_raw = self.filter_data_based_on_distance_and_angle(course, car_x, car_y, car_angle, max_distance, min_angle, max_angle)
        transformer = CoordinateTransform(car_x, car_y, car_angle)
        filtered_x, filtered_y = transformer.to_car_frame(filtered_x_raw,filtered_y_raw)
        filtered_x = torch.tensor(filtered_x, dtype=torch.float32)
        filtered_y = torch.tensor(filtered_y, dtype=torch.float32)
        
        if not isinstance(filtered_x, torch.Tensor):
            raise TypeError('filtered_x must be a torch.Tensor')
        #print(filtered_y.size(), filtered_x.size())
        noisy_y = filtered_y + torch.randn(filtered_x.size()) * np.sqrt(0.001)
        train_x = self._mask_data_with_seed(filtered_x)
        train_y = self._mask_data_with_seed(noisy_y)
        #train_y[np.random.choice(train_y.size(0), int(percentage * train_y.size(0)), replace=False)] *=-1
        #train_y = self.remove_lowp_data(train_y)
        train =torch.stack((train_x, train_y), dim=1)
        return train

    def _mask_data_with_seed(self, data):
        torch.manual_seed(self.seed)
        mask = torch.rand(len(data)) < self.prob_mask
        return data[mask]


class MyModel:
    def __init__(self, model_l, model_r, likelihood_l, likelihood_r):
        self.model_l = model_l
        self.model_r = model_r
        self.likelihood_l = likelihood_l
        self.likelihood_r = likelihood_r

    def predict_mu_sigma(self, v, dt):
        input_tensor = torch.tensor([v*dt*0.001]).float()
        mu_l = self.likelihood_l(self.model_l(input_tensor)).mean[-1].item()
        sigma_l = self.likelihood_l(self.model_l(input_tensor)).variance[-1].item()
        mu_r = self.likelihood_r(self.model_r(input_tensor)).mean[-1].item()
        sigma_r = self.likelihood_r(self.model_r(input_tensor)).variance[-1].item()
        return (mu_l, sigma_l), (mu_r, sigma_r)

    '''
    model_l_GPy = GPy.models.TPRegression(train_l[:,0].numpy().reshape(-1, 1), train_l[:,1].numpy().reshape(-1, 1), kernel_simple, deg_free=2)
    model_r_GPy = GPy.models.TPRegression(train_r[:,0].numpy().reshape(-1, 1), train_r[:,1].numpy().reshape(-1, 1), kernel_simple, deg_free=100)
    #model_l_GPy = GPy.models.GPRegression(train_l[:1].numpy().reshape(-1, 1), train_l[1:].numpy().reshape(-1, 1), kernel_simple)
    #model_r_GPy = GPy.models.GPRegression(train_r[:1].numpy().reshape(-1, 1), train_r[1:].numpy().reshape(-1, 1), kernel_simple)
    model_l_GPy.optimize()
    model_r_GPy.optimize()
    test_x = np.linspace(-2, 2, 50).reshape(-1, 1)
    mean_l_Gpy, var_l_GPy = model_l_GPy.predict(test_x)
    mean_r_Gpy, var_r_GPy = model_r_GPy.predict(test_x)
    test=np.stack((np.vstack((test_x, mean_l_Gpy)).T,np.vstack((test_x, mean_r_Gpy)).T))
    Xnew = np.array([[3*v * dt*0.001]])
    mu_l_Gpy,_=model_l_GPy.predict(Xnew)
    mu_r_Gpy,_=model_r_GPy.predict(Xnew)
    '''
kernel_simple = GPy.kern.Linear(input_dim=1) +GPy.kern.Bias(input_dim=1)
kernel_rbf = GPy.kern.RBF(input_dim=1, ARD=True)
kernel_mat = GPy.kern.Matern52(1)
kernel_compose = kernel_simple+kernel_rbf