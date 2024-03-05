import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# 変分ガウス過程モデルの定義
class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# データの生成
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
#train_y = torch.exp(train_x) + torch.randn(train_x.size()) * 0.2
outliers = torch.randint(0, 100, (10,))
train_y[outliers] += 5 * torch.randn(10)

# モデルと尤度の定義
inducing_points = train_x[:50]  # 代表点を選択
model = GPModel(inducing_points)
likelihood = gpytorch.likelihoods.StudentTLikelihood()

# 最適化器の定義
model.train()
likelihood.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# 損失関数の定義（変分エルボ）
mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())

# 訓練ループ
training_iter = 50
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()


likelihood_gp = gpytorch.likelihoods.GaussianLikelihood()
model_gp= ExactGPModel(train_x, train_y, likelihood_gp)
model_gp.train()
likelihood_gp.train()
optimizer_gp = torch.optim.Adam(model_gp.parameters(), lr=0.1)
mll_gp = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gp, model_gp)
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer_gp.zero_grad()
    # Output from model
    output = model_gp(train_x)
    # Calc loss and backprop gradients
    loss_gp = -mll_gp(output, train_y)
    loss_gp.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss_gp.item(),
        model_gp.covar_module.base_kernel.lengthscale.item(),
        model_gp.likelihood.noise.item()
    ))
    optimizer_gp.step()
# テストデータでの予測
model.eval()
likelihood.eval()
model_gp.eval()
likelihood_gp.eval()
test_x = torch.linspace(0, 1, 100)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    observed_pred_gp=likelihood_gp(model_gp(test_x))

mean = observed_pred.mean
std = observed_pred.stddev
lower, upper = mean - 2 * std, mean + 2 * std
f_mean = observed_pred_gp.mean
f_var = observed_pred_gp.variance
f_covar = observed_pred_gp.covariance_matrix
# 予測結果のプロット
with torch.no_grad():
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    for i in range(2):  # mean の行数だけ繰り返す
        plt.plot(test_x.numpy(), mean[i].numpy(), label=f'Sample {i+1}')
    ax.plot(test_x.numpy(), observed_pred_gp.mean.numpy(), 'b')
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Approx. Confidence'])
    plt.show()

