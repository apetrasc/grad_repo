import gpytorch
import torch


class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

class TPUtils:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, train_x, train_y, learning_rate=0.1, training_iter=50):
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        
        # スチューデントのt尤度を使用
        likelihood = gpytorch.likelihoods.StudentTLikelihood().to(self.device)
        
        # ここで適切な数の誘導点を選ぶ必要があります
        inducing_points = train_x[:500]  # 例として最初の500点を誘導点とする
        model = VariationalGPModel(inducing_points).to(self.device)
        
        optimizer = torch.optim.Adam([
            {'params': model.parameters()}, 
            {'params': likelihood.parameters()}
        ], lr=learning_rate)

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

        model.train()
        likelihood.train()

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        
        return model, likelihood

    def predict_with_model(self, model, likelihood, test_x):
        model.eval()
        likelihood.eval()
        test_x = test_x.to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
        
        return observed_pred.mean, observed_pred.variance

    def set_eval_mode(self, *models_and_likelihoods):
        """
        与えられたモデルと尤度を評価モードに設定する。
        """
        for item in models_and_likelihoods:
            item.eval()