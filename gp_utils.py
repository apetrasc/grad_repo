import torch
import gpytorch

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
class GPUtils:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {self.device}")

    def train_model(self, train_x, train_y, learning_rate=0.1, training_iter=50):
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = ExactGPModel(train_x, train_y, likelihood).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

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