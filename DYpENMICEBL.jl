using Random, Statistics,LinearAlgebra
using BayesianOptimization, GaussianProcesses
using DelimitedFiles
using ProximalOperators,ProximalAlgorithms
# Read mice body length data
dat = readdlm("MiceBL.txt",',')
# Extract and standardize marker data
inp = dat[1:end,3:end]
X = (inp .- mean(inp,dims=1)) ./ std(inp,dims=1)
# Generate the train and test data for fold 1
fold = 1
cv1ind = randperm(MersenneTwister(fold), size(X)[1])
ytrain = dat[cv1ind[1:1270],1]
ytest = dat[cv1ind[1271:size(X)[1]],1]
Xtrain = X[cv1ind[1:1270],:]
Xtest= X[cv1ind[1271:size(X)[1]],:]
ebvtrain = dat[cv1ind[1:1270],2]
m, n = size(Xtrain)
# Fixed learning rate Lf
Lf = opnorm(Xtrain)^2
# Convergence tolerance
TOL = 1e-5
# Function to obtain MSE
mean_squared_error(label, output) = mean((output .- label) .^ 2) / 2
# The Davies-Yin prior Elastic-Net
DYpEN(par::Vector) = DYpEN(par[1],par[2],par[3])
function DYpEN(par1,par2,par3)
  eps = par1
  lam1 = par2
  lam2 = par3
  f = LeastSquares(Xtrain, ytrain)
  g = LeastSquares(Xtrain, ebvtrain,eps)
  h = ElasticNet(lam1,lam2)
  x0 = zeros(n)
  drls1 = ProximalAlgorithms.DavisYin(maxit=5000, tol=TOL)
  z, it = drls1(x0 = x0, f = f, g = g, h = h, Lf = Lf)
  MSE = mean_squared_error(Xtest*z,ytest)
  return MSE
end
# Bayesian optimization of hyperparameters
optDYpEN = BOpt(par->DYpEN(par[1],par[2],par[3]),
  ElasticGPE(3, mean = MeanConst(0.), kernel = Mat52Ard(ones(3), 2.),
  logNoise = 1., capacity = 500),
  UpperConfidenceBound(),
  MAPGPOptimizer(every = 10, noisebounds = [-4.,2.],
    kernbounds = [[-5*ones(3); -5], [4*ones(3); 4]],
    maxeval = 40),
  [1.0,100.0,0.1], [100.0,400.0,50.0], repetitions = 3, maxiterations = 250,
  sense = Min,
  verbosity = Progress)
resultoptDYpEN = boptimize!(optDYpEN)
