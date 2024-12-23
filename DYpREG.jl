using Random, Statistics, LinearAlgebra
using BayesianOptimization, GaussianProcesses
using DelimitedFiles
using ProximalOperators, ProximalAlgorithms

# Read Cleve2 data
#dat = readdlm("C://Users//pwaldman20//Documents//Cleve2.csv", ',')
dat = readdlm("C://A proximal//Cleve4.csv",',')

# Extract and standardize data
inp = dat[1:end, 3:end]
X = (inp .- mean(inp, dims=1)) ./ std(inp, dims=1)

# Generate train and test data
fold = 1
cv1ind = randperm(MersenneTwister(fold), size(X)[1])
ytrain = dat[cv1ind[1:1900], 1]
ytest = dat[cv1ind[1901:end], 1]
Xtrain = X[cv1ind[1:1900], :]
Xtest = X[cv1ind[1901:end], :]
ebvtrain = dat[cv1ind[1:1900], 2]
m, n = size(Xtrain)

# Fixed learning rate Lf
Lf = opnorm(Xtrain)^2

# Convergence tolerance
TOL = 1e-5

# Mean squared error function
mean_squared_error(label, output) = mean((output .- label) .^ 2) / 2

# Define DYpEN
function DYpEN(eps, lam1, lam2)
    f = LeastSquares(Xtrain, ytrain)
    g = LeastSquares(Xtrain, ebvtrain, eps)
    h = ElasticNet(lam1, lam2)
    x0 = zeros(n)
    drls1 = ProximalAlgorithms.DavisYin(maxit=5000, tol=TOL)
    z, it = drls1(x0 = x0, f = f, g = g, h = h, Lf = Lf)
    MSE = mean_squared_error(Xtest * z, ytest)
    pcor = cor(Xtest * z, ytest)
    return MSE, pcor
end

# Define DYpRR
function DYpRR(eps, lam1)
    f = LeastSquares(Xtrain, ytrain)
    g = LeastSquares(Xtrain, ebvtrain, eps)
    h = SqrNormL2(lam1)
    x0 = zeros(n)
    drls1 = ProximalAlgorithms.DavisYin(maxit=5000, tol=TOL)
    z, it = drls1(x0 = x0, f = f, g = g, h = h, Lf = Lf)
    MSE = mean_squared_error(Xtest * z, ytest)
    pcor = cor(Xtest * z, ytest)
    return MSE, pcor
end

# Define DYpLASSO
function DYpLASSO(eps, lam1)
    f = LeastSquares(Xtrain, ytrain)
    g = LeastSquares(Xtrain, ebvtrain, eps)
    h = NormL1(lam1)
    x0 = zeros(n)
    drls1 = ProximalAlgorithms.DavisYin(maxit=5000, tol=TOL)
    z, it = drls1(x0 = x0, f = f, g = g, h = h, Lf = Lf)
    MSE = mean_squared_error(Xtest * z, ytest)
    pcor = cor(Xtest * z, ytest)
    return MSE, pcor
end

# Select and optimize the method
function optimize_method(method::String)
    if method == "DYpEN"
        opt = BOpt(par -> DYpEN(par[1], par[2], par[3]),
            ElasticGPE(3, mean = MeanConst(0.), kernel = Mat52Ard(ones(3), 2.),
            logNoise = 1., capacity = 500),
            UpperConfidenceBound(),
            MAPGPOptimizer(every = 10, noisebounds = [-4., 2.],
                kernbounds = [[-5 * ones(3); -5], [4 * ones(3); 4]],
                maxeval = 40),
            [0.01, 1.0, 1.0], [1.0, 100.0, 100.0], repetitions = 3, maxiterations = 100,
            sense = Min,
            verbosity = Progress)
    elseif method == "DYpRR"
        opt = BOpt(par -> DYpRR(par[1], par[2]),
            ElasticGPE(2, mean = MeanConst(0.), kernel = Mat52Ard(ones(2), 2.),
            logNoise = 1., capacity = 500),
            UpperConfidenceBound(),
            MAPGPOptimizer(every = 10, noisebounds = [-4., 2.],
                kernbounds = [[-5 * ones(2); -5], [4 * ones(2); 4]],
                maxeval = 40),
            [0.01, 5000.0], [2.5, 50000.0], repetitions = 3, maxiterations = 100,
            sense = Min,
            verbosity = Progress)
    elseif method == "DYpLASSO"
        opt = BOpt(par -> DYpLASSO(par[1], par[2]),
            ElasticGPE(2, mean = MeanConst(0.), kernel = Mat52Ard(ones(2), 2.),
            logNoise = 1., capacity = 500),
            UpperConfidenceBound(),
            MAPGPOptimizer(every = 10, noisebounds = [-4., 2.],
                kernbounds = [[-5 * ones(2); -5], [4 * ones(2); 4]],
                maxeval = 40),
            [1.0, 10.0], [100.0, 500.0], repetitions = 3, maxiterations = 100,
            sense = Min,
            verbosity = Progress)
    else
        error("Invalid method! Choose 'DYpEN', 'DYpRR', or 'DYpLASSO'.")
    end

    result = boptimize!(opt)
    best_params = result.best_params

    if method == "DYpEN"
        best_MSE, best_pcor = DYpEN(best_params[1], best_params[2], best_params[3])
    elseif method == "DYpRR"
        best_MSE, best_pcor = DYpRR(best_params[1], best_params[2])
    else
        best_MSE, best_pcor = DYpLASSO(best_params[1], best_params[2])
    end

    println("Best Hyperparameters: ", best_params)
    println("Best MSE: ", best_MSE)
    println("Best Pearson Correlation: ", best_pcor)
end

# Example usage
optimize_method("DYpREG")  # Replace with "DYpRR" or "DYpLASSO" as needed
