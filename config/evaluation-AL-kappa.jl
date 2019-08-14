exp_name = "AL-kappa"

data_dirs = ["Ionosphere","PageBlocks"]

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
oracle_param = Dict{Symbol, Any}( :type => PoolOracle, :param => Dict{Symbol, Any}())

quality_metrics = Dict(:kappa => cohens_kappa, :f1 => f1_score)
selection_metric = :kappa

n_iter = 50
n_c_steps = 20

learning_methods = vcat(
    # Kernel Active Learning with custom selection strategy
    [ALKernel(nn_stability = nns,
             stability_window = 10,
             stability_threshold = st,
             C_steps = n_c_steps,
             max_iteration = n_iter,
             initial_n = 4,
             max_gamma = 100.0,
             adjusted = true) for nns in 1:1:30 for st in [0.0]])

seeds = [i * 42 for i in 1:10]
