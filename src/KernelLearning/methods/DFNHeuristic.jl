struct DFNHeuristic{C <: SVDD.InitializationStrategyC}
    gamma_strategy::DFNGammaStrategy
    C_Strategy::C
    n_samples::Int
end

function find_parameters(method::DFNHeuristic{RandomSampleGrid}, data, labels, oracle, solver, qm, seed)
    model = VanillaSVDD(data)

    # draw random sample
    method.C_Strategy.pools = fill(:U, size(data, 2))
    samples = StatsBase.sample(1:size(data, 2), method.n_samples, replace=false)
    method.C_Strategy.pools[samples] .= ifelse.(OneClassActiveLearning.ask_oracle(oracle, samples) .== :inlier, :Lin, :Lout)

    init_strategy = GammaFirstCombinedStrategy(method.gamma_strategy, method.C_Strategy)
    SVDD.initialize!(model, init_strategy)
    return (MLKernels.getvalue(model.kernel_fct.alpha), model.C)
end

function find_parameters(method::DFNHeuristic{FixedCStrategy}, data, labels, oracle, solver, qm, seed)
    model = VanillaSVDD(data)

    init_strategy = SimpleCombinedStrategy(method.gamma_strategy, method.C_Strategy)
    SVDD.initialize!(model, init_strategy)
    return (MLKernels.getvalue(model.kernel_fct.alpha), model.C)
end

function Base.string(m::DFNHeuristic)
    return "DFNHeuristic-$(typeof(m.C_Strategy))"
end
