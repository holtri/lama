mutable struct RandomSampleGrid <: SVDD.InitializationStrategyC
    solver
    C_range
    pools
    selection_metric
end

function SVDD.calculate_C(model, strategy::RandomSampleGrid, kernel)
    m = deepcopy(model)

    init_strategy = FixedParameterInitialization(kernel, 1.0)
    SVDD.initialize!(m, init_strategy)
    model_range = typeof(model)[]

    for C in strategy.C_range
        @debug "Training with C=$c"
        set_param!(m, Dict(:C => C))
        set_adjust_K!(m, true)
        SVDD.fit!(m, strategy.solver)
        push!(model_range, m)
        m = deepcopy(m)
    end

    idx_pools = (strategy.pools .!= :U)
    labels = ifelse.(strategy.pools[idx_pools] .== :Lin, :inlier, :outlier)

    class_range = map(x -> SVDD.classify.(SVDD.predict(x, m.data[:, idx_pools])), model_range)
    cms_pools = map(x -> ConfusionMatrix(x, labels), class_range)
    qm = strategy.selection_metric.(cms_pools)
    qm_best_idx = findmax(qm)[2]
    C = strategy.C_range[qm_best_idx]
    return C
end
