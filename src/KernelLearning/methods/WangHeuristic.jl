struct WangHeuristic
    init_strategy::SVDD.InitializationStrategyCombined
end

function WangHeuristic(;init_strategy)
    WangHeuristic(init_strategy)
end

function find_parameters(method::WangHeuristic, data, labels, oracle, solver, qm, seed)
    model = VanillaSVDD(data)
    SVDD.initialize!(model, method.init_strategy)
    return (MLKernels.getvalue(model.kernel_fct.alpha), model.C)
end

function Base.string(m::WangHeuristic)
    return "WangHeuristic"
end
