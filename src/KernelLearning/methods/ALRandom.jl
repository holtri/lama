struct ALRandom
    n_samples::Int
    nn_stability::Int
    max_gamma::Float64
    adjusted::Bool
    C_steps::Int
end

function ALRandom(;n_samples, nn_stability, max_gamma, adjusted, C_steps)
    ALRandom(n_samples, nn_stability, max_gamma, adjusted, C_steps)
end

function Base.string(m::ALRandom)
    return "ALRandom_n=$(m.n_samples)_nns=$(m.nn_stability)_adj=$(m.adjusted)"
end

function find_parameters(method::ALRandom, data, labels, oracle, solver, qm, seed)
    @show "Running $method"

    nnpool = NNPool(data, k=method.nn_stability)
    sample = StatsBase.sample(1:size(data,2), method.n_samples, replace=false)
    for i in sample
        add_to_pools!(nnpool, i, ask_oracle(oracle, i), method.nn_stability)
    end

    gamma = find_gamma(nnpool, data, method.nn_stability; max_γ=method.max_gamma, adjusted=method.adjusted)
    @show "Found gamma $gamma"
    max_C = find_max_feasible_C(data, gamma, solver)
    min_C = find_min_feasible_C(data, gamma, solver)

    C_range, model_range = calculate_C_grid(data, labels, oracle, gamma, min_C, max_C, method.C_steps, solver)
    qm = calculate_qm(nnpool, data, labels, model_range, qm)
    qm_best_idx = findmax(qm)[2]
    C = C_range[qm_best_idx]
    return (gamma, C)
end
