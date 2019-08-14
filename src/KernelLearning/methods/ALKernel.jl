struct ALKernel
    nn_stability::Int
    stability_window::Int
    stability_threshold::Float64
    C_steps::Int
    max_iteration::Int
    initial_n::Int
    max_gamma::Float64
    adjusted::Bool
end

function ALKernel(;nn_stability, stability_window, stability_threshold, C_steps, max_iteration, initial_n, max_gamma, adjusted)
    ALKernel(nn_stability, stability_window, stability_threshold, C_steps, max_iteration, initial_n, max_gamma, adjusted)
end

function Base.string(m::ALKernel)
    return "ALKernel_nns=$(m.nn_stability)_adj=$(m.adjusted)"
end

function find_parameters(method::ALKernel, data, labels, oracle, solver, qm, seed)
    nnpool, gamma_history, stability_history, query_history = search_gamma(data, labels, oracle, method.nn_stability,
                                                                           method.stability_window, method.stability_threshold, method.max_iteration,
                                                                           method.initial_n, method.max_gamma, seed, method.adjusted)
    gamma = gamma_history.values[end]
    
    max_C = find_max_feasible_C(data, gamma, solver)
    min_C = find_min_feasible_C(data, gamma, solver)

    C_range, model_range = calculate_C_grid(data, labels, oracle, gamma, min_C, max_C, method.C_steps, solver)
    qm = calculate_qm(nnpool, data, labels, model_range, qm)
    qm_best_idx = findmax(qm)[2]
    C = C_range[qm_best_idx]
    return (gamma, C)
end
