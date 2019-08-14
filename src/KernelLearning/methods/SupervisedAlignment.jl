struct SupervisedAlignment
    nn_stability::Int
    C_steps::Int
    balanced::Bool
    adjusted::Bool
    max_gamma::Float64
end

function SupervisedAlignment(;nn_stability, C_steps, balanced, adjusted, max_gamma)
    SupervisedAlignment(nn_stability, C_steps, balanced, adjusted, max_gamma)
end

function Base.string(m::SupervisedAlignment)
    return "SupervisedAlignment_adj=$(m.adjusted)"
end

function find_parameters(method::SupervisedAlignment, data, labels, oracle, solver, qm, seed)
    nnpool = NNPool(data, k=1)

    for i in 1:size(data, 2)
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
