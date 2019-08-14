struct SupervisedGridSearch
    gamma_range
    C_range
end

function SupervisedGridSearch(;gamma_range, C_range)
    SupervisedGridSearch(gamma_range, C_range)
end

function Base.string(m::SupervisedGridSearch)
    return "SupervisedGridSearch_C-$(minimum(m.C_range))-$(maximum(m.C_range))_gamma-$(minimum(m.gamma_range))-$(maximum(m.gamma_range))"
end

function find_parameters(method::SupervisedGridSearch, data, labels, oracle, solver, qm, seed)
    model = VanillaSVDD(data)
    best_gamma = 1.0
    best_C = 1.0
    best_score = -Inf
    for γ in method.gamma_range
        SVDD.initialize!(model, SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(γ)), FixedCStrategy(1.0)))
        set_adjust_K!(model, true)
        for C in method.C_range
            set_param!(model, Dict(:C => C))
            SVDD.fit!(model, solver)
            current_score = qm(ConfusionMatrix(SVDD.classify.(SVDD.predict(model, data)), labels))

            if current_score > best_score
                best_C = C
                best_gamma = γ
                best_score = current_score
            end
        end
    end
    return (best_gamma, best_C)
end
