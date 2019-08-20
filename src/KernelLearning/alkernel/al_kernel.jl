function add_initial_sample!(nnpool, labels, oracle, nn_stability, initial_n, query_history, gamma_progress)
    outlier_sample = StatsBase.sample(findall(labels .== :outlier), trunc(Int, initial_n / 2.0), replace=false)
    inlier_sample = StatsBase.sample(findall(labels .== :inlier), trunc(Int, initial_n / 2.0), replace=false)

    for (i,o) in enumerate(inlier_sample ∪ outlier_sample)
        add_to_pools!(nnpool, o, ask_oracle(oracle, o), nn_stability)
        push!(query_history, i, o)
        push!(gamma_progress, i, 1.0)
    end
end

function search_gamma(data, labels, oracle, nn_stability, stability_window,
                      stability_threshold, max_iteration, initial_n, max_γ, seed, adjusted)

    Random.seed!(seed)
    gamma_history = History(Float64)
    stability_history = History(Float64)
    query_history = History(Int64)

    nnpool = NNPool(data, k=nn_stability)
    add_initial_sample!(nnpool, labels, oracle, nn_stability, initial_n, query_history, gamma_history)

    for i in initial_n+1:max_iteration
        gamma = find_gamma(nnpool, data, nn_stability; max_γ=max_γ, adjusted=adjusted)

        K = kernelmatrix(Val(:col), GaussianKernel(gamma), data)
        current_alignment = alignment(nnpool, data, K, nn_stability; adjusted=adjusted)

        query = next_query(nnpool, data, K, query_history.values, nn_stability; adjusted=adjusted)
        add_to_pools!(nnpool, query, ask_oracle(oracle, query), nn_stability)

        stability = 0.0
        if (i > stability_window + 3)
            # smoothing
            tmp = rollmean(gamma_history.values, stability_window)
            # take average lag-1 difference
            stability = mean(min_max_normalize(abs.(tmp[1:end-1] - tmp[2:end]))[end-2:end])
        end

        push!(query_history, i, query)
        push!(gamma_history, i, gamma)
        push!(stability_history, i, stability)
        @info i, gamma, stability
        if (stability > 0) && (stability < stability_threshold)
            @info "Gamma stable. Stability: $stability"
            break
        end
    end
    return nnpool, gamma_history, stability_history, query_history
end

function calculate_C_grid(data, labels, oracle, gamma, min_C, max_C, steps, solver)
    C_range = range(min_C, max_C, length=steps)
    model_range = VanillaSVDD[]
    for c in C_range
        @debug "Training with C=$c"
        m = VanillaSVDD(data)
        SVDD.initialize!(m, SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(gamma)), FixedCStrategy(c)))
        set_adjust_K!(m, true)
        SVDD.fit!(m, solver)
        push!(model_range, m)
    end
    return (collect(C_range), model_range)
end

function calculate_qm(nnpool, data, labels, model_range, qm)
    idx_pools = (nnpool.pools .!= :U)
    class_range = map(m -> SVDD.classify.(SVDD.predict(m, data[:, idx_pools])), model_range)
    cms_pools = map(x -> ConfusionMatrix(x, labels[idx_pools]), class_range)
    return qm.(cms_pools)
end

function find_min_feasible_C(data, gamma, solver, c_max = 0.5, c_min = 0.0, iter=10)
    c = (c_max - c_min) / 2
    c_opt = c

    for i in 1:10
        m = VanillaSVDD(data)
        SVDD.initialize!(m, SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(gamma)), FixedCStrategy(c)))
        set_adjust_K!(m, true)
        res = SVDD.fit!(m, solver)
        if res == JuMP.MathOptInterface.OPTIMAL
            c_opt = c
            c_max = c
            c = c_min + (c - c_min) / 2
        else
            c_min = c
            c = c + (c_max - c) / 2
        end
    end
    return c_opt
end

function find_max_feasible_C(data, gamma, solver, c_max = 1.0, c_min = 0.0, iter=10)
    c = (c_max - c_min) / 2
    c_opt = c

    for i in 1:10
        m = VanillaSVDD(data)
        SVDD.initialize!(m, SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(gamma)), FixedCStrategy(c)))
        set_adjust_K!(m, true)
        SVDD.fit!(m, solver)

        if any(SVDD.classify.(SVDD.predict(m, data)) .== :outlier)
            c_min = c
            c = c + (c_max - c) / 2
        else
            c_opt = c
            c_max = c
            c = c_min + (c - c_min) / 2
        end
    end
    return c_opt
end
