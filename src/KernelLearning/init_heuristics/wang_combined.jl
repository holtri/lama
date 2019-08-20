struct WangCombinedStrategy <: SVDD.InitializationStrategyCombined
    solver
    gamma_search_range
    C
    scoring_function
end

fnr(x::ConfusionMatrix) = 1-tpr(x)

function SVDD.calculate_gamma(model, strategy::WangCombinedStrategy)
    m = deepcopy(model)
    data_target, data_outliers = SVDD.generate_binary_data_for_tuning(m.data)
    artificial_data = hcat(data_target, data_outliers)
    ground_truth = vcat(fill(:inlier, size(data_target, 2)),
                    fill(:outlier, size(data_outliers, 2)))

    @info("[Gamma Search] Searching for parameters gamma and C.")
    best_gamma = 1.0
    best_C = 1.0
    best_score = Inf
    for gamma in strategy.gamma_search_range

        init_strategy = FixedParameterInitialization(GaussianKernel(gamma), 1.0)
        initialize!(m, init_strategy)
        set_adjust_K!(m, true)
        for C in strategy.C
            try
                @info("[Gamma Search] Testing C = $C, gamma = $gamma")
                set_param!(m, Dict(:C => C))
                SVDD.fit!(m, strategy.solver)
            catch e
                @info("[Gamma Search] Fitting failed for C = $C, gamma $gamma.")
                println(e)
                continue
            end
            cm = ConfusionMatrix(SVDD.classify.(SVDD.predict(m, artificial_data)), ground_truth)

            score = 0.5 * fpr(cm) + 0.5 * fnr(cm)
            if best_score > score
                @info("[Gamma Search] New best found with gamma = $gamma and score = $score.")
                best_gamma = gamma
                best_C = C
                best_score = score
            end
        end
    end
    @info("Best paremeters: C=$best_C, gamma=$best_gamma")
    return (best_C, MLKernels.GaussianKernel(best_gamma))
end

function SVDD.get_parameters(model, strategy::WangCombinedStrategy)
    SVDD.calculate_gamma(model, strategy)
end
