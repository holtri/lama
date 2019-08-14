using Distributed

include("$(@__DIR__)/../config/config.jl")
addprocs(num_workers; exeflags="--project=$(abspath(joinpath(@__DIR__, "..")))")

@everywhere begin
    modulepath = abspath(joinpath(@__DIR__, "../src/KernelLearning"));
    push!(LOAD_PATH, modulepath);
    using KernelLearning
end

@everywhere using OneClassActiveLearning, SVDD
@everywhere using MLKernels
@everywhere using JSON, Serialization
@everywhere using Logging
@everywhere using Random

@everywhere function evaluate_parameters(gamma, C, data, labels, solver)
    model = VanillaSVDD(data)
    SVDD.initialize!(model, SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(gamma)), FixedCStrategy(C)))
    set_adjust_K!(model, true)
    SVDD.fit!(model, solver)
    return ConfusionMatrix(SVDD.classify.(SVDD.predict(model, data)), labels)
end

@everywhere function run_experiment(e::Dict{Symbol, Any})
    Random.seed!(e[:seed])

    io = open(joinpath(e[:log_dir], "$(e[:hash]).log"), "a")
    logger = SimpleLogger(io)
    global_logger(logger)

    if isfile(e[:output_file])
        @info("Skipping $(e[:hash])")
        return nothing
    end

    @info("Running $(e[:hash]), $(e[:data_set_name]), $(string(e[:method]))")
    @show e[:hash], e[:data_set_name], e[:method]

    result = deepcopy(e)
    result[:method] = string(e[:method])
    data, labels = load_data(e[:data_file])

    result[:datastats] = Dict(:n_observations => size(data, 2),
                               :n_dims => size(data, 1),
                               :outlier_ratio => sum(labels .== :outlier) / size(data, 2))

    gamma, C = KernelLearning.find_parameters(e[:method], data, labels, e[:oracle], e[:solver], e[:quality_metrics][e[:selection_metric]], e[:seed])
    cm = evaluate_parameters(gamma, C, data, labels, e[:solver])
    result[:result] = Dict(:gamma => gamma, :C => C, :metrics => Dict{Symbol, Float64}())
    for (metric_name, metric) in e[:quality_metrics]
        m = metric(cm)
        result[:result][:metrics][metric_name] = m
        @info("$metric_name: $m")
    end

    open(e[:output_file], "w") do f
        JSON.print(f, result)
    end

    close(io)
    return nothing
end

exp_dir = joinpath(data_output_root, exp_name)
@info("Running experiment in $exp_dir")

# store experiments
experiments = deserialize(open(joinpath(exp_dir, "experiments.jser")))
open(joinpath(exp_dir, "experiments.json"), "w") do f
   JSON.print(f, experiments)
end

# run experiments
pmap(run_experiment, experiments, on_error=ex->print(ex))
