!isempty(ARGS) || error("No experiment config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the experiment config.")
@info "Experiment config supplied: '$(ARGS[1])'"
config_file = ARGS[1]

worker_config = "$(@__DIR__)/../config/config.jl"
@info "Loading worker config: $worker_config"
include(worker_config)

using JuMP, Gurobi
using SVDD, OneClassActiveLearning
using Serialization, Logging, Random

push!(LOAD_PATH, joinpath(@__DIR__, "../src/KernelLearning"))
using KernelLearning

function init_result_dir(exp_dir)
    if isdir(exp_dir)
        print("Type 'yes' or 'y' to delete and overwrite experiment $(exp_dir): ")
        argin = readline()
        if argin == "yes" || argin == "y"
            rm(exp_dir, recursive=true)
        else
            error("Overwriting anyways... Just kidding, nothing happened.")
        end
    else
        mkpath(exp_dir)
        mkpath(joinpath(exp_dir, "results"))
        mkpath(joinpath(exp_dir, "log"))
    end
end

function generate_experiments(methods, data_files, solver, oracle_param, quality_metrics)
    exp_dir = joinpath(data_output_root, exp_name)
    init_result_dir(exp_dir)

    experiments = []
    for data_file in data_files
        out_dir = split(data_file, '/')[end-1]
        output_path = joinpath(exp_dir, "results", out_dir)
        isdir(output_path) || mkdir(output_path)

        data, labels = load_data(data_file)

        oracle = OneClassActiveLearning.initialize_oracle(oracle_param[:type], data, labels, oracle_param[:param])
        for seed in seeds
            for method in methods
                experiment = Dict{Symbol, Any}(
                    :data_file => data_file,
                    :data_set_name => out_dir,
                    :oracle => oracle,
                    :method => method,
                    :seed => seed,
                    :solver => solver,
                    :quality_metrics => quality_metrics,
                    :selection_metric => selection_metric)
                exp_hash = hash(sprint(print, experiment))
                @assert exp_hash == hash(sprint(print, deepcopy(experiment)))
                experiment[:hash] = "$exp_hash"

                out_name = splitext(splitdir(data_file)[2])[1]
                out_name = joinpath(output_path, "$(out_name)_$(string(method))_$(exp_hash).json")
                experiment[:output_file] = out_name
                experiment[:log_dir] = joinpath(exp_dir, "log")
                push!(experiments, deepcopy(experiment))
            end
        end
    end

    cp(@__FILE__, joinpath(exp_dir, splitdir(@__FILE__)[2]), follow_symlinks=true)
    @info "Created $exp_dir with $(length(experiments)) instances."

    open(joinpath(exp_dir, "experiment_hashes"), "a") do f
        for e in experiments
            write(f, "$(e[:hash])\n")
        end
    end
    serialize(joinpath(exp_dir, "experiments.jser"), experiments)
    return nothing
end

include(config_file)

all(isdir.(joinpath.(data_input_root, data_dirs))) || error("Not all data dirs are valid.")
data_files = vcat(([joinpath.(data_input_root, x, readdir(joinpath(data_input_root, x))) for x in data_dirs])...)
@info("Found $(length(data_files)) data files.")

generate_experiments(learning_methods, data_files, solver, oracle_param, quality_metrics)
