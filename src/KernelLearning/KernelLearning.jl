module KernelLearning

using StatsBase
using NearestNeighbors
using MLKernels
using LinearAlgebra
using Optim
using Serialization

using JuMP, Gurobi
using Random
using Logging
using LinearAlgebra
using Distributions
using RollingFunctions

using SVDD, OneClassActiveLearning
using ValueHistories
using MLLabelUtils

using Distances

include("init_heuristics/wang_combined.jl")
include("init_heuristics/xiao_gamma.jl")
include("init_heuristics/c_on_sample.jl")

methods_dir = joinpath(@__DIR__, "methods")
for f in readdir(methods_dir)
    include(joinpath(methods_dir, f))
end

include("alkernel/nnpool.jl")
include("alkernel/alignment.jl")
include("alkernel/al_kernel.jl")

include("fixes.jl")

export
    NNPool,

    ALKernel,
    ALRandom,
    WangHeuristic,
    SupervisedAlignment,
    SupervisedGridSearch,
    DFNHeuristic,
    QMSHeuristic,

    find_parameters,

    DFNGammaStrategy,
    RandomSampleGrid
end
