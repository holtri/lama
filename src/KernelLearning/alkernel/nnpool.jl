
struct NNPool
    nn_pools::Dict{Symbol, Dict{Int, Int}}
    pools::Vector{Symbol} # TODO: use labelmap

    nn::Vector{Vector{Int}}
    reverse_nn::Vector{Vector{Int}}
    symmetric_nn::Vector{Vector{Int}}
end

function NNPool(data; k = 20)
    nn_pools = Dict{Symbol, Dict{Int, Int}}()
    nn_pools[:Lin] = Dict{Int, Int}()
    nn_pools[:Lout] = Dict{Int, Int}()

    kdtree = KDTree(data; leafsize = k)
    nn = knn(kdtree, data, k, true)[1]
    reverse_nn = reverseKNN(nn)
    symmetric_nn = [n ∩ r for (n,r) in zip(nn, reverse_nn)]

    NNPool(nn_pools, fill(:U, size(data,2)), nn, reverse_nn, symmetric_nn)
end

function Base.push!(nnpool::NNPool, id::Int, pool_label::Symbol)
    if haskey(nnpool.nn_pools[pool_label], id)
        nnpool.nn_pools[pool_label][id] = nnpool.nn_pools[pool_label][id] + 1
    else
        nnpool.nn_pools[pool_label][id] = 1
    end
end

function add_to_pools!(nnpool::NNPool, id, label::Symbol, k::Int)
    pool_label = ifelse(label == :inlier, :Lin, :Lout)

    nnpool.pools[id] = pool_label

    if label == :inlier
        neighbors = nnpool.nn[id][1:k]
    else
        snn_elements = min(length(nnpool.symmetric_nn[id]), k)
        neighbors = nnpool.symmetric_nn[id][1:snn_elements]
    end
    for n in neighbors
        push!(nnpool, n, pool_label)
    end
end

function reverseKNN(nn)
    map(x -> findall(in.(x, nn)), eachindex(nn))
end
