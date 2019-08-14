
A(K1, K2) = dot(K1, K2) / sqrt(norm(K1) * norm(K2))

function alignment(nnpool::NNPool, data::Array{Float64, 2}, K_full::Array{Float64, 2}, nn_stability; adjusted)
    idx = collect(keys(nnpool.nn_pools[:Lin]) ∪ keys(nnpool.nn_pools[:Lout]));

    y = zeros(size(data, 2));
    y[idx] .= [get(nnpool.nn_pools[:Lin], id, 0) / (get(nnpool.nn_pools[:Lin], id, 0) + get(nnpool.nn_pools[:Lout], id, 0)) for id in idx]
    y[idx] = ifelse.(y[idx] .> 0.5, 1, -1)

    idx_in = findall(y .== 1)
    idx_out = findall(y .== -1)

    K_opt = y * y'

    kdtree = KDTree(data; leafsize = nn_stability)
    nn = knn(kdtree, data, nn_stability, true)[1]
    reverse_nn = KernelLearning.reverseKNN(nn)
    symmetric_nn = [n ∩ r for (n,r) in zip(nn, reverse_nn)];

    mask = falses(size(K_opt))

    for (_, id) in enumerate(findall(nnpool.pools .!= :U))
        if y[id] == 1
            nn_in = nn[id][1:end] ∩ idx_in
            nn_out = nn[id][1:end] ∩ idx_out
        else
            nn_in = setdiff(nn[id][1:end], reverse_nn[id][1:end]) ∩ idx_in
            nn_out = symmetric_nn[id][1:end] ∩ idx_out
        end

        if length(nn_in) > 0
            mask[id, nn_in[1:min(end, nn_stability)]] .= true
            mask[nn_in[1:min(end, nn_stability)], id] .= true
        end

        if length(nn_out) > 0
            mask[id, nn_out[1:min(end, nn_stability)]] .= true
            mask[nn_out[1:min(end, nn_stability)], id] .= true
        end
    end

    if adjusted
        K_full = centerkernelmatrix(K_full)
        K_opt = centerkernelmatrix(K_opt)
    end

    return A(K_full[mask], K_opt[mask])
end

function alignment(nnpool::NNPool, data::Array{Float64, 2}, gamma::Float64, nn_stability; adjusted=true)
    K_full = kernelmatrix(Val(:col), GaussianKernel(gamma), data)
    return alignment(nnpool, data, K_full, nn_stability; adjusted=adjusted)
end

function alignment_score(nnpool::NNPool, data::Array{Float64, 2}, K_full::Array{Float64, 2}, query_history, k; adjusted=true, n_sample = 100)
    candidates = [x for x ∈ 1:size(K_full, 1) if x ∉ query_history]
    scores = zeros(length(candidates))
    A = alignment(nnpool, data, K_full, k; adjusted=adjusted)

    if length(candidates) > n_sample
        candidates_subset = StatsBase.sample(eachindex(candidates), n_sample, replace = false)
    else
        candidates_subset = eachindex(candidates)
    end

    for i in candidates_subset
        tmp_in = deepcopy(nnpool)
        add_to_pools!(tmp_in, candidates[i], :inlier, k)
        A_in = alignment(tmp_in, data, K_full, k; adjusted=adjusted)

        tmp_out = deepcopy(nnpool)
        add_to_pools!(tmp_out, candidates[i], :outlier, k)
        A_out = alignment(tmp_out, data, K_full, k; adjusted=adjusted)

        scores[i] = min(abs(A - A_in), abs(A - A_out))
    end
    return (candidates, scores)
end

function next_query(nnpool::NNPool, data::Array{Float64, 2}, K_full::Array{Float64, 2}, query_history, nn_stability; adjusted)
    @assert length(query_history) < size(data,2)
    candidates, scores = alignment_score(nnpool, data, K_full, query_history, nn_stability; adjusted=adjusted)
    candidates[sortperm(scores, rev=true)][1]
end

function find_gamma(nnpool::NNPool, data::Array{Float64, 2}, nn_stability; adjusted=true, min_γ=0.1, max_γ=30.0)
    neg_alignment(gamma) = -alignment(nnpool, data, gamma, nn_stability; adjusted=adjusted)
    res = Optim.optimize(neg_alignment, min_γ, max_γ, abs_tol=0.1)
    return res.minimizer
end

function sample_alignments(data, labels, p, nn_stability; adjusted=true, n_iter=50, max_γ=30)
    n_in = trunc(Int, p * sum(labels .== :inlier))
    n_out = trunc(Int, p * sum(labels .== :outlier))

    @assert n_in > 0 && n_out > 0
    @debug n_in, n_out

    nnpool = NNPool(data, k=nn_stability)
    res = Vector{Float64}()

    for i in 1:n_iter
        nnpool.nn_pools[:Lin] = Dict{Int, Int}()
        nnpool.nn_pools[:Lout] = Dict{Int, Int}()
        nnpool.pools .= :U

        lin = StatsBase.sample(findall(labels .== :inlier), n_in, replace=false)
        lout = StatsBase.sample(findall(labels .== :outlier), n_out, replace=false)

        for id in (lin ∪ lout)
            add_to_pools!(nnpool, id, labels[id], nn_stability)
        end

        try
            gamma = find_gamma(nnpool, data, nn_stability; max_γ=max_γ, adjusted=adjusted)
            push!(res, gamma)
        catch ex
            println(ex)
        end
    end
    return res
end
