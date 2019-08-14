struct DFNGammaStrategy <: SVDD.InitializationStrategyGamma
    min_γ
    max_γ
end

DFNGammaStrategy() = DFNGammaStrategy(0.0, 100.0)

function dfn(nn, gamma)
    near = mean(exp(-gamma*nn_dist[2]) for nn_dist in nn[2])
    far = mean(exp(-gamma*nn_dist[end]) for nn_dist in nn[2])
    return 2*near - 2*far
end

function _dfn(data)
    kdtree = KDTree(data)
    nn = knn(kdtree, data, size(data, 2), true)
    return ((gamma) -> dfn(nn, gamma))
end

function SVDD.calculate_gamma(model, strategy::DFNGammaStrategy)
    f(gamma) = - _dfn(model.data)(gamma)
    res = Optim.optimize(f, strategy.min_γ, strategy.max_γ, abs_tol=0.1)
    return res.minimizer
end
