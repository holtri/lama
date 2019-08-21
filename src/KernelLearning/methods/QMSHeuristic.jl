struct QMSHeuristic
    k::Int
    p
    η
end

function QMSHeuristic(k; p=0.0, η = 0.0)
    QMSHeuristic(k, p, η)
end

function QMSHeuristic(;p = 0.05, η = 0.0)
    QMSHeuristic(0, p=p, η = η)
end

nu_to_C(nu, n) = 1/(n*nu)

function find_parameters(method::QMSHeuristic, data, labels, oracle, solver, qm, seed)

    if method.k == 0
        k = trunc(Int, method.p * size(data, 2)) + 1 # cf. p. 5069: K = ceil(p * n)
    else
        k = method.k
    end

    kdtree = KDTree(data; leafsize = k + 1)
    nn = knn(kdtree, data, k + 1, true)

    S_K = [1/k * sum(nn[2][i][2:end]) for i in 1:size(data, 2)]

    order = sortperm(S_K)

    # Detect change point, cf. Z. Ghafoori, S. Rajasegarar, S. M. Erfani, S. Karunasekera, and C. A. Leckie, “Unsupervised Parameter Estimation for One-Class Support Vector Machines,” in Advances in Knowledge Discovery and Data Mining, 2016, pp. 183–195.
    S_K_delta = S_K[order][2:end] - S_K[order][1:end-1]
    S_K_delta2 = S_K_delta[2:end] - S_K_delta[1:end-1];
    curvature = S_K_delta2 ./ (1 .+ S_K_delta[1:end-1].^2).^0.5;

    m = findmax(curvature)[2]

    # H1.nu
    nu = (size(data, 2) - (1 - method.η) * m) / size(data, 2)
    C = nu_to_C(nu, size(data, 2))

    # H2.gamma
    δ_min, q = findmin([nn[2][i][2] for i in 1:size(data,2)])
    δ_max = 1/(size(data,2) - 1) * sum(euclidean(data[:, q], data[:, i]) for i in 1:size(data, 2) if i != q)
    γ = -log(δ_min/δ_max) / (δ_max^2 - δ_min^2)

    return (γ, C)
end
