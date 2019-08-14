using ScikitLearn

@sk_import datasets: make_moons
@sk_import datasets: make_circles
@sk_import datasets: make_blobs
@sk_import datasets: make_s_curve

function generate_data_A(n_inlier = 300, n_outlier = 35)
    inliers = make_moons(n_inlier, true, 0.1, random_state=seed)[1]'
    out_tmp = make_blobs(n_outlier, n_features=2, centers=[(-.5, -.6), (1.5, 1.5), (-1, 1.5), (0.5, 1.5)], cluster_std = 0.2, random_state=seed)
    outliers = out_tmp[1]'
    data = hcat(inliers, outliers)
    labels = vcat(fill(:inlier, size(inliers, 2)), fill(:outlier, size(outliers,2)))
    return (data, labels)
end

function generate_data_B(n_inlier = 300, n_outlier = 35)
    inliers = make_circles(n_samples=n_inlier, shuffle=true, noise=0.1, random_state=seed, factor=0.8)[1]'
    out_tmp = make_blobs(n_outlier, n_features=2, centers=[(0, 0), (-1, 1), (1, -1), (1, 1), (-1, -1)], cluster_std = 0.2, random_state=seed)

    outliers = out_tmp[1]'
    data = hcat(inliers, outliers)
    labels = vcat(fill(:inlier, size(inliers, 2)), fill(:outlier, size(outliers,2)))
    return (data, labels)
end

function generate_data_C(n_inlier = 300, n_outlier = 35)
    inliers = make_blobs(n_inlier, n_features=2, centers=3, cluster_std = 0.2, center_box=(0.0, 0.0), random_state=seed)[1]'
    outliers = make_circles(n_outlier, true, 0.05, random_state=seed)[1]'

    data = hcat(inliers, outliers)
    labels = vcat(fill(:inlier, size(inliers, 2)), fill(:outlier, size(outliers,2)))
    return (data, labels)
end
