@recipe function plot_qs(
        qs::VersionSpacePQs,
        history::Vector{Int},
        pools::Vector{Symbol}; grid_resolution = 100, axis_overhang = 0.2)
    data = qs.lb_occ.data
    grid_range, grid_data = OCALPlots.get_grid(extrema(data)..., grid_resolution, axis_overhang)
    grid_scores = reshape(qs_score(qs, grid_data, labelmap(fill(:U, size(grid_range,2)))), grid_resolution, grid_resolution)

    kernel_gamma = MLKernels.getvalue(qs.lb_occ.kernel_fct.alpha)
    title := "gamma=$(round(kernel_gamma, digits=2)), C_LB=$(round(qs.lb_occ.C, digits=2)), C_UB=$(round(qs.ub_occ.C, digits=2)) i=$(length(history))"

    cbar := false
    @series begin
        seriestype := :contourf
        seriescolor --> :PuBu
        levels := range(0.0, maximum(grid_scores), length=10)
        grid_range, grid_range, grid_scores
    end

    markeralpha --> 0.7
    markersize --> 5

    sub_idx_history = [i ∈ history for i in 1:size(data, 2)]

    colors = Dict(:inlier => (U = :lightblue, Lin = :orange, Lout = :orange),
                  :outlier => (U = :darkblue, Lin = :orange, Lout = :orange))
    shapes = (U = :circle, Lin = :square, Lout = :utriangle)

    for (k,v) in filter(d -> first(d)!=:Lout, labelmap(pools))
        @series begin
            markeralpha := 1.0
            label := "inlier-$k"
            seriestype := :scatter
            markercolor := colors[:inlier][k]
            markershape := shapes[k]
            OCALPlots.split_2d_array(data, [i ∈ v for i in 1:size(data,2)] .& (labels .== :inlier))
        end
    end

    for (k,v) in filter(d -> first(d)!=:Lin, labelmap(pools))
        @series begin
            markeralpha := 1.0
            label := "outlier-$k"
            seriestype := :scatter
            markercolor := colors[:outlier][k]
            markershape := shapes[k]
            OCALPlots.split_2d_array(data, [i ∈ v for i in 1:size(data,2)] .& (labels .== :outlier))
        end
    end

    if !isempty(history)
        query_id = history[end]
        @series begin
            label := "current-selection"
            seriestype := :scatter
            markercolor := :red
            markeralpha := 1.0
            markershape := :star5
            markersize := 12
            OCALPlots.split_2d_array(data, 1:size(data, 2) .== query_id)
        end
    end
end
