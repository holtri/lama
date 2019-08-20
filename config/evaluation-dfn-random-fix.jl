exp_name = "dfn-random-fixed"

data_dirs = [
             # "ALOI", # too many observations
             "Annthyroid",
             # "Arrhythmia", # too many attributes
             "Cardiotocography",
             "Glass",
             "HeartDisease",
             "Hepatitis",
             "Ionosphere",
             # "InternetAds", # too many attributes
             # "KDDCup99", # too many observations
             "Lymphography",
             "PageBlocks",
             "Parkinson",
             "PenDigits",
             "Pima",
             "Shuttle",
             "SpamBase",
             "Stamps",
             "Waveform",
             "WBC",
             "WDBC",
             "WPBC"]

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
oracle_param = Dict{Symbol, Any}( :type => PoolOracle, :param => Dict{Symbol, Any}())

quality_metrics = Dict(:kappa => cohens_kappa, :f1 => f1_score)
selection_metric = :kappa

n_iter=50

learning_methods = vcat(
    [DFNHeuristic(KernelLearning.DFNGammaStrategy(),
                  KernelLearning.RandomSampleGrid(solver,
                                                 range(0.001, stop=0.1, length=20),
                                                 nothing,
                                                 quality_metrics[selection_metric]),
                 n_iter)]
)

seeds = [42 * i for i in 1:5]
