exp_name = "Competitors"

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
    # DFN author recommendation
    [DFNHeuristic(KernelLearning.DFNGammaStrategy(),
                  FixedCStrategy(0.05),
                  n_iter)],
    # DFN with random grid
    [DFNHeuristic(KernelLearning.DFNGammaStrategy(),
                  KernelLearning.RandomSampleGrid(solver,
                                                 range(0.001, stop=0.1, length=20),
                                                 nothing,
                                                 quality_metrics[selection_metric]),
                 n_iter)],
    # Wang parametrization from paper
    [WangHeuristic(init_strategy= KernelLearning.WangCombinedStrategy(solver,
                                                                     [10^x for x in -4.0:1.0:4.0],
                                                                     [0.01, 0.05],
                                                                     SVDD.f1_scoring)),
     # Wang extended parameter range
     WangHeuristic(init_strategy= KernelLearning.WangCombinedStrategy(solver,
                                                                      10.0.^range(-.1, stop=2, length=20),
                                                                      range(0.001, stop=0.1, length=20),
                                                                      SVDD.f1_scoring))],
    # Fully supervised baseline
    [SupervisedGridSearch(gamma_range = 10.0.^range(-.1, stop=2, length=20),
                          C_range = range(0.001, stop=0.1, length=20))],
    # QMS
    [QMSHeuristic(p=0.05)]
)

seeds = [42 * i for i in 1:5]
