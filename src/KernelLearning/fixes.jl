function ask_oracle(oracle::PoolOracle, query_id::Int)
    return OneClassActiveLearning.ask_oracle(oracle, [query_id])[1]
end
