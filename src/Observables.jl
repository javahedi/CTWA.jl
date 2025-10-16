module Observables

    using LinearAlgebra

    export single_spin_expectation, two_point_expectation, delta_S

    """
        single_spin_expectation(sol, clusters, cluster_len, N)

    Compute time-dependent single-spin expectations <σx_i>, <σy_i>, <σz_i>
    Returns: Dict(:x => Matrix, :y => Matrix, :z => Matrix)
    Each matrix is size (n_times, N)
    """
    function single_spin_expectation(sol, clusters, cluster_len, N)
        n_times = length(sol.t)
        X = zeros(n_times, N)
        Y = zeros(n_times, N)
        Z = zeros(n_times, N)

        for ti in 1:n_times
            offset = 0
            spin_counter = 1
            for (ci, cl) in enumerate(clusters)
                n_spin = length(cl)
                for s in 1:n_spin
                    X[ti, spin_counter] = sol.u[ti][offset + 3*(s-1) + 1]
                    Y[ti, spin_counter] = sol.u[ti][offset + 3*(s-1) + 2]
                    Z[ti, spin_counter] = sol.u[ti][offset + 3*(s-1) + 3]
                    spin_counter += 1
                end
                offset += cluster_len[ci]
            end
        end

        return Dict(:x => X, :y => Y, :z => Z)
    end

    """
        two_point_expectation(sol, clusters, cluster_len, N, op1, op2)

    Compute <σ_op1_i σ_op2_j> correlations.
    op1, op2 = :x, :y, :z
    Returns: Array of size (n_times, N, N)
    """
    function two_point_expectation(sol, clusters, cluster_len, N, op1::Symbol, op2::Symbol)
        n_times = length(sol.t)
        corr = zeros(n_times, N, N)
        obs = single_spin_expectation(sol, clusters, cluster_len, N)

        for ti in 1:n_times
            for i in 1:N
                for j in 1:N
                    corr[ti,i,j] = obs[op1][ti,i] * obs[op2][ti,j]
                end
            end
        end

        return corr
    end

    """
        delta_S(sol, clusters, cluster_len, N, op::Symbol)

    Compute variance of total spin S_op = sum_i σ_op_i:
    ΔS_op = <S_op^2> - <S_op>^2
    Returns vector of length n_times
    """
    function delta_S(sol, clusters, cluster_len, N, op::Symbol)
        obs = single_spin_expectation(sol, clusters, cluster_len, N)
        n_times = length(sol.t)
        ΔS = zeros(n_times)

        for ti in 1:n_times
            S_op_vec = obs[op][ti, :]
            ΔS[ti] = sum(S_op_vec.^2) + sum([S_op_vec[i]*S_op_vec[j] for i in 1:N, j in 1:N if i!=j]) - sum(S_op_vec)^2
            # Actually: sum_i S_i^2 + sum_{i!=j} S_i S_j - (sum_i S_i)^2 = 0
            # Better: ΔS = sum_i S_i^2 - (sum_i S_i)^2 + sum_{i!=j} (S_i S_j - S_i S_j) = sum_i S_i^2 - (sum_i S_i)^2
            ΔS[ti] = sum(S_op_vec.^2) - (sum(S_op_vec))^2
        end

        return ΔS
    end

end # module
