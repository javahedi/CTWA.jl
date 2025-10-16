
module Clustering

    using CTWA
    using Statistics
    using Random

    export naive_clustering, rg_clustering, print_clusters

    # ----------------------------------------------------------
    #  Naive (block) clustering
    # ----------------------------------------------------------
    """
        naive_clustering(N::Int, cluster_size::Int) -> Vector{Vector{Int}}

    Partitions `N` spins into contiguous clusters of fixed `cluster_size`.

    Example:
    ```julia
    naive_clustering(10, 3)
    # → [[1,2,3], [4,5,6], [7,8,9], [10]]
    ````

    """
    function naive_clustering(N::Int, cluster_size::Int)
        clusters = Vector{Vector{Int}}()
        i = 1
        while i <= N
            push!(clusters, collect(i:min(i + cluster_size - 1, N)))
            i += cluster_size
        end
        return clusters
    end

    # ----------------------------------------------------------

    # RG-inspired clustering based on coupling strengths

    # ----------------------------------------------------------

    """
    rg_clustering(N::Int, Jdict::Dict{Tuple{Int,Int}, Dict{Symbol,Float64}},
    cluster_size::Int)

    Construct clusters by grouping the most strongly coupled spins together, up to
    a target `cluster_size`.

    Heuristic steps:

    1. Find the strongest |J| bond between remaining spins.
    2. Create a cluster containing that pair.
    3. Grow the cluster by adding spins most strongly connected to it until `cluster_size`.
    4. Repeat until all spins are assigned.

    Returns a `Vector{Vector{Int}}` of clusters.
    """
    function rg_clustering(
                            N::Int,
                            Jdict::Dict{Tuple{Int,Int}, Dict{Symbol,Float64}},
                            cluster_size::Int)


        remaining = collect(1:N)
        clusters = Vector{Vector{Int}}()

        # Helper function: maximum absolute coupling between two sites
        function max_coupling(si, sj)
            key = si < sj ? (si, sj) : (sj, si)
            if haskey(Jdict, key)
                return maximum(abs.(values(Jdict[key])))
            else
                return 0.0
            end
        end

        while !isempty(remaining)
            cluster = Int[]

            # Step 1: find the strongest pair within remaining spins
            max_val = -Inf
            best_pair = nothing
            for i = 1:length(remaining)-1
                for j = i+1:length(remaining)
                    val = max_coupling(remaining[i], remaining[j])
                    if val > max_val
                        max_val = val
                        best_pair = (remaining[i], remaining[j])
                    end
                end
            end

            if isnothing(best_pair)
                # No couplings left; just take leftover spins
                cluster = remaining[1:min(cluster_size, length(remaining))]
                remaining = setdiff(remaining, cluster)
                push!(clusters, cluster)
                continue
            end

            push!(cluster, best_pair[1], best_pair[2])

            # Step 2: grow cluster up to desired size
            while length(cluster) < cluster_size && length(remaining) > length(cluster)
                candidates = setdiff(remaining, cluster)
                best_val = -Inf
                best_spin = 0
                for c in candidates
                    val = sum(max_coupling(c, s) for s in cluster)
                    if val > best_val
                        best_val = val
                        best_spin = c
                    end
                end
                push!(cluster, best_spin)
            end

            # Step 3: remove clustered spins from remaining
            remaining = setdiff(remaining, cluster)
            push!(clusters, cluster)
        end

        return clusters


    end

    # ----------------------------------------------------------

    # Utility to print cluster structure

    # ----------------------------------------------------------

    """
    print_clusters(clusters::Vector{Vector{Int}})

    Pretty print cluster information.
    """
    function print_clusters(clusters::Vector{Vector{Int}})
        println("Cluster decomposition:")
        for (ci, c) in enumerate(clusters)
            println("  Cluster $ci: ", c)
        end
        println("Total clusters: ", length(clusters))
    end

end # module

