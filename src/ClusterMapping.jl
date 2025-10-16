module ClusterMapping
    
    using CTWA
    using SparseArrays
    using LinearAlgebra


    export microscopic_to_cluster, add_single_field!, add_two_site_intra!


    # -----------------------------
    #  maping from original hamiltonina to cluster version
    # -----------------------------

    # Pauli index mapping: 0=I, 1=X, 2=Y, 3=Z
    const PAULI_DIGIT = Dict(:x => 1, :y => 2, :z => 3)

    # ==============================================================
    #  Utility: map Pauli digit vector → basis index (skipping identity)
    # ==============================================================

    """
        digits_to_index(digits::Vector{Int}) -> Int

    Converts a base-4 digit vector representing Pauli operators 
    (`0=I, 1=X, 2=Y, 3=Z`) into a **1-based index** within the 
    traceless basis (identity excluded).

    Example:
    digits_to_index([1,0,0])  # -> 1
    digits_to_index([0,2,0])  # -> 2*4^(2-1) = 8 -> 8 (after skipping identity)
    """
    function digits_to_index(digits::Vector{Int})
        idx = 0
        n = length(digits)
        for k in 1:n
            idx += digits[k] * 4^(n-k)
        end
        if idx == 0
            error("Identity operator is excluded in traceless basis")
        end
        return idx
    end


    # ==============================================================
    #Add field/coupling contributions to Bcluster
    # ==============================================================
    """
    add_single_field!(Bvec, cluster, local_site, pauli_sym, val)
    Adds a single-site field term val * σᵃ for the site local_site
    inside cluster to the cluster’s local effective field vector Bvec.
    """
    function add_single_field!(Bvec::Vector{Float64}, 
                                cluster::Vector{Int}, 
                                local_site::Int, 
                                pauli_sym::Symbol, 
                                val::Float64)

        n = length(cluster)
        local_pos = findfirst(==(local_site), cluster)
        @assert !isnothing(local_pos) "site $local_site not in cluster"
        digits = zeros(Int, n)
        digits[local_pos] = PAULI_DIGIT[pauli_sym]
        idx = digits_to_index(digits)
        Bvec[idx] += val
        return nothing
    end

    """
    add_two_site_intra!(Bvec, cluster, siteA, pauliA, siteB, pauliB, val)
    Adds a two-site intra-cluster coupling term
    val * σᵃᵢ σᵇⱼ to the local Bvec (folded into the cluster field).
    """
    function add_two_site_intra!(Bvec::Vector{Float64}, 
                                cluster::Vector{Int}, 
                                siteA::Int, pauliA::Symbol, 
                                siteB::Int, pauliB::Symbol, 
                                val::Float64)

        n = length(cluster)
        posA = findfirst(==(siteA), cluster)
        posB = findfirst(==(siteB), cluster)
        @assert !isnothing(posA) && !isnothing(posB)
        digits = zeros(Int, n)
        digits[posA] = PAULI_DIGIT[pauliA]
        digits[posB] = PAULI_DIGIT[pauliB]
        idx = digits_to_index(digits)
        Bvec[idx] += val
        return nothing
    end

    
    """
    microscopic_to_cluster(clusters, Bvec, Jdict)
    -> Bcluster::Dict{Int,Vector{Float64}},
    Jcluster::Dict{Tuple{Int,Int},SparseMatrixCSC{Float64,Int}}
    Constructs the cluster-level Hamiltonian representation from the
    microscopic local fields Bvec and couplings Jdict.
    Bvec is an N×3 matrix of fields [hx, hy, hz].
    Jdict is a dictionary of pairwise couplings like:
    Dict((i,j) => Dict(:xx=>Jxx, :yy=>Jyy, :zz=>Jzz)).
    clusters is a vector of integer vectors (each cluster’s spin indices).
    Returns:
    Bcluster[ci] → effective local field vector for cluster ci
    Jcluster[(ci,cj)] → sparse matrix coupling between cluster ci and cj
    """
    function microscopic_to_cluster(clusters::Vector{Vector{Int}}, 
                                    Bvec::AbstractMatrix{<:Real}, 
                                    Jdict::Dict)
        C = length(clusters)
        Bcluster = Dict{Int, Vector{Float64}}()
        Jcluster = Dict{Tuple{Int,Int}, SparseMatrixCSC{Float64,Int}}()
        # compute traceless basis lengths
        basis_len = Dict(ci => 4^length(clusters[ci]) - 1 for ci in 1:C)


        # -----------------------------
        # 1. Local fields
        # -----------------------------
        for (ci, cluster) in enumerate(clusters)
            Bcluster[ci] = zeros(Float64, basis_len[ci])
            for (si, (hx, hy, hz)) in zip(cluster, eachrow(Bvec[cluster, :]))
                abs(hx) > 0 && add_single_field!(Bcluster[ci], cluster, si, :x, hx)
                abs(hy) > 0 && add_single_field!(Bcluster[ci], cluster, si, :y, hy)
                abs(hz) > 0 && add_single_field!(Bcluster[ci], cluster, si, :z, hz)
            end
        end

        # -----------------------------
        # 2. Intra-cluster couplings
        # -----------------------------
        for (ci, cluster) in enumerate(clusters)
            for a = 1:length(cluster)-1
                for b = a+1:length(cluster)
                    si, sj = cluster[a], cluster[b]
                    key = si < sj ? (si, sj) : (sj, si)
                    if !haskey(Jdict, key)
                        continue
                    end
                    Jij = Jdict[key]
                    # Jij is a Dict like :xx=>val, :xy=>val, ...
                    for (ksym, val) in Jij
                        # ksym is Symbol like :xx, :xy, ...
                        str = String(ksym)
                        pa = (str[1] == 'x' ? :x : str[1] == 'y' ? :y : :z)
                        pb = (str[2] == 'x' ? :x : str[2] == 'y' ? :y : :z)
                        add_two_site_intra!(Bcluster[ci], cluster, si, pa, sj, pb, val)
                    end
                end
            end
        end

        # -----------------------------
        # 3. Inter-cluster couplings
        # -----------------------------
        for ci in 1:C-1
            for cj in ci+1:C
                leni = basis_len[ci]; lenj = basis_len[cj]
                rows, cols, vals = Int[], Int[], Float64[]
                cluster_i, cluster_j = clusters[ci], clusters[cj]
                for si in cluster_i, sj in cluster_j
                    key = si < sj ? (si,sj) : (sj,si)
                    if !haskey(Jdict, key)
                        continue
                    end
                    Jij = Jdict[key]
                    for (ksym, val) in Jij
                        s = String(ksym)
                        pa = s[1]=='x' ? :x : s[1]=='y' ? :y : :z
                        pb = s[2]=='x' ? :x : s[2]=='y' ? :y : :z

                        digits_i = zeros(Int, length(cluster_i))
                        digits_j = zeros(Int, length(cluster_j))

                        posi = findfirst(==(si), cluster_i)
                        posj = findfirst(==(sj), cluster_j)

                        digits_i[posi] = PAULI_DIGIT[pa]
                        digits_j[posj] = PAULI_DIGIT[pb]

                        idxi = digits_to_index(digits_i)
                        idxj = digits_to_index(digits_j)

                        push!(rows, idxi)
                        push!(cols, idxj)
                        push!(vals, val)
                    end
                end
                # build sparse matrix if any nonzero
                if isempty(vals)
                    Jcluster[(ci,cj)] = spzeros(Float64, leni, lenj)
                else
                    Jcluster[(ci,cj)] = sparse(rows, cols, vals, leni, lenj)
                end
            end
        end

        return Bcluster, Jcluster
    end

end # ClusterMapping