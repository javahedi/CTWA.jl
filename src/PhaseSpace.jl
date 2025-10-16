module PhaseSpace

    using CTWA
    using Combinatorics
    using Random

    export PHASE_POINTS, single_spin_state_vec, sample_single_spin, sample_cluster

    # ==============================================================
    #  Phase-point table (discrete Wigner representation)
    # ==============================================================

    """
        PHASE_POINTS

    Discrete phase-space representation for single-spin (qubit) Wigner points.

    Each key corresponds to a **qubit orientation index** (1 or 2), and the
    inner dictionaries give the 3-component phase-space coordinates for the
    four Pauli configurations `(p, q) ∈ {0,1}²`.

    This is used for sampling symbolic cluster states in a quasi-probabilistic
    representation (e.g., in the Truncated Wigner Approximation for spins).

    Reference:  
    - Wootters, *Ann. Phys.* **176**, 1–21 (1987).  
    - Schachenmayer *et al.*, *NJP* **17**, 065009 (2015).
    """
    const PHASE_POINTS = Dict(
        1 => Dict(
            (0,0)=>[ 1,  1,  1],
            (0,1)=>[-1, -1,  1],
            (1,0)=>[ 1, -1, -1],
            (1,1)=>[-1,  1, -1]
        ),
        2 => Dict(
            (0,0)=>[ 1, -1,  1],
            (0,1)=>[-1,  1,  1],
            (1,0)=>[ 1,  1, -1],
            (1,1)=>[-1, -1, -1]
        )
    )


    # ==============================================================
    #  Single-spin density vectors
    # ==============================================================

    """
        single_spin_state_vec(state::Symbol) -> Vector{Float64}

    Returns the **mean Bloch vector** `[⟨σx⟩, ⟨σy⟩, ⟨σz⟩]` corresponding to
    a symbolic spin state.

    Supported symbolic states:
    - `:plusx`, `:minusx` → ±x-polarized
    - `:plusy`, `:minusy` → ±y-polarized
    - `:plusz`, `:minusz` → ±z-polarized
    - `:thermal` → random ±1 in each component (maximally mixed)

    Example:
    ```julia
    single_spin_state_vec(:plusz)  # -> [0.0, 0.0, 1.0]
    ````

    """
    function single_spin_state_vec(state::Symbol)
        if state == :plusx
            return [1.0, 0.0, 0.0]
        elseif state == :minusx
            return [-1.0, 0.0, 0.0]
        elseif state == :plusy
            return [0.0, 1.0, 0.0]
        elseif state == :minusy
            return [0.0, -1.0, 0.0]
        elseif state == :plusz
            return [0.0, 0.0, 1.0]
        elseif state == :minusz
            return [0.0, 0.0, -1.0]
        elseif state == :thermal
            return [rand([-1.0,1.0]), rand([-1.0,1.0]), rand([-1.0,1.0])]
        else
        error("Unknown single-spin state: $state")
        end
    end

    # ==============================================================

    # Sample single-spin from symbolic Bloch vector

    # ==============================================================

    """
    sample_single_spin(rvec::Vector{Float64}) -> Vector{Float64}

    Samples a single-spin phase-space point from a Bloch vector `rvec`.

    * If `rvec == [0,0,0]`, returns a fully random ±1 vector (thermal).
    * Otherwise, for any zero components, samples ±1 randomly.

    Example:

    ```julia
    sample_single_spin([1.0, 0.0, 0.0])  # -> [1.0, ±1.0, ±1.0]
    ```

    """
    function sample_single_spin(rvec::Vector{Float64})
        if rvec == [0.0, 0.0, 0.0]
            return [rand([-1.0,1.0]), rand([-1.0,1.0]), rand([-1.0,1.0])]
        end

        sampled = Float64[]
        for val in rvec
            if val == 0.0
                push!(sampled, rand([-1.0,1.0]))
            else
                push!(sampled, val)
            end
        end
        return sampled

    end

    # ==============================================================

    # Cluster-level sampling

    # ==============================================================

    """
    sample_cluster(cluster::Vector{Int}, state_list::Vector{Symbol}) -> Vector{Float64}

    Constructs the **full phase-space vector** for a given cluster of spins.

    Each spin is sampled according to its symbolic state (`state_list[i]`).
    The output vector includes:

    1. Single-spin components (⟨σx⟩, ⟨σy⟩, ⟨σz⟩)
    2. All higher-order tensor-product terms (2-spin, 3-spin, … up to n-spin)

    Example:


    cluster = [1,2,3]
    state_list = [:plusx, :plusx, :plusx]
    r_cluster = sample_cluster(cluster, state_list)


    """
    function sample_cluster(cluster::Vector{Int}, state_list::Vector{Symbol})
        n = length(cluster)
        r_spin_list = Vector{Vector{Float64}}(undef, n)

        for (i, site) in enumerate(cluster)
            rvec = single_spin_state_vec(state_list[site])
            r_spin_list[i] = sample_single_spin(rvec)
        end

        cluster_vec = Float64[]

        # 1. Single-spin operators
        for r in r_spin_list
            append!(cluster_vec, r)
        end

        # 2. All tensor-product terms (2-spin, 3-spin, ..., n-spin)
        for k in 2:n
            for subset in combinations(1:n, k)
                tvec = [1.0]
                for idx in subset
                    tvec = vec(kron(tvec, r_spin_list[idx]))
                end
                append!(cluster_vec, tvec)
            end
        end

        return cluster_vec
    end


end # module


