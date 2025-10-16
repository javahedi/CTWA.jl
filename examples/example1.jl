using CTWA
using LinearAlgebra
using Plots

# -----------------------------
#  System parameters
# -----------------------------
N = 10               # Number of spins
α = 1.5              # Long-range power-law exponent
tspan = (0.0, 10.0)  # Time interval
cluster_sizes_list = [2, 3, 4]  # Cluster sizes to test

# -----------------------------
#  Construct long-range Ising Hamiltonian
# -----------------------------
Bx = zeros(N)
By = zeros(N)
Bz = zeros(N)

Jdict = Dict{Tuple{Int,Int}, Dict{Symbol,Float64}}()
for i in 1:N-1
    for j in i+1:N
        Jdict[(i,j)] = Dict(:zz => 1.0 / abs(i-j)^α)
    end
end

model = ModelParams(N, Bx, By, Bz, Jdict)

# -----------------------------
#  Initial state: all |+x>
# -----------------------------
state_list = fill(:plusx, N)

# -----------------------------
#  Loop over cluster sizes
# -----------------------------
for clust_size in cluster_sizes_list
    println("\n=== Running CTWA with cluster size = $clust_size ===")

    # 1. Naive clustering
    clusters = naive_clustering(N, clust_size)
    print_clusters(clusters)

    # 2. Cluster mapping
    Bmat = hcat(Bx, By, Bz)  # N x 3 matrix
    Bcluster, Jcluster = microscopic_to_cluster(clusters, Bmat, Jdict)

    # 3. Compute cluster basis sizes
    cluster_len = [length(Bcluster[ci]) for ci in 1:length(clusters)]

    # 4. Construct traceless Pauli basis for cluster size
    #    (4^n - 1 matrices of size 2^n x 2^n)
    n = clust_size
    dim = 4^n - 1
    # Generate Pauli basis matrices for n spins
    function pauli_matrix(p)
        σ = [Matrix{ComplexF64}(I(2)), [0 1;1 0], [0 -im; im 0], [1 0;0 -1]]
        return σ[p+1]  # 0=I, 1=X, 2=Y, 3=Z
    end

    function traceless_basis(n)
        basis = Vector{Matrix{ComplexF64}}()
        for digits in Iterators.product(ntuple(_->0:3, n)...)
            if all(x->x==0, digits)
                continue  # skip identity
            end
            op = pauli_matrix(digits[1])
            for k in 2:n
                op = kron(op, pauli_matrix(digits[k]))
            end
            push!(basis, op)
        end
        return basis
    end

    cluster_basis = traceless_basis(n)

    # 5. Compute structure constants
    println("Computing f_list for cluster size $n ...")
    f_list = compute_f_tensor(cluster_basis)

    # 6. Sample initial phase-space vector u0
    u0 = Float64[]
    for ci in 1:length(clusters)
        append!(u0, sample_cluster(clusters[ci], state_list))
    end

    # 7. Integrate cluster EOM
    println("Evolving clusters ...")
    sol = evolve_cluster(u0, Bcluster, Jcluster, cluster_len, f_list, tspan=tspan)

    # 8. Compute <Mx(t)> = 1/N sum_i <σx_i>(t)
    # Single-spin σx components are first 3 entries per spin in cluster vectors
    avg_mx = Float64[]
    times = sol.t
    offset = 0
    for ti in 1:length(times)
        x_sum = 0.0
        spin_idx = 1
        offset = 0
        for (ci, cl) in enumerate(clusters)
            n_spin = length(cl)
            # For simplicity, assume σx components are first n_spin entries
            # This depends on how sample_cluster packs single-spin σx/y/z; here we take first component of each spin
            for s in 1:n_spin
                # single-spin σx is first entry of spin vector in cluster
                x_val = sol.u[ti][offset + 3*(s-1) + 1]  # σx
                x_sum += x_val
            end
            offset += cluster_len[ci]
        end
        push!(avg_mx, x_sum / N)
    end

    # 9. Plot
    plot(times, avg_mx, label="Cluster size $clust_size", xlabel="t", ylabel="<Mx(t)>")
end

