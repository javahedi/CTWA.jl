using CTWA
using LinearAlgebra
using Plots
using Random
using Statistics
using Dates

# -----------------------------
# System parameters
# -----------------------------
N = 36                   # Number of spins
α = 3.0                  # Long-range power-law exponent

cluster_sizes_list = [2, 3]  # Cluster sizes to test
tspan = (0.0, 4.0)      # Time interval
n_points = 101
saveat = range(tspan[1], tspan[2], length=n_points)
Ntraj = 200              # Number of phase-space trajectories

# -----------------------------
# Construct long-range Ising Hamiltonian
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

# -----------------------------
# Initial state: all |+x>
# -----------------------------
state_list = fill(:plusx, N)

# -----------------------------
# Prepare storage for observables
# -----------------------------
avg_mx_dict = Dict{Int, Vector{Float64}}()
deltaSx_dict = Dict{Int, Vector{Float64}}()

# -----------------------------
# Loop over cluster sizes
# -----------------------------
for clust_size in cluster_sizes_list
    println("\n=== Running CTWA with cluster size = $clust_size ===")
    start_time = now()

    # 1. Naive clustering
    clusters = naive_clustering(N, clust_size)
    print_clusters(clusters)

    # 2. Cluster mapping
    Bmat = hcat(Bx, By, Bz)  # N x 3 matrix
    Bcluster, Jcluster = microscopic_to_cluster(clusters, Bmat, Jdict)
    cluster_len = [length(Bcluster[ci]) for ci in 1:length(clusters)]

    # 3. Construct traceless Pauli basis
    function pauli_matrix(p)
        σ = [Matrix{ComplexF64}(I(2)), [0 1;1 0], [0 -im; im 0], [1 0;0 -1]]
        return σ[p+1]
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

    cluster_basis = traceless_basis(clust_size)

    # 4. Compute structure constants
    println("Computing f_list for cluster size $clust_size ...")
    f_start = time()
    f_list = compute_f_tensor(cluster_basis)
    println("f_list computed in $(round(time() - f_start, digits=3)) seconds")

    # 5. Initialize accumulators for ensemble averages
    Sx_accum = zeros(n_points)
    Sx2_accum = zeros(n_points)

    println("Evolving $Ntraj trajectories ...")
    for traj in 1:Ntraj
        # Sample initial phase-space vector
        u0 = Float64[]
        for ci in eachindex(clusters)
            append!(u0, sample_cluster(clusters[ci], state_list))
        end

        # Integrate cluster EOM
        sol = evolve_cluster(u0, Bcluster, Jcluster, 
                            cluster_len, f_list, tspan=tspan, saveat=saveat)

        # Compute total Sx for each time step
        for ti in 1:n_points
            Sx = 0.0
            offset = 0
            for (ci, cl) in enumerate(clusters)
                n_spin = length(cl)
                for s in 1:n_spin
                    Sx += sol.u[ti][offset + 3*(s-1) + 1]  # σx
                end
                offset += cluster_len[ci]
            end
            Sx_accum[ti] += Sx
            Sx2_accum[ti] += Sx^2
        end
    end

    # 6. Compute <Sx>/N and ΔSx/N^2
    avg_mx_dict[clust_size] = Sx_accum ./ (Ntraj * N)
    deltaSx_dict[clust_size] = (Sx2_accum ./ Ntraj .- (Sx_accum ./ Ntraj).^2) ./ (N^2)

    println("Cluster size $clust_size done in $(now() - start_time)")
end

# -----------------------------
# Plot results
# -----------------------------
plot(layout=(1,2), size=(1100,300))

# Left: <Mx(t)>
for clust_size in cluster_sizes_list
    plot!(saveat, avg_mx_dict[clust_size], label="Cluster size $clust_size", subplot=1)
end
xlabel!(subplot=1, "t")
ylabel!(subplot=1, "<Mx(t)>")
title!(subplot=1, "CTWA <Mx>")

# Right: ΔSx(t)/N^2
for clust_size in cluster_sizes_list
    plot!(saveat, deltaSx_dict[clust_size], label="Cluster size $clust_size", subplot=2)
end
xlabel!(subplot=2, "t")
ylabel!(subplot=2, "ΔSx/N²")
title!(subplot=2, "CTWA ΔSx/N²")

savefig("example1_obs.png")
println("Plot saved as example1_obs.png")
