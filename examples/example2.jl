using CTWA
using LinearAlgebra
using Plots
using Random
using Statistics
using Dates
using DifferentialEquations

# -----------------------------
# System parameters
# -----------------------------
L = 16                 # Total number of spins
α = 1.0                # Disorder exponent for J distribution
cluster_size = 2       # Cluster size for CTWA
tspan = (0.0, 20.0)
t0, tf = tspan
n_points = 21
#saveat = t0 .+ (tf-t0) .* 10 .^ range(0,1,length=n_points) ./ 10
saveat = range(t0, tf, length=n_points)
Ntraj = 1000            # CTWA trajectories per disorder realization
Ndisorder = 10         # Number of disorder realizations
subsystem_indices = 1:(L÷2)  # Half-chain for entanglement entropy

# -----------------------------
# Helper functions
# -----------------------------
function sample_disorder(L, α)
    u = rand(L-1)
    return u.^(1/α)
end

function traceless_basis(n)
    σ = [Matrix{ComplexF64}(I(2)), [0 1;1 0], [0 -im; im 0], [1 0;0 -1]]
    basis = Vector{Matrix{ComplexF64}}()
    for digits in Iterators.product(ntuple(_->0:3, n)...)
        if all(x->x==0, digits)
            continue
        end
        op = σ[digits[1]+1]
        for k in 2:n
            op = kron(op, σ[digits[k]+1])
        end
        push!(basis, op)
    end
    return basis
end

"""
Compute connected correlation matrix ⟨Sᵅᵢ Sᵅⱼ⟩ - ⟨Sᵅᵢ⟩⟨Sᵅⱼ⟩
for all pairs of sites and a given spin component (α = x, y, z).

Inputs:
- S_alpha :: Matrix{Float64}  (Ntraj × L)
  Expectation values ⟨Sᵅ⟩ per trajectory and site.

Returns:
- C :: Symmetric{Float64,Matrix{Float64}}  (L×L)
  Connected correlation matrix.
"""
function correlation_matrix(S_alpha::Matrix{Float64})
    L = size(S_alpha, 2)
    C = zeros(L, L)
    for i in 1:L, j in i:L
        mean_ij = mean(S_alpha[:, i] .* S_alpha[:, j])
        mean_i  = mean(S_alpha[:, i])
        mean_j  = mean(S_alpha[:, j])
        Cij = mean_ij - mean_i * mean_j
        C[i, j] = C[j, i] = Cij
    end
    return Symmetric(C)
end


"""
Compute an entropy-like measure from correlation matrix C.
Interpreted as information content (proxy for entanglement).
"""
function entanglement_entropy(C::AbstractMatrix)
    vals = real(eigvals(Matrix(C)))
    # Normalize and clamp eigenvalues
    vals = clamp.(abs.(vals) ./ maximum(abs.(vals)), 1e-12, 1 - 1e-12)
    return -sum(vals .* log.(vals) + (1 .- vals) .* log.(1 .- vals))
end


"""
Compute average entropy-like measure for x, y, z components
restricted to a given subsystem (e.g., half chain).
"""
function half_chain_entropy(Sx::Matrix, Sy::Matrix, Sz::Matrix)
    Cx = correlation_matrix(Sx)
    Cy = correlation_matrix(Sy)
    Cz = correlation_matrix(Sz)
    return (entanglement_entropy(Cx) +
            entanglement_entropy(Cy) +
            entanglement_entropy(Cz)) / 3
end


# # Compute single-site + two-site correlators and Rényi-2 entropy
# function S2_two_sites_trajectories(Sx_mat::Matrix{Float64}, 
#                                     Sy_mat::Matrix{Float64}, 
#                                     Sz_mat::Matrix{Float64})
#     Ntraj, L = size(Sx_mat)
#     S2_vals = zeros(L*(L-1) ÷ 2)
#     idx = 1
#     for i in 1:L-1
#         for j in i+1:L
#             sumsq = 0.0
#             for α in 0:3
#                 for β in 0:3
#                     x_i = α==1 ? Sx_mat[:,i] : α==2 ? Sy_mat[:,i] : α==3 ? Sz_mat[:,i] : ones(Ntraj)
#                     x_j = β==1 ? Sx_mat[:,j] : β==2 ? Sy_mat[:,j] : β==3 ? Sz_mat[:,j] : ones(Ntraj)
#                     sumsq += mean( (x_i .* x_j).^2 )  # average of squared product
#                 end
#             end
#             S2_vals[idx] = 2 - log2(sumsq)
#             idx += 1
#         end
#     end
#     return mean(S2_vals)
# end



function renyi2_average(Sx_mat::Matrix{Float64},
                        Sy_mat::Matrix{Float64},
                        Sz_mat::Matrix{Float64})
    """
    Compute average two-site Rényi-2 entropy ⟨S₂⟩ from CTWA trajectory data.
    Inputs:
        Sx_mat, Sy_mat, Sz_mat :: Matrix{Float64} (Ntraj × L)
    Returns:
        Float64 : average over all site pairs
    """
    Ntraj, L = size(Sx_mat)
    S2_vals = zeros(Float64, L * (L - 1) ÷ 2)
    idx = 1

    for i in 1:L-1
        for j in i+1:L
            sxi = mean(Sx_mat[:, i]); syi = mean(Sy_mat[:, i]); szi = mean(Sz_mat[:, i])
            sxj = mean(Sx_mat[:, j]); syj = mean(Sy_mat[:, j]); szj = mean(Sz_mat[:, j])

            spi = sxi + im * syi
            smj = sxj - im * syj

            val = 0.25 * (1.0 + szi^2 + szj^2 + (szi * szj)^2 + 8.0 * abs(spi * smj)^2)
            S2_vals[idx] = -log2(val)
            idx += 1
        end
    end

    return mean(S2_vals)
end


# -----------------------------
# Storage
# -----------------------------
S_half_dict = zeros(n_points)
S2_dict = zeros(n_points)


 # Cluster basis and f_list
cluster_basis = traceless_basis(cluster_size)
f_list        = compute_f_tensor(cluster_basis)


# -----------------------------
# Main loop: disorder realizations
# -----------------------------
for disorder in 1:Ndisorder
    println("\n=== Disorder realization $disorder ===")
    Jlist = sample_disorder(L, α)
    
    # Construct Hamiltonian
    Bx = zeros(L); By = zeros(L); Bz = zeros(L)
    Jdict = Dict{Tuple{Int,Int}, Dict{Symbol,Float64}}()
    for i in 1:L-1
        Jdict[(i,i+1)] = Dict(:xx=>Jlist[i], :yy=>Jlist[i], :zz=>Jlist[i])
    end

    # Initial state: |+x>
    #state_list = fill(:plusz, L)
    # neel 
    state_list = [isodd(i) ? :plusz : :minusz for i in 1:L]


    # Clustering
    #clusters = naive_clustering(L, cluster_size)
    clusters = rg_clustering( L, Jdict, cluster_size)
    print_clusters(clusters)
    Bmat = hcat(Bx, By, Bz)
    Bcluster, Jcluster = microscopic_to_cluster(clusters, Bmat, Jdict)
    cluster_len = [length(Bcluster[ci]) for ci in 1:length(clusters)]

   
    # Generate Ntraj initial conditions
    u0s = [vcat([sample_cluster(cl, state_list) for cl in clusters]...) for _ in 1:Ntraj]

    # Solve ensemble
    sols = evolve_cluster_ensemble(u0s, Bcluster, Jcluster, cluster_len, f_list; tspan=tspan, saveat=saveat)

    # Collect half-chain observables
    for (ti, t) in enumerate(saveat)
        Sx_mat = zeros(Ntraj, length(subsystem_indices))
        Sy_mat = zeros(Ntraj, length(subsystem_indices))
        Sz_mat = zeros(Ntraj, length(subsystem_indices))
        for (traj_idx, sol) in enumerate(sols.u)
            offset = 0
            for (ci, cl) in enumerate(clusters)
                n_spin = length(cl)
                for (s_idx, s) in enumerate(cl)
                    global_idx = offset + s_idx
                    if global_idx in subsystem_indices
                        Sx_mat[traj_idx, global_idx - subsystem_indices[1] + 1] = sol[ti][offset + 3*(s_idx-1) + 1]
                        Sy_mat[traj_idx, global_idx - subsystem_indices[1] + 1] = sol[ti][offset + 3*(s_idx-1) + 2]
                        Sz_mat[traj_idx, global_idx - subsystem_indices[1] + 1] = sol[ti][offset + 3*(s_idx-1) + 3]
                    end
                end
                offset += cluster_len[ci]
            end
        end
        S_half_dict[ti] += half_chain_entropy(Sx_mat, Sy_mat, Sz_mat)
        S2_dict[ti]     += renyi2_average(Sx_mat, Sy_mat, Sz_mat)
    end

end

# Average over disorder realizations
S_half_dict ./= Ndisorder
S2_dict     ./=Ndisorder

@show  S_half_dict, S2_dict
# -----------------------------
# Plot
# -----------------------------
plot(saveat, S_half_dict;
     lw=2,
     label="Half-chain entanglement",
     #xscale=:log10,   # logarithmic x-axis
     xlabel="t",
     ylabel="S_half",
     title="Disordered NN Heisenberg chain, L=$L, α=$α")
savefig("example2_entanglement.png")


plot(saveat, S2_dict;
     lw=2,
     label="Average two-site Rényi-2",
     #xscale=:log10,   # logarithmic x-axis
     xlabel="t",
     ylabel="⟨S₂(t)⟩",
     title="Disordered NN Heisenberg chain, L=$L, α=$α")
savefig("example2_Rényi_.png")


