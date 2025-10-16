module Dynamics

    using CTWA
    using LinearAlgebra
    using SparseArrays
    using OrdinaryDiffEq 
    using DifferentialEquations

    export compute_f_tensor, f_tensor_mul, compute_cluster_dHdx
    export cluster_eom!, evolve_cluster, evolve_cluster_ensemble

    # ==============================================================
    #  Compute structure constants f_pqr from traceless Pauli basis
    # ==============================================================

    """
        compute_f_tensor(basis::Vector{<:AbstractMatrix})

    Compute the structure constants `f_pqr` defined by

        [X_p, X_q] = i f_pqr X_r

    for a given traceless Hermitian operator basis `basis`.
    The result is returned as a sparse list of nonzero entries:

        f_list = Vector{Tuple{Int,Int,Int,Float64}}

    where each tuple corresponds to (p, q, r, value).
    """
    function compute_f_tensor(basis::Vector{<:AbstractMatrix})
        dim = length(basis)                   # number of basis elements = 4^n - 1
        n = round(Int, log(dim + 1) / log(4)) # cluster size (number of spins)
        f_list = Tuple{Int,Int,Int,Float64}[]

        for p in 1:dim, q in 1:dim
            comm = 1im * (basis[p]*basis[q] - basis[q]*basis[p])
            for r in 1:dim
                val = real(tr(comm' * basis[r])) / (2.0^n)
                if abs(val) > 1e-12
                    push!(f_list, (p, q, r, val))
                end
            end
        end

        return f_list
    end


    # ==============================================================
    #  Classical EOM using precomputed f_list
    # ==============================================================

    """
        f_tensor_mul(x, dHdx, f_list)

    Compute dx/dt = f_pqr dHdx[q] x[r] using precomputed `f_list`.
    """
    function f_tensor_mul(x::AbstractVector{Float64}, 
                        dHdx::AbstractVector{Float64}, 
                        f_list::Vector{Tuple{Int,Int,Int,Float64}})
        dx = zeros(Float64, length(x))
        @inbounds for (p, q, r, val) in f_list
            dx[p] += val * dHdx[q] * x[r]
        end
        return dx
    end



    # ==============================================================
    #  Effective cluster Hamiltonian gradient
    # ==============================================================

    """
        compute_cluster_dHdx(ci, x, Bcluster, Jcluster, cluster_sizes)

    Compute ∂H/∂x for cluster `ci`, including inter-cluster couplings.
    """
    function compute_cluster_dHdx(ci::Int, x::Vector{Float64},
                                Bcluster::Dict{Int,Vector{Float64}},
                                Jcluster::Dict{Tuple{Int,Int},SparseMatrixCSC{Float64,Int}},
                                cluster_sizes::Vector{Int})

        offset = sum(cluster_sizes[1:ci-1])
        n = cluster_sizes[ci]
        xi = @view x[offset+1 : offset+n]

        dHdx = copy(Bcluster[ci])

        for cj in eachindex(cluster_sizes)
            cj == ci && continue
            offsetj = sum(cluster_sizes[1:cj-1])
            xj = @view x[offsetj+1 : offsetj+cluster_sizes[cj]]
            key = (min(ci,cj), max(ci,cj))
            if haskey(Jcluster, key)
                Jij = Jcluster[key]
                if ci < cj
                    dHdx .+= Jij * xj
                else
                    dHdx .+= Jij' * xj
                end
            end
        end

        return dHdx
    end


    # ==============================================================
    #  Cluster equations of motion
    # ==============================================================

    """
        cluster_eom!(du, u, p, t)

    Cluster equations of motion using one global structure tensor `f_list`.

    `p` should contain:
    - `B::Dict{Int,Vector{Float64}}`
    - `J::Dict{Tuple{Int,Int},SparseMatrixCSC{Float64,Int}}`
    - `cluster_sizes::Vector{Int}`
    - `f_list::Vector{Tuple{Int,Int,Int,Float64}}`
    """
    function cluster_eom!(du, u, p, t)
        Bcluster, Jcluster, cluster_sizes, f_list = p.B, p.J, p.cluster_sizes, p.f_list
        offset = 0
        for ci in eachindex(cluster_sizes)
            n = cluster_sizes[ci]
            xi = @view u[offset+1 : offset+n]
            dHdx = compute_cluster_dHdx(ci, u, Bcluster, Jcluster, cluster_sizes)
            du[offset+1 : offset+n] .= f_tensor_mul(xi, dHdx, f_list)
            offset += n
        end
    end


    # ==============================================================
    #  Time evolution wrapper
    # ==============================================================

    """
        evolve_cluster(u0, Bcluster, Jcluster, cluster_sizes, f_list;
                    tspan=(0.0, 10.0), solver=Tsit5(), kwargs...)

    Integrate the classical cluster EOM with the given initial conditions.
    """
    function evolve_cluster(u0::Vector{Float64}, 
                            Bcluster::Dict{Int,Vector{Float64}}, 
                            Jcluster::Dict{Tuple{Int,Int},SparseMatrixCSC{Float64,Int}},
                            cluster_sizes::Vector{Int}, 
                            f_list::Vector{Tuple{Int,Int,Int,Float64}};
                            tspan=(0.0, 4.0), solver=Tsit5(), kwargs...)

        params = (; B=Bcluster, J=Jcluster, cluster_sizes, f_list)
        prob = ODEProblem(cluster_eom!, u0, tspan, params)
        return solve(prob, solver; kwargs...)
    end



    # function evolve_cluster_ensemble(u0s::Vector{Vector{Float64}}, 
    #                                  Bcluster::Dict{Int,Vector{Float64}}, 
    #                                 Jcluster::Dict{Tuple{Int,Int},SparseMatrixCSC{Float64,Int}}, 
    #                                 cluster_sizes::Vector{Int},  
    #                                 f_list::Vector{Tuple{Int,Int,Int,Float64}};
    #                                 tspan=(0.0,10.0), 
    #                                 solver=Tsit5(), 
    #                                 saveat=nothing)

    #     params = (; B=Bcluster, J=Jcluster, cluster_sizes, f_list)



    #     # Create ODEProblem for each initial condition
    #     prob = ODEProblem(cluster_eom!, u0s[1], tspan, params)
    #     # Define problem function for ensemble
    #     function prob_func(prob, i, repeat)
    #         remake(prob, u0=u0s[i])
    #     end

    #     # EnsembleProblem from vector of problems
    #     ep = EnsembleProblem(prob, prob_func=prob_func)
    #     return solve(ep, solver,
    #                 reltol=1e-8, abstol=1e-8,
    #                 saveat=saveat,
    #                 EnsembleThreads(),
    #                 trajectories=length(u0s))
    # end
    function evolve_cluster_ensemble(u0s::Vector{Vector{Float64}}, 
                                    Bcluster::Dict{Int,Vector{Float64}}, 
                                    Jcluster::Dict{Tuple{Int,Int},SparseMatrixCSC{Float64,Int}}, 
                                    cluster_sizes::Vector{Int},  
                                    f_list::Vector{Tuple{Int,Int,Int,Float64}};
                                    tspan=(0.0,10.0), 
                                    solver=Tsit5(), 
                                    saveat=nothing)

        params = (; B=Bcluster, J=Jcluster, cluster_sizes, f_list)

        # Create base ODEProblem
        prob = ODEProblem(cluster_eom!, u0s[1], tspan, params)

        # Ensemble problem function
        prob_func(prob, i, repeat) = remake(prob, u0=u0s[i])

        # EnsembleProblem
        ep = EnsembleProblem(prob, prob_func=prob_func)

        return solve(ep, solver,
                    reltol=1e-8, abstol=1e-8,
                    saveat=saveat,
                    EnsembleThreads(),
                    trajectories=length(u0s))

       
    end




end # module
