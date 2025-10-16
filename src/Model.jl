module Model

    using LinearAlgebra
    using Random
    using SparseArrays

    export ModelParams, generate_model, build_Jdict, print_model_summary

    """
        struct ModelParams

    Holds the microscopic Hamiltonian parameters of a spin system.

    Fields:
    - `N::Int`: Number of spins.
    - `Bx, By, Bz::Vector{Float64}`: Local magnetic fields per site.
    - `Jdict::Dict{Tuple{Int,Int}, Dict{Symbol,Float64}}`: Pairwise coupling dictionary.
    """
    Base.@kwdef struct ModelParams
        N::Int
        Bx::Vector{Float64}
        By::Vector{Float64}
        Bz::Vector{Float64}
        Jdict::Dict{Tuple{Int,Int}, Dict{Symbol,Float64}}
    end


    # ----------------------------------------------------------
    #   Utility: Build Jdict from given coupling matrices
    # ----------------------------------------------------------
    """
        build_Jdict(Jx::AbstractMatrix, Jy::AbstractMatrix, Jz::AbstractMatrix) -> Dict

    Constructs a dictionary of pairwise couplings:
        Jdict[(i,j)] = Dict(:xx => Jx[i,j], :yy => Jy[i,j], :zz => Jz[i,j])
    Only upper-triangular (i<j) entries are stored.
    """
    function build_Jdict(Jx::AbstractMatrix, Jy::AbstractMatrix, Jz::AbstractMatrix)
        N = size(Jx, 1)
        Jdict = Dict{Tuple{Int,Int}, Dict{Symbol,Float64}}()
        for i in 1:N-1
            for j in i+1:N
                if Jx[i,j] != 0 || Jy[i,j] != 0 || Jz[i,j] != 0
                    Jdict[(i,j)] = Dict(
                        :xx => Jx[i,j],
                        :yy => Jy[i,j],
                        :zz => Jz[i,j]
                    )
                end
            end
        end
        return Jdict
    end


    # ----------------------------------------------------------
    #   Random / custom model generation
    # ----------------------------------------------------------
    """
        generate_model(N::Int; B::Tuple{Float64,Float64,Float64}=(0,0,1), 
                    J::Tuple{Float64,Float64,Float64}=(1,1,1),
                    topology::Symbol=:chain, seed::Int=0)

    Generates a `ModelParams` object with specified system size and coupling structure.

    # Arguments
    - `N`: Number of spins.
    - `B`: Tuple (Bx,By,Bz) of uniform local fields.
    - `J`: Tuple (Jx,Jy,Jz) of coupling strengths.
    - `topology`: `:chain`, `:all2all`, or `:random`.
    - `seed`: Optional RNG seed.

    # Returns
    `ModelParams`
    """
    function generate_model(N::Int; 
            B::Tuple{Float64,Float64,Float64}=(0.0, 0.0, 1.0),
            J::Tuple{Float64,Float64,Float64}=(1.0, 1.0, 1.0),
            topology::Symbol=:chain,
            seed::Int=0)

        seed != 0 && Random.seed!(seed)

        Bx = fill(B[1], N)
        By = fill(B[2], N)
        Bz = fill(B[3], N)

        Jx = zeros(N, N)
        Jy = zeros(N, N)
        Jz = zeros(N, N)

        if topology == :chain
            for i in 1:N-1
                Jx[i, i+1] = J[1]
                Jy[i, i+1] = J[2]
                Jz[i, i+1] = J[3]
            end
        elseif topology == :all2all
            for i in 1:N-1, j in i+1:N
                Jx[i,j] = J[1]
                Jy[i,j] = J[2]
                Jz[i,j] = J[3]
            end
        elseif topology == :random
            for i in 1:N-1, j in i+1:N
                if rand() < 0.5
                    Jx[i,j] = J[1] * (2rand() - 1)
                    Jy[i,j] = J[2] * (2rand() - 1)
                    Jz[i,j] = J[3] * (2rand() - 1)
                end
            end
        else
            error("Unknown topology: $topology")
        end

        Jdict = build_Jdict(Jx, Jy, Jz)

        return ModelParams(N, Bx, By, Bz, Jdict)
    end


    # ----------------------------------------------------------
    #   Summary Printer
    # ----------------------------------------------------------
    function print_model_summary(model::ModelParams)
        println("Spin Model Summary")
        println("==================")
        println("N = $(model.N)")
        println("Local field B = (Bx,By,Bz) = ($(mean(model.Bx)), $(mean(model.By)), $(mean(model.Bz)))")
        println("Number of couplings = $(length(model.Jdict))")
        for ((i,j), J) in first(collect(model.Jdict), min(5,length(model.Jdict)))
            println("J($i,$j) = ", J)
        end
        if length(model.Jdict) > 5
            println("... ($(length(model.Jdict)-5) more couplings)")
        end
        return nothing
    end

end # module
