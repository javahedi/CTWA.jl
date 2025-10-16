module CTWA

    # Export main functionality from submodules
    export Model, Clustering, ClusterMapping, PhaseSpace, Dynamics, Observables

    # Include submodules in dependency order
    include("Model.jl")
    include("Clustering.jl")
    include("ClusterMapping.jl")
    include("PhaseSpace.jl")
    include("Dynamics.jl")
    include("Observables.jl")


     # Re-export specific functions from submodules
     using .Model: ModelParams, generate_model, build_Jdict, print_model_summary
     using .Clustering: naive_clustering, rg_clustering, print_clusters
     using .ClusterMapping: microscopic_to_cluster, add_single_field!, add_two_site_intra!
     using .PhaseSpace: PHASE_POINTS, single_spin_state_vec, sample_single_spin, sample_cluster
     using .Dynamics: compute_f_tensor, f_tensor_mul, compute_cluster_dHdx, cluster_eom!, evolve_cluster
     using .Observables: single_spin_expectation, two_point_expectation, delta_S


     export ModelParams, generate_model, build_Jdict, print_model_summary
     export naive_clustering, rg_clustering, print_clusters
     export microscopic_to_cluster, add_single_field!, add_two_site_intra!
     export PHASE_POINTS, single_spin_state_vec, sample_single_spin, sample_cluster
     export compute_f_tensor, f_tensor_mul, compute_cluster_dHdx, cluster_eom!, evolve_cluster
     export single_spin_expectation, two_point_expectation, delta_S


end # module
