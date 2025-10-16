# CTWA

![CTWA Logo](assets/logo.png)

[![Build Status](https://github.com/javahedi/CTWA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/javahedi/CTWA.jl/actions/workflows/CI.yml?query=branch%3Amain)



**CTWA.jl** — Cluster Truncated Wigner Approximation for spin systems in Julia.


This package implements a **cluster-based Truncated Wigner Approximation (CTWA)** for simulating quantum spin dynamics, supporting arbitrary clusters, long-range interactions, and discrete phase-space sampling.  
The approach is described in:

> Braemer, A., Vahedi, J., & Gärtner, M. (2024). *Cluster truncated Wigner approximation for bond-disordered Heisenberg spin models*. Phys. Rev. B, 110(5), 054204. [DOI: 10.1103/PhysRevB.110.054204](https://link.aps.org/doi/10.1103/PhysRevB.110.054204)


---

## Features

- Define arbitrary **spin models** with local fields and pairwise couplings.
- Flexible **clustering** strategies:
  - Naive (block) clustering
  - RG-inspired clustering based on coupling strengths
- Map microscopic Hamiltonians to **cluster-level representations**.
- Sample cluster **phase-space vectors** using discrete Wigner representation.
- Compute **structure constants** for cluster operators.
- Integrate **cluster classical equations of motion** using `OrdinaryDiffEq.jl`.
- Compute observables like **magnetization** and other single- or multi-spin quantities.

---

## Installation

Clone the repository and activate the project:

```bash
git clone https://github.com/yourusername/CTWA.jl.git
cd CTWA.jl
julia --project=.
````

Then add required dependencies:

```julia
using Pkg
Pkg.instantiate()
```

---

## Usage

### Load the package

```julia
using CTWA
```

### 1. Define a spin model

```julia
N = 10
Bx = zeros(N)
By = zeros(N)
Bz = zeros(N)

# Example: long-range Ising couplings
α = 1.5
Jdict = Dict{Tuple{Int,Int}, Dict{Symbol,Float64}}()
for i in 1:N-1, j in i+1:N
    Jdict[(i,j)] = Dict(:zz => 1.0 / abs(i-j)^α)
end

model = ModelParams(N, Bx, By, Bz, Jdict)
print_model_summary(model)
```

---

### 2. Cluster the spins

```julia
cluster_size = 2
clusters = naive_clustering(N, cluster_size)
print_clusters(clusters)
```

Or use RG-inspired clustering:

```julia
clusters = rg_clustering(N, Jdict, cluster_size)
print_clusters(clusters)
```

---

### 3. Map to cluster Hamiltonian

```julia
Bmat = hcat(Bx, By, Bz)
Bcluster, Jcluster = microscopic_to_cluster(clusters, Bmat, Jdict)
```

---

### 4. Sample initial cluster phase-space vectors

```julia
state_list = fill(:plusx, N)  # all spins in |+x>
u0 = Float64[]
for ci in 1:length(clusters)
    append!(u0, sample_cluster(clusters[ci], state_list))
end
```

---

### 5. Compute structure constants

```julia
# Construct traceless Pauli basis for cluster
n = cluster_size
cluster_basis = traceless_basis(n)
f_list = compute_f_tensor(cluster_basis)
```

---

### 6. Integrate cluster dynamics

```julia
cluster_len = [length(Bcluster[ci]) for ci in 1:length(clusters)]
sol = evolve_cluster(u0, Bcluster, Jcluster, cluster_len, f_list, tspan=(0.0,10.0))
```

---

### 7. Compute observables

```julia
# Example: average magnetization in x
avg_mx = mean([sol.u[t][1:3:end] for t in 1:length(sol.t)])
```

---

## Example scripts

See the `examples/` folder:

* `example1.jl` — long-range Ising model with cluster sizes 2, 3, 4.
* Additional examples can be added for different Hamiltonians and observables.

---

## References

* Wootters, *Ann. Phys.* **176**, 1–21 (1987) — Discrete Wigner function.
* Polkovnikov, *Ann. Phys.* **325**, 1790–1852 (2010) — Phase-space methods for quantum dynamics.
* J. Wurtz, et, al. Ann. Phys. **395**, 341 (2018) - Cluster truncated Wigner approximation in strongly interacting systems.



---

## License

[MIT](LICENSE)




