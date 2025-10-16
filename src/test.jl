using LinearAlgebra
using SparseArrays
using Combinatorics
using ..Dynamics  # assuming Dynamic.jl is loaded as a module

# -----------------------------
# 1. Generate basis for n=2 cluster
# -----------------------------
n = 2
basis = [ComplexF64.(b) for b in Dynamics.generate_traceless_basis(n)] 
dim = length(basis)
println("Cluster n=$n, traceless basis size = $dim")

# -----------------------------
# 2. Compute structure constants
# -----------------------------

f_pqr = Dynamics.compute_f_tensor(basis)
println("Number of nonzero f_pqr entries = ", length(f_pqr))

# -----------------------------
# 3. Test some basic commutators
# -----------------------------
# Map basis indices to simple labels: X1=1, Y1=2, Z1=3, X2=4, Y2=5, Z2=6,... for easier check
# We'll just print a few key commutators
println("\nCheck intra-spin commutators (spin 1):")
for (p,q,r) in [(1,2,3), (2,1,3),(2,3,1),(3,1,2)]
    val = get(f_pqr,(p,q,r),0.0)
    println("f[$p,$q,$r] = $val (should be ±2)")
end

println("\nCheck inter-spin commutator (spin1 vs spin2, should vanish):")
# e.g., X1 vs Y2
p,q = 1,5
any_r = [get(f_pqr,(p,q,r),0.0) for r in 1:dim]
println("f[1,5,:] = ", any_r, " (all zeros expected)")

# -----------------------------
# 4. Optional: small test with f_tensor_mul
# -----------------------------
x = randn(dim)
dHdx = randn(dim)
dx = Dynamics.f_tensor_mul(x,dHdx,f_pqr)
println("\ndx computed with f_tensor_mul (length = $(length(dx)))")
