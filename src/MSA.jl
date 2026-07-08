"""
    MSA

Multiple-scales / slow-manifold analysis of a settling, added-mass particle in a
deep-water surface wave (the decoupled Maxey–Riley problem).

All quantities are in the **`v_z` convention**: `Params.vz` is the terminal settling
velocity directly (not `St*θ`); the Maxey–Riley gravity term is `-vz/St`.

Every function takes a `Params` explicitly — there are no module-level globals, and
nothing in `src/` has side effects merely from `using MSA` (figure-makers in
`figures.jl` produce output only when CALLED).

`using MSA` alone re-exports the solver/plotting verbs you need (`solve`, `ODEProblem`,
`plot`, `contour`, `savefig`, `find_zero`, `quadgk`, the `L"..."` macro, ...) — no need
to separately `using DifferentialEquations, Plots`.
"""
module MSA

using Reexport
@reexport using DifferentialEquations
@reexport using Roots
@reexport using QuadGK
@reexport using Plots
@reexport using LaTeXStrings
using LinearAlgebra
using Statistics

include("params.jl")
include("dynamics.jl")
include("hamiltonian.jl")
include("recover.jl")
include("paths.jl")
include("figures.jl")

# --- public API ---
export Params
# dynamics (det-functions for ODEProblem)
export maxey_riley!, fluid_settling!, sigma_omega!, slow_manifold_xz!
# Hamiltonian / slow manifold
export amplitude, hamiltonian, xdot, eigendata,
       hamiltonian_B3, projection, ham_trafo
# lab-frame reconstruction from an isobar
export recover_xzt
# solver utilities
export ode_path_gen, path_distance_var,
       sigma_omega_ic, sigma_omega_solution, collapse_sigma_omega,
       recover_xz_from_sigma_omega
# figure-makers
export mr_fs_isobar_figure, compare_full_fs_ham, b3_contours, xz_contours,
       sigma_omega_contours, sigma_omega_recovery_figure, st_accuracy_figure,
       make_all_figures, figure_suffix

end # module
