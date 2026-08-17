# MSA

Local Julia package for the multiple-scales / slow-manifold analysis of a settling,
added-mass particle in a deep-water surface wave (the decoupled Maxey–Riley problem).
Calculations with `sympy` in `calculating_asymp.{ipynb,html}`.

## Convention

Everything is in the **`v_z` convention**: `p.vz` is the terminal settling velocity
*directly* (not `St*θ`). The Maxey–Riley gravity term is `-vz/St`, so a particle
settles at exactly `vz` in still fluid, independent of `St`.

## Design rules

* `Params` (`St, eps, b, vz`) is immutable and physics-only.
* **Every function takes `p` explicitly.** There are no module-level globals.
* Nothing in `src/` runs merely from `using MSA` — figure-makers there produce output
  only when *called*, not at load time.
* Integration details (initial condition, tspan, det-function, callbacks) are passed
  as explicit arguments to the solver helpers — never stored on `p`.
* `using MSA` alone is enough: it re-exports `solve`, `ODEProblem`, `plot`, `contour`,
  `savefig`, `find_zero`, `quadgk`, `L"..."`, etc., so no separate
  `using DifferentialEquations, Plots` is needed.
* Every figure is one exported function in `src/figures.jl`, callable directly from any
  `using MSA` session. `make_all_figures()` rebuilds them all with their canonical
  parameter sets.

## Layout

```
src/
  MSA.jl          module: includes the below, reexports deps, exports the public API
  params.jl       Params struct
  dynamics.jl     det-functions: maxey_riley!, fluid_settling!, sigma_omega!, slow_manifold_xz!
  hamiltonian.jl  amplitude, hamiltonian, xdot, eigendata, + B3/σ-ω Hamiltonian helpers
  recover.jl      isobar -> (x,z,t) lab-frame reconstruction
  paths.jl        ode_path_gen, path_distance_var (double-avg error metric), σ-ω recovery
  figures.jl      one function per figure + make_all_figures (canonical parameter sets)
figures/          generated output (git-ignored)
```

## Usage

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using Revise, MSA        # Revise gives hot-reload; no more slow re-include / OOM loop

p = Params(St=0.5, eps=0.33, b=0.9, vz=0.05)
mr_fs_isobar_figure(p)   # everything is available from `using MSA` alone
```

Rebuild every figure at once — in a REPL, or as a fresh process:

```
julia --project=. -e 'using MSA; make_all_figures()'
```
