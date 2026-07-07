# MSA

Local Julia package for the multiple-scales / slow-manifold analysis of a settling,
added-mass particle in a deep-water surface wave (the decoupled Maxey–Riley problem).

## Convention

Everything is in the **`v_z` convention**: `p.vz` is the terminal settling velocity
*directly* (not `St*θ`). The Maxey–Riley gravity term is `-vz/St`, so a particle
settles at exactly `vz` in still fluid, independent of `St`.

## Design rules

* `Params` (`St, eps, b, vz`) is immutable and physics-only.
* **Every function takes `p` explicitly.** There are no module-level globals.
* `src/` holds pure definitions with **zero side effects** at load time.
* Integration details (initial condition, tspan, det-function, callbacks) are passed
  as explicit arguments to the solver helpers — never stored on `p`.
* `scripts/` holds runnable figure-makers, one per figure. Each activates this project,
  `using MSA`, builds a `Params`, solves, and `savefig`s into `figures/` with all
  parameters in the filename.

## Layout

```
src/
  MSA.jl          module: includes the below, exports the public API
  params.jl       Params struct
  dynamics.jl     det-functions: maxey_riley!, fluid_settling!, sigma_omega!, slow_manifold_xz!
  hamiltonian.jl  amplitude, hamiltonian, xdot, eigendata, + B3/σ-ω Hamiltonian helpers
  recover.jl      isobar -> (x,z,t) lab-frame reconstruction
  paths.jl        ode_path_gen, path_distance_var (double-avg error metric), σ-ω recovery
scripts/          one runnable script per figure
figures/          generated output (git-ignored)
```

## Usage

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using Revise, MSA        # Revise gives hot-reload; no more slow re-include / OOM loop
```

Run a figure script as a fresh process (each is a clean process, so no REPL memory build-up):

```
julia --project=. scripts/mr_fs_isobar.jl
```
