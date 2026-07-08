# MSA — quick tutorial

Practical recipes for the things we do over and over. Everything is in the **`v_z`
convention**: `p.vz` is the terminal settling velocity, gravity term `-vz/St`.

The golden rule of the package: **every function takes `p::Params`**. Build a `Params`,
pass it around, done.

---

## 0. One-time setup

```bash
cd ~/Wolke/Uni/PhD/MSA
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # resolves deps (first time only)
```

---

## 1. The interactive loop (this replaces the old re-`include` / OOM cycle)

Start one REPL and keep it open all day. `Revise` hot-reloads any edit you make to
`src/` — no restart, no re-include.

```bash
cd ~/Wolke/Uni/PhD/MSA
julia --project=.
```
```julia
julia> using Revise, MSA     # that's it — solve, ODEProblem, plot, contour, etc. all come along

julia> p = Params(St=0.5, eps=0.33, b=0.9, vz=0.05)
julia> mr_fs_isobar_figure(p)
```

Now edit any `src/` file (including `src/figures.jl`), save, and just call the function
again in the same REPL — the change is live, no restart. Revise tracks everything under
`using MSA` automatically, including the figure-makers.

If a long session ever gets sluggish or memory-heavy, `GC.gc()` or just restart the REPL
— you no longer lose anything, since all state rebuilds from `p` in two lines.

---

## 2. Recurring recipes

### A. Regenerate a figure with different parameters
Every figure is one exported function (in `src/figures.jl`) that takes a `Params` and
saves into `figures/` with all params in the name.

```julia
# interactive — call it for any p, right after `using MSA`:
julia> mr_fs_isobar_figure(Params(St=1.0, eps=0.33, b=0.5, vz=0.02))
julia> mr_fs_isobar_figure(Params(St=10.0, eps=0.33, b=0.1, vz=0.1))

julia> make_all_figures()          # rebuild every figure with its canonical params
```
```bash
# same, as a one-off fresh process:
julia --project=. -e 'using MSA; mr_fs_isobar_figure(Params(St=1.0, eps=0.33, b=0.5, vz=0.02))'
```

### B. Solve and plot a Maxey–Riley trajectory
```julia
p   = Params(St=0.5, eps=0.33, b=0.9, vz=0.05)
sol = solve(ODEProblem(maxey_riley!, [0.0,0.0,0.0,0.0], (0.0, 100.0), p),
            saveat=0.05, dense=false)                       # dense=false + saveat = no OOM
plot([u[1] for u in sol.u], [u[2] for u in sol.u], xlabel="x", ylabel="z")
```
Release *at fluid velocity* instead of from rest: `[0.0, 0.0, p.eps, 0.0]`.
Fluid + settling reference (2-D state): `solve(ODEProblem(fluid_settling!, [0.0,0.0], (0.0,100.0), p), ...)`.

### C. Reconstruct the isobar trajectory — no ODE solved
```julia
rec = recover_xzt(p; X0=0.0, z0=0.0, Xspan=100)             # Xspan defaults to 5/vz
plot!(rec.x, rec.z, label="isobar")                         # rec has .x .z .t .X
```

### D. Sweep a parameter, overlay the curves
```julia
plt = plot(xlabel="x", ylabel="z", legend=:bottomright)
for vz in (0.02, 0.05, 0.1)
    rec = recover_xzt(Params(St=0.5, eps=0.33, b=0.9, vz=vz); Xspan=5/vz)
    plot!(plt, rec.x, rec.z, label="vz=$vz")
end
plt
```

### E. Hamiltonian / isobar contour map
```julia
H = hamiltonian(p)                                          # closure H(X,z); level curves = isobars
X = range(0, 20pi, 400); z = range(-5, 2, 400)
contour(X, z, [H(x, zz) for zz in z, x in X], fill=true, levels=30,
        xlabel="X = x-t", ylabel="z")
```
`hamiltonian_B3(p)` gives `H̄(B₃)` and `ham_trafo(p)` gives `H(σ,ω)` for the other
coordinate views (the ready-made `b3_contours(p)`, `xz_contours(p)`,
`sigma_omega_contours(p)` draw these).

### F. Reduced (σ,ω) system: solve, recover, collapse
```julia
so   = sigma_omega_solution(p, [0.0,0.0,0.33,0.0], (0.0, 200.0))   # so.sol, so.sigma, so.omega
full = recover_xz_from_sigma_omega(p, so.sigma, so.omega, [0.0,0.0], (0.0,200.0))

# project onto the 2-D slow eigenspace, then recover — should barely move (B₁,B₂ ≈ 0)
σs, ωs = collapse_sigma_omega(p, so.sigma, so.omega)
simpl  = recover_xz_from_sigma_omega(p, σs, ωs, [0.0,0.0], (0.0,200.0))
```

### G. Grade an approximation against full M-R (double-averaging metric)
```julia
e_slow  = path_distance_var(p, maxey_riley!, [0.0,0.0,0.33,0.0],
                               slow_manifold_xz!, [0.0,0.0], (0.0, 200.0))
e_fluid = path_distance_var(p, maxey_riley!, [0.0,0.0,0.33,0.0],
                               fluid_settling!,  [0.0,0.0], (0.0, 200.0))
```
For the full St-sweep figure, call `st_accuracy_figure()`.

### H. Inspect the linear data (no solving)
```julia
amplitude(p)        # complex response amplitude A  (= ε when b=1)
eigendata(p)        # (; lambda1, lambda3, vhat, M)
xdot(p)(0.0, 0.0)   # wave-frame dX/dt at (X,z)=(0,0)
```

---

## 3. Add a new figure

Add the function to `src/figures.jl` (this is what makes it callable from any
`using MSA` session):

```julia
"""
    my_figure(p::Params; outdir=DEFAULT_FIGDIR)

One-line description of what this draws.
"""
function my_figure(p::Params; outdir=DEFAULT_FIGDIR)
    # ... build things from p, solve, plot ...
    mkpath(outdir)
    fn = joinpath(outdir, "my_figure_$(figure_suffix(p)).svg")
    savefig(fn); println("saved ", fn)
end
```

Then: add `my_figure` to the `# figure-makers` export list in `src/MSA.jl`, and (if it's
a paper figure) add a call with its canonical `Params` to `make_all_figures`.

Keep the rule: **all run parameters in the filename** (`figure_suffix(p)` does this for
you), SVG into `figures/` (git-ignored — figures are always regenerable).

---

## 4. Conventions & gotchas

- **`vz`, not `St·θ`.** `p.vz` is the terminal settling velocity directly. If you have an
  old `θ`, pass `vz = St*θ`. (This is what the coupled-physics figures use, `θ = 1-b`,
  i.e. `vz = St*(1-b)`.)
- **`b = 1` is the tracer limit:** `amplitude` stays finite (`A = ε`) but `eigendata`/
  `projection` are singular there — expected.
- **OOM guard:** always solve with `dense=false` and `saveat=...` when you only need the
  samples; the dense interpolant is what blew up warm REPLs before.
- `using MSA` re-exports `solve`, `ODEProblem`, `plot`, `contour`, `savefig`,
  `find_zero`, `quadgk`, `L"..."` — nothing extra to `using`.
- For a clean, reproducible render use a **fresh process**
  (`julia --project=. -e 'using MSA; make_all_figures()'`); use the **one long Revise
  REPL** for exploration.
