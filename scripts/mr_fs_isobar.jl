# Lab-frame trajectory comparison in the DECOUPLED setting (added mass `b` and terminal
# settling velocity `v_z` independent). From the common release point (x, z) = (0, 0):
#   * isobar of the simplified Hamiltonian H(X, z)   (blue)
#   * full Maxey-Riley released from rest             (red)
#   * full Maxey-Riley released at fluid velocity     (green)
#   * particle = fluid + settling                     (purple, dashed)
#
# Reproduces the old code/NEW_plot_mr_fs_isobar.jl figure using the MSA package.
#
#   julia --project=. scripts/mr_fs_isobar.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using MSA
using DifferentialEquations, Plots

"Draw and save the four-curve comparison for parameters `p`."
function mr_fs_isobar_figure(p::Params; N=3000, tend=nothing, prefix="mr_fs_isobar",
                             outdir=joinpath(@__DIR__, "..", "figures"))
    T = tend === nothing ? 5/p.vz : tend

    sol4(u0) = solve(ODEProblem(maxey_riley!, u0, (0.0, T), p),
                     abstol=1e-9, reltol=1e-9, saveat=0.05, dense=false, maxiters=1e8)
    mr_rest  = sol4([0.0, 0.0, 0.0, 0.0])       # M-R from rest
    mr_fluid = sol4([0.0, 0.0, p.eps, 0.0])     # M-R at fluid velocity (ε u at the origin)
    fs = solve(ODEProblem(fluid_settling!, [0.0, 0.0], (0.0, T), p),
               abstol=1e-9, reltol=1e-9, saveat=0.05, dense=false)

    rec = recover_xzt(p; X0=0.0, z0=0.0, Xspan=T, N=N)   # isobar -> (x, z)

    plt = plot(rec.x, rec.z, label="isobar (simplified Hamiltonian)", lw=2.5, color=:blue,
               xlabel="x", ylabel="z", size=(1000, 650), legend=:bottomright)
    plot!(plt, [u[1] for u in mr_rest.u],  [u[2] for u in mr_rest.u],
          label="M-R (from rest)", lw=2, color=:red)
    plot!(plt, [u[1] for u in mr_fluid.u], [u[2] for u in mr_fluid.u],
          label="M-R (from fluid velocity)", lw=2, color=:seagreen)
    plot!(plt, [u[1] for u in fs.u], [u[2] for u in fs.u],
          label="fluid + settling", lw=2, ls=:dash, color=:purple)

    mkpath(outdir)
    fn = joinpath(outdir,
        "$(prefix)_St$(p.St)_eps$(p.eps)_b$(p.b)_vz$(round(p.vz, digits=3)).svg")
    savefig(plt, fn)
    println("saved ", fn, "   A = ", round(amplitude(p), digits=5))
    return plt
end

# reference parameter set (matches figures/mr_fs_isobar_St0.5_eps0.33_b0.5_vz0.05.svg)
mr_fs_isobar_figure(Params(St=0.5, eps=0.33, b=0.5, vz=0.05))
