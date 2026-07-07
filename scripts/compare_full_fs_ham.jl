# Full Maxey-Riley (released ON the slow manifold) vs. the two reduced descriptions:
#   * isobar reconstruction of the slow-manifold Hamiltonian   (blue)
#   * full Maxey-Riley                                          (orange)
#   * particle = fluid + settling                               (purple, dashed)
#
# Successor to code/fixed_xzt.jl. NOTE: the old fixed_xzt.jl mixed conventions (the
# isobar settled at St*θ while the M-R det-function settled at θ), which is what made
# its isobar and M-R disagree by ~9%. Here everything uses a single `vz`, so the isobar
# and the full M-R agree to solver tolerance — this is the corrected figure.
#
#   julia --project=. scripts/compare_full_fs_ham.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using MSA
using DifferentialEquations, Plots, LaTeXStrings

function compare_full_fs_ham(p::Params; X0=0.0, z0=0.0, N=5000,
                             outdir=joinpath(@__DIR__, "..", "figures"))
    A  = amplitude(p)
    T  = 5/p.vz

    # full Maxey-Riley, released exactly on the slow manifold: v = A·E - i·vz at (X0,z0)
    E0 = exp(z0 + im*X0)
    u0 = [X0, z0, real(A*E0), imag(A*E0) - p.vz]
    mr = solve(ODEProblem(maxey_riley!, u0, (0.0, T), p),
               abstol=1e-10, reltol=1e-10, dtmax=0.01, maxiters=1e9)

    # particle = fluid + settling
    fs = solve(ODEProblem(fluid_settling!, [X0, z0], (0.0, T), p),
               abstol=1e-10, reltol=1e-10, dtmax=0.01)

    # isobar reconstruction of H through (X0, z0)
    rec = recover_xzt(p; X0=X0, z0=z0, Xspan=T, N=N)

    plt = plot(rec.x, rec.z, label="isobar reconstruction", lw=2, color=:blue,
               xlabel=L"x", ylabel=L"z", size=(1000, 640),
               title="St=$(p.St), b=$(p.b), vz=$(p.vz)")
    plot!(plt, [mr(s)[1] for s in mr.t], [mr(s)[2] for s in mr.t],
          label="Maxey-Riley (released on manifold)", lw=3, color=:orange, alpha=0.6)
    plot!(plt, [fs(s)[1] for s in fs.t], [fs(s)[2] for s in fs.t],
          label="fluid + settling", lw=2, ls=:dash, color=:purple)

    mkpath(outdir)
    fn = joinpath(outdir, "compare_full_fs_ham_St$(p.St)_eps$(p.eps)_b$(p.b)_vz$(round(p.vz, digits=4)).svg")
    savefig(plt, fn)
    println("endpoints:  isobar x=$(round(rec.x[end],digits=4)) z=$(round(rec.z[end],digits=3)) | ",
            "MR x=$(round(mr(T)[1],digits=4)) z=$(round(mr(T)[2],digits=3)) | ",
            "fs x=$(round(fs(T)[1],digits=4)) z=$(round(fs(T)[2],digits=3))")
    println("saved ", fn)
    return plt
end

compare_full_fs_ham(Params(St=0.5, eps=0.33, b=0.9, vz=0.05))
