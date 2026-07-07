# Contour maps of the reduced Hamiltonian in its various coordinates. Successor to the
# doplots* functions in code/collapse_onto_eigenspace.jl.
#
#   * H̄(B₃) in the slow-eigenspace coordinate  — Cartesian and polar   (hamiltonian_B3)
#   * H(X, z) in wave-frame coordinates, whose level curves are isobars (hamiltonian)
#   * H(σ, ω) via the collapse-onto-eigenspace map                       (ham_trafo)
#
#   julia --project=. scripts/hamiltonian_contours.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using MSA
using Plots, LaTeXStrings

const OUTDIR = joinpath(@__DIR__, "..", "figures")
suffix(p) = "St$(p.St)_eps$(p.eps)_b$(p.b)_vz$(round(p.vz, digits=4))"

"H̄(B₃) over the complex B₃-plane — Cartesian and polar grids."
function b3_contours(p::Params; n=100)
    Hbar = hamiltonian_B3(p)
    mkpath(OUTDIR)
    # Cartesian
    x = range(-10, 10, length=n); y = range(-10, 10, length=n)
    for (part, f) in (("real", real), ("imag", imag))
        Z = @. f(Hbar(x' + im*y))
        contour(x, y, Z, fill=true, xlabel=L"\Re(B_3)", ylabel=L"\Im(B_3)",
                colorbar_title=part == "real" ? L"\Re(H)" : L"\Im(H)")
        savefig(joinpath(OUTDIR, "contour_of_$(part)_hamiltonian_$(suffix(p)).svg"))
    end
    # polar
    r = range(0, 10, length=n); ang = range(-pi, pi, length=n)
    for (part, f) in (("real", real), ("imag", imag))
        Z = @. f(Hbar(r' * exp(im*ang)))
        contour(r, ang, Z, fill=true, xlabel=L"r(B_3)", ylabel=L"\arg(B_3)",
                colorbar_title=part == "real" ? L"\Re(H)" : L"\Im(H)")
        savefig(joinpath(OUTDIR, "polar_contour_of_$(part)_hamiltonian_$(suffix(p)).svg"))
    end
end

"H(X, z) in wave-frame coordinates; its level curves are the isobars."
function xz_contours(p::Params; xlim=(0, 20pi), zlim=(-5, 2), n=500)
    H = hamiltonian(p)
    X = range(xlim[1], xlim[2], length=n); z = range(zlim[1], zlim[2], length=n)
    Hgrid = [H(x, zz) for zz in z, x in X]
    contour(X, z, Hgrid, fill=true, levels=30, xlabel=L"X = x-t", ylabel=L"z",
            colorbar_title=L"H(X,z)")
    mkpath(OUTDIR)
    savefig(joinpath(OUTDIR, "hamiltonian_XZ_$(suffix(p)).svg"))
end

"H(σ, ω) via the collapse-onto-slow-eigenspace map."
function sigma_omega_contours(p::Params; lim=500.0, n=100)
    Ht = ham_trafo(p)
    s = range(-lim, lim, length=n); o = range(-lim, lim, length=n)
    for (part, f) in (("real", real), ("imag", imag))
        Z = @. f(Ht(s', o))
        contour(s, o, Z, fill=true, xlabel=L"\sigma", ylabel=L"\omega",
                colorbar_title=part == "real" ? L"\Re(H)" : L"\Im(H)")
        savefig(joinpath(OUTDIR, "sigma_omega_contour_of_$(part)_hamiltonian_$(suffix(p)).svg"))
    end
end

# Coupled-physics parameter set from the old collapse script: θ = 1-b, so vz = St(1-b).
p = Params(St=0.8, eps=0.33, b=0.9, vz=0.8*(1 - 0.9))
b3_contours(p)
xz_contours(p)
sigma_omega_contours(p)
println("saved Hamiltonian contour figures for ", suffix(p))
