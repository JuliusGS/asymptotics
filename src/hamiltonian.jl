# Slow-manifold amplitude, Hamiltonian, and linear (eigen) data — all pure functions
# of `p::Params`, in the v_z convention. The old `θ` (terminal velocity `St*θ`) maps to
# `θ = vz/St` here, so every `St*θ` becomes `vz` and every `St²*θ` becomes `St*vz`.

"""
    eigendata(p::Params) -> (; lambda1, lambda3, vhat, M)

Linear data of the reduced `(σ, ω)` system about its fixed point:
* `lambda1 = -1/St`                                   — fast eigenvalue
* `lambda3 = -vz - im`                                — slow eigenvalue (`= -St*θ - i`)
* `M       = (1 - i St - St*vz)/(ε (1-b))`            — appears in `A` and the Hamiltonian
* `vhat    = conj(M) = (St²θ - i St - 1)/(ε (b-1))`   — spans the slow eigenspace

Singular at `b = 1` (the tracer limit), where the slow eigenspace degenerates but the
response amplitude [`amplitude`](@ref) stays finite (`A = ε`).
"""
function eigendata(p::Params)
    St = p.St; eps = p.eps; b = p.b; vz = p.vz
    lambda1 = -1/St
    lambda3 = -vz - im
    M    = (1 - im*St - St*vz) / (eps*(1 - b))
    vhat = conj(M)
    return (; lambda1, lambda3, vhat, M)
end

"""
    amplitude(p::Params) -> ComplexF64

Complex response amplitude `A` of the particle on the slow manifold:
`A = b ε + (1-b) ε / (1 - i St - vz St)`  (so `A = ε` exactly when `b = 1`).
"""
function amplitude(p::Params)
    St = p.St; eps = p.eps; b = p.b; vz = p.vz
    return b*eps + (1 - b)*eps / (1 - im*St - vz*St)
end

"""
    hamiltonian(p::Params) -> Function

Generating Hamiltonian (stream function) of the slow-manifold flow in wave-frame
coordinates `(X, z)`, `X = x - t`:
`H(X, z) = -z + vz X + eᶻ Re(A e^{iX})`,  `A = amplitude(p)`.
Its level curves are the isobars. Returns the closure `H(X, z)`.
"""
function hamiltonian(p::Params)
    A = amplitude(p); vz = p.vz
    return (X, z) -> -z + vz*X + exp(z)*real(A*exp(im*X))
end

"""
    xdot(p::Params) -> Function

Wave-frame velocity `dX/dt = -1 + eᶻ Re(A e^{iX})` on the slow manifold.
Returns the closure `Xdot(X, z)`.
"""
function xdot(p::Params)
    A = amplitude(p)
    return (X, z) -> -1 + exp(z)*real(A*exp(im*X))
end

"""
    hamiltonian_B3(p::Params) -> Function

The reduced Hamiltonian in the slow-eigenspace coordinate `B₃` (before unwrapping to
`(X, z)`):
`H̄(B₃) = -i λ₃ log(B̄₃) + (1 + b ε M̄) B̄₃ + i λ̄₃ log(B₃) + (1 + b ε M) B₃`.
Multivalued through `log`; becomes single-valued once written in `(X, z)`
(see [`hamiltonian`](@ref)). Returns the closure `Hbar(B3)`.
"""
function hamiltonian_B3(p::Params)
    ed = eigendata(p); b = p.b; eps = p.eps
    l3 = ed.lambda3; M = ed.M
    return B3 -> -im*l3*log(conj(B3)) + (1 + b*eps*conj(M))*conj(B3) +
                  im*conj(l3)*log(B3) + (1 + b*eps*M)*B3
end

"""
    projection(p::Params) -> Matrix{ComplexF64}

Orthogonal projector (4×4) onto the slow eigenspace `span(v₃, v₄)` of the reduced
`(σ, ω)` system, used to collapse a full `(σ, ω)` state onto the slow manifold.
"""
function projection(p::Params)
    vh = eigendata(p).vhat
    A = ComplexF64[0 vh; conj(vh) 0; 1 0; 0 1]
    return A * (transpose(A)*A)^(-1) * transpose(A)
end

"""
    ham_trafo(p::Params) -> Function

Evaluate the reduced Hamiltonian at a full `(σ, ω)` state: collapse `(σ, ω)` onto the
slow eigenspace (subtract the fixed point, project, add it back), decompose into `B₃`,
and return `H̄(B₃)`. Returns the closure `(sigma, omega) -> H`.
"""
function ham_trafo(p::Params)
    vh = eigendata(p).vhat; vz = p.vz
    P = projection(p)
    Hbar = hamiltonian_B3(p)
    fp = ComplexF64[0, 0, -1 - im*vz, -1 + im*vz]
    A = ComplexF64[0 vh; conj(vh) 0; 1 0; 0 1]
    Apinv = pinv(A)
    return function (sigma, omega)
        collapsed = P*(ComplexF64[sigma, conj(sigma), omega, conj(omega)] - fp) + fp
        s_s = collapsed[1]; o_s = collapsed[3]
        B = Apinv * (ComplexF64[s_s, conj(s_s), o_s, conj(o_s)] - fp)
        return Hbar(B[1])
    end
end
