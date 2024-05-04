# Simulate a system of N particles with random initial positions and momentum

using CairoMakie


function generateparticles(N; xmax=10, ymax=10, mmin=0.2, mmax=1, vmax=1)
    """Generate N particles (2D) in (0:xmax, 0:ymax) of masses [1:mmax] with max (initial) speed vmax"""
    p = (rand(N, 2) .- 0.5) .* 2vmax / sqrt(2)
    q = rand(N, 2)
    m = rand(N) .* (mmax-mmin) .+ mmin
    q[:,1] .*= xmax  # scale to fill the space
    q[:,2] .*= ymax
    p .*= m  # scale by mass
    return q, p, m
end

function Hp(p, m)
    """Return ∂H/∂p"""
    return p ./ m
end

function norm(x; ϵ=0.1)
    """Return sqrt norm of the vector x. ϵ is the minimum distance possible"""
    return max(sqrt(sum([xx^2 for xx in x])), ϵ)
end

function Hq(q, m, G)
    """Return ∂H/∂q (m is mass vector of particles)"""
    N = size(q)[1]
    hq = zeros(N, 2)
    for k in 1:N
        kth = zeros(2)
        for i in vcat(1:k-1,k+1:N)
            kth += (q[i,:]-q[k,:]) .* (m[i] / norm(q[i,:]-q[k,:])^3)
        end
        kth *= G * m[k]
        hq[k,:] = kth
    end
    return -hq
end

function stormer(q0, p0, m, tmax; Δt=0.1, G=1.0)
    """Stormer-Verlet symplectic solver. First order"""
    tnum = length(0:Δt:tmax)
    N = size(q0)[1]
    P = zeros(tnum, N, 2)
    Q = zeros(tnum, N, 2)
    P[1,:,:] = p0
    Q[1,:,:] = q0
    
    for t in 2:tnum
        p = P[t-1,:,:]
        q = Q[t-1,:,:]
    
        q12 = q + Δt/2 * Hp(p, m)
        #p12 = p - Δt/2 * Hq(q12)
        P[t,:,:] = p - Δt * Hq(q12, m, G)
        Q[t,:,:] = q12 + Δt/2 * Hp(P[t,:,:], m)
    end 
    return Q, P
end

######## PLOTTERS ########

function plotpaths(Q; fname="particles.png")
    """Plot the paths the particles take as lines"""
    fig = Figure()
    ax = Axis(fig[1,1])
    for k in 1:size(Q)[2]
        lines!(ax, Q[:,k,1], Q[:,k,2]) 
    end
    save(fname, fig)
    println("Saved plot as $fname")
end

function plotstate(q; fname="particles.png")
    """Plot the current state"""
    fig = Figure()
    ax = Axis(fig[1,1])
    scatter!(ax, q[:,1], q[:,2]) 
    save(fname, fig)
    println("Saved plot as $fname")
end


function animatepaths(Q; tskip=0, taillength=5, fname="particles.mp4")
    """Animate the paths the planets take"""
    t = Observable(taillength) # current time parameter. Varys over the animation
    fig = Figure()
    ax = Axis(fig[1,1], title = @lift("t= $(round($t, digits=2)) time"),)

    tail = @lift($t-taillength+1:$t)

    scatter!(ax, @lift(Q[$t,:,1]), @lift(Q[$t,:,2]))
    for k in 1:size(Q)[2]
        #scatter!(ax, @lift(Q[$t,k,1]), @lift(Q[$t,k,2]))
        lines!(ax, @lift(Q[$tail,k,1]), @lift(Q[$tail,k,2]))
    end

    limits!(ax,
        minimum(Q[:,:,1]), maximum(Q[:,:,1]),
        minimum(Q[:,:,2]), maximum(Q[:,:,2]),
    )

    record(fig, fname, taillength:tskip+1:size(Q)[1]; framerate = 20) do tstep
        t[] = tstep
    end
    println("Saved animation to $fname")
end


function run()
    q0, p0, m = generateparticles(20, mmax=1)
    Q, P = stormer(q0, p0, m, 50, Δt=0.1, G=0.1)
    plotpaths(Q)
    animatepaths(Q)
end



