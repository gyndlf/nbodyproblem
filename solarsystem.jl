# Simulate the orbits of the outer planets of the solar system
# Using real world (universe?) data from "Geometric Numerical Integration" by Hairer, Lubich and Wanner (2006).
#

using CairoMakie

# Mass in units of solar masses (i.e. sun has mass 1)
# Position in astronomical units (1AU ~ 1.49e8 km)
# Time in days

# Sun and inner planets combined
m0 = 1.00000597682
x0 = [0.0 0.0 0.0]
v0 = [0.0 0.0 0.0]

# Jupiter
m1 = 0.000954786104043
x1 = [-3.5023653 -3.8169847 -1.5507963]
v1 = [0.00565429 -0.00412490 -0.00190589]

# Saturn
m2 = 0.000285583733151
x2 = [9.0755314 -3.0458353 -1.6483708]
v2 = [0.00168318 0.00483525 0.00192462]

# Uranus
m3 = 0.0000437273164546
x3 = [8.3101420 -16.2901086 -7.2521278]
v3 = [0.00354178 0.00137102 0.00055029]

# Neptune
m4 = 0.0000517759138449
x4 = [11.4707666 -25.7294829 -10.8169456]
v4 = [0.00288930 0.00114527 0.00039677]

# Pluto
m5 = 1.0 / (1.3e8)
x5 = [-15.5387357 -25.2225594 -3.1902382]
v5 = [0.00276725 -0.00170702 -0.00136504]

G = 2.95912208286e-4  # Gravitational constant in the same units

m = [m0, m1, m2, m3, m4, m5]
q0 = vcat(x0, x1, x2, x3, x4, x5)  # 6x3 initial positions
p0 = vcat(v0, v1, v2, v3, v4, v5) .* m  # 6x3 initial momentums


######## SOLVERS ########

function euler(q0, p0, tmax; Δt=10)
    """Explicit euler solver. Does not conserve energy"""
    tnum = length(1:Δt:tmax)
    P = zeros(tnum, 6, 3)
    Q = zeros(tnum, 6, 3)
    P[1,:,:] = p0
    Q[1,:,:] = q0
    
    for t in 2:tnum
        p = P[t-1,:,:]
        q = Q[t-1,:,:]
        P[t,:,:] = p - Δt*Hq(q)
        Q[t,:,:] = q + Δt*Hp(p)
    end
    return P, Q
end

function stormer(q0, p0, tmax; Δt=10.0)
    """Stormer-Verlet symplectic solver. First order"""
    tnum = length(1:Δt:tmax)
    P = zeros(tnum, 6, 3)
    Q = zeros(tnum, 6, 3)
    P[1,:,:] = p0
    Q[1,:,:] = q0
    
    for t in 2:tnum
        p = P[t-1,:,:]
        q = Q[t-1,:,:]
    
        q12 = q + Δt/2 * Hp(p)
        #p12 = p - Δt/2 * Hq(q12)
        P[t,:,:] = p - Δt * Hq(q12)
        Q[t,:,:] = q12 + Δt/2 * Hp(P[t,:,:])
    end 
    return P, Q
end

######## PLOTTERS ########

function plotpaths(Q; fname="planets.png")
    """Plot the paths the planets take as lines"""
    planets = ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    fig = Figure()
    ax = Axis3(fig[1,1])
    for k in 1:6
        lines!(ax, (Q[:,k,i] for i in 1:3)..., label=planets[k])
    end
    Legend(fig[1,2], ax, "Planets")
    save(fname, fig)
    println("Saved plot as $fname")
end

function animatepaths(Q; Δt=10, tskip=50, taillength=1500, fname="planets.mp4")
    """Animate the paths the planets take"""
    t = Observable(taillength) # current time parameter. Varys over the animation
    fig = Figure()
    ax = Axis3(fig[1,1], title = @lift("t= $(round($t*Δt/365, digits=2)) years"),)

    planets = ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    tail = @lift($t-taillength+1:$t)

    for k in 1:6
        scatter!(ax, @lift(Q[$t,k,1]), @lift(Q[$t,k,2]), @lift(Q[$t,k,3]), label=planets[k])
        lines!(ax, @lift(Q[$tail,k,1]), @lift(Q[$tail,k,2]), @lift(Q[$tail,k,3]))
    end

    Legend(fig[1,2], ax, "Planets")
    limits!(ax,
        minimum(Q[:,:,1]), maximum(Q[:,:,1]),
        minimum(Q[:,:,2]), maximum(Q[:,:,2]),
        minimum(Q[:,:,3]), maximum(Q[:,:,3])
    )

    record(fig, fname, taillength:tskip:size(Q)[1]; framerate = 20) do tstep
        t[] = tstep
        ax.azimuth[] = 1.7pi + 0.3 * sin(2pi * tstep / (50*120)) 
    end
    println("Saved animation to $fname")
end

######## RUN IT ######## 

tmax = 500 * 365  # 500 years
P, Q = stormer(q0, p0, tmax)

plotpaths(Q)
animatepaths(Q)
