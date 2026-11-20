#this module contains what is needed for the linear system
module hele_fast
using Plots, SparseArrays, LaTeXStrings, LinearAlgebra, Dierckx

#This is hoppers conformal map for equal cylinders
function map(a,ζ)
    sqrt(2)*(1-a^4)*ζ/(sqrt(1+a^4)*(1+(a^2)*ζ^2))
end

#Returns an initial shape for the coalescing drops
function hopper_init(rb)
    a = 1 - rb/2 
    dθ = rb/100
    num = floor(Int, pi/(2*dθ))
    θ = LinRange(0, pi/2, num)
    zvals = [map(a, exp(θ[i]*im)) for i in 1:length(θ)]
    return real.(zvals), imag.(zvals)
end

function ∂x1(f, N)
    df = Vector{Float64}(undef, N)
    @inbounds for i in 2:(N-1)
        df[i] = (f[i+1] - f[i-1]) * 0.5
    end
    df[1] = 0.0
    df[N] = -f[N-1]
    return df
end

function ∂y1(f, N)
    df = Vector{Float64}(undef, N)
    @inbounds for i in 2:(N-1)
        df[i] = (f[i+1] - f[i-1]) * 0.5
    end
    df[1] = f[2]
    df[N] = 0.0
    return df
end

function ∂κ_∂s(f, ds)
    N = length(f)
    df = Vector{Float64}(undef, N)
    @inbounds for i in 2:(N-1)
        df[i] = (f[i+1] - f[i-1]) / (ds[i] + ds[i-1])
    end
    df[1] = 0.0
    df[N] = 0.0 
    return df
end

function ∂x2(f, N)
    df = Vector{Float64}(undef, N)
    @inbounds for i in 2:(N-1)
        df[i] = f[i+1] + f[i-1] - 2.0*f[i]
    end
    df[1] = -2.0*f[1] + 2.0*f[2]
    df[N] = -2.0*f[N]
    return df
end

function ∂y2(f, N)
    df = Vector{Float64}(undef, N)
    @inbounds for i in 2:(N-1)
        df[i] = f[i+1] + f[i-1] - 2.0*f[i]
    end
    df[1] = -2.0*f[1]
    df[N] = -2.0*f[N] + 2.0*f[N-1]
    return df
end

# Kept as Sparse Matrices for blazing-fast O(N) solving
function ∂2x_∂2s_mat(N, ds)
    w  = @. 2.0 / (ds[1:end-1] + ds[2:end])
    dl = @. w / ds[1:end-1]
    du = @. w / ds[2:end]
    d  = @. -dl - du
    
    diag_minus1 = vcat(dl, 0.0)             
    diag_0      = vcat(-2.0/ds[1]^2, d, 0.0)  
    diag_plus1  = vcat(2.0/ds[1]^2, du)       
    
    return spdiagm(-1 => diag_minus1, 0 => diag_0, 1 => diag_plus1)
end

function ∂2y_∂2s_mat(N, ds)
    w  = @. 2.0 / (ds[1:end-1] + ds[2:end])
    dl = @. w / ds[1:end-1]
    du = @. w / ds[2:end]
    d  = @. -dl - du
    
    diag_minus1 = vcat(dl, 2.0/ds[N-1]^2)     
    diag_0      = vcat(0.0, d, -2.0/ds[N-1]^2)
    diag_plus1  = vcat(0.0, du)             
    
    return spdiagm(-1 => diag_minus1, 0 => diag_0, 1 => diag_plus1)
end

function curv_normal(x, y)
    N = length(x)
    dx = ∂x1(x, N)
    dy = ∂y1(y, N)
    d2x = ∂x2(x, N)
    d2y = ∂y2(y, N)

    κ = Vector{Float64}(undef, N)
    nx = Vector{Float64}(undef, N)
    ny = Vector{Float64}(undef, N)

    @inbounds for i in 1:N
        dsq = dx[i]^2 + dy[i]^2
        dnorm = sqrt(dsq)
        κ[i] = (dx[i]*d2y[i] - dy[i]*d2x[i]) / (dsq * dnorm)
        nx[i] = dy[i] / dnorm
        ny[i] = -dx[i] / dnorm
    end
    
    nx[1] = 1.0; ny[1] = 0.0
    nx[N] = 0.0; ny[N] = 1.0

    return κ, nx, ny
end

function find_ds(x, y)
    N = length(x)
    ds = Vector{Float64}(undef, N-1)
    @inbounds for i in 1:(N-1)
        ds[i] = sqrt((x[i+1]-x[i])^2 + (y[i+1]-y[i])^2)
    end
    return ds
end
# ========================================================
# 4. SINGLE-THREADED, CACHE-ALIGNED BOUNDARY INTEGRALS
# ========================================================
function E_mat(x, y, ds)
    N = length(x)
    E = Matrix{Float64}(undef, N, N) 
    inv4pi = 1.0 / (4.0 * pi)
    inv2pi = 1.0 / (2.0 * pi)

    # NO THREADS: But Loop order (j outer, i inner) remains for cache-line speed
    for j in 2:(N-1)
        xj = x[j]
        yj = y[j]
        ds_avg = 0.5 * (ds[j] + ds[j-1])
        diag_val = (0.5 * ds[j] * log(0.5 * ds[j] / exp(1)) + 
                    0.5 * ds[j-1] * log(0.5 * ds[j-1] / exp(1))) * inv2pi

        @fastmath @inbounds for i in 1:N 
            xi = x[i]; yi = y[i]
            
            dx = xi - xj; sx = xi + xj
            dy = yi - yj; sy = yi + yj
            
            r1 = dx*dx + dy*dy
            r2 = sx*sx + dy*dy
            r3 = dx*dx + sy*sy
            r4 = sx*sx + sy*sy
            
            if i != j
                E[i,j] = ds_avg * log(r1 * r2 * r3 * r4) * inv4pi
            else
                E[j,j] = diag_val + ds_avg * log(r2 * r3 * r4) * inv4pi
            end
        end
    end
    
    @fastmath @inbounds for i in 1:N
        xi = x[i]; yi = y[i]
        
        dx = xi - x[1]; sx = xi + x[1]; dy = yi - y[1]
        E[i,1] = ds[1] * log((dx*dx + dy*dy) * (sx*sx + dy*dy)) * inv4pi 
        
        dx = xi - x[N]; dy = yi - y[N]; sy = yi + y[N]
        E[i,N] = ds[N-1] * log((dx*dx + dy*dy) * (dx*dx + sy*sy)) * inv4pi
    end

    E[1,1] = (ds[1]*log(0.5*ds[1]/exp(1))) * inv2pi + ds[1]*log(4.0*x[1]*x[1]) * inv4pi
    E[N,N] = (ds[N-1]*log(0.5*ds[N-1]/exp(1))) * inv2pi + ds[N-1]*log(4.0*y[N]*y[N]) * inv4pi
    
    return E
end

function rhs_int(x, y, nx, ny, κ, ds)
    N = length(x)
    B = Vector{Float64}(undef, N)
    pi2_inv = 1.0 / (2.0 * pi)

    # NO THREADS: Standard fast-math loop
    for i in 1:N 
        xi = x[i]; yi = y[i]; ki = κ[i]
        test = 0.0
        
        @fastmath @inbounds for j in 2:(N-1)
            if j != i
                xj = x[j]; yj = y[j]
                nxj = nx[j]; nyj = ny[j]; kj = κ[j]
                
                ds_avg = 0.5 * (ds[j] + ds[j-1])
                dk = kj - ki
                
                dx1 = xj - xi; dy1 = yj - yi
                dx2 = xj + xi; dy2 = yj - yi
                dx3 = xj - xi; dy3 = yj + yi
                dx4 = xj + xi; dy4 = yj + yi
                
                t1 = (nxj * dx1 + nyj * dy1) / (dx1*dx1 + dy1*dy1)
                t2 = (nxj * dx2 + nyj * dy2) / (dx2*dx2 + dy2*dy2)
                t3 = (nxj * dx3 + nyj * dy3) / (dx3*dx3 + dy3*dy3)
                t4 = (nxj * dx4 + nyj * dy4) / (dx4*dx4 + dy4*dy4)
                
                test += dk * ds_avg * (t1 + t2 + t3 + t4) * pi2_inv
            end
        end

        @fastmath @inbounds begin
            if i != 1
                dx1 = x[1] - xi; dy1 = y[1] - yi
                dx2 = -x[1] - xi
                t1 = (nx[1] * dx1 + ny[1] * dy1) / (dx1*dx1 + dy1*dy1)
                t2 = (-nx[1] * dx2 + ny[1] * dy1) / (dx2*dx2 + dy1*dy1)
                test += (κ[1] - ki) * ds[1] * (t1 + t2) * pi2_inv
            end

            if i != N
                dx1 = x[N] - xi; dy1 = y[N] - yi
                dy3 = -y[N] - yi
                t1 = (nx[N] * dx1 + ny[N] * dy1) / (dx1*dx1 + dy1*dy1)
                t2 = (nx[N] * dx1 - ny[N] * dy3) / (dx1*dx1 + dy3*dy3)
                test += (κ[N] - ki) * ds[N-1] * (t1 + t2) * pi2_inv
            end
        end
        B[i] = -test
    end
    return B
end

function compute_vel(x, y)
    N = length(x)
    κ, nx, ny = curv_normal(x, y)
    ds = find_ds(x, y)
    E_mat0 = E_mat(x, y, ds)
    b_vec = rhs_int(x, y, nx, ny, κ, ds)
    
    # Fast in-place LAPACK solve. Destroys E_mat0, turns b_vec into u_normal
    F = lu!(E_mat0)
    u_normal = ldiv!(F, b_vec) 
    
    #u_tan = -∂κ_∂s(κ, ds)

    #compute normal velocity, store in nx
    @inbounds for i in 1:N
        nx[i] *= u_normal[i] #- u_tan[i] * ny[i]
        ny[i] *= u_normal[i]  #+ u_tan[i] * nx[i]
    end

    return nx, ny
end

function ein_step(x, y, dt)
    N = length(x)
    ux, uy = compute_vel(x, y)
    ds = find_ds(x, y)

    Dx2 = ∂2x_∂2s_mat(N, ds)
    Dy2 = ∂2y_∂2s_mat(N, ds)

    # Sparse Matrix multiplication is extremely fast in Julia
    Dx4 = Dx2 * Dx2
    Dy4 = Dy2 * Dy2

    push!(ds,ds[end])

    λ = dt*spdiagm(0 => ds)


    # I is LinearAlgebra.I. Adding it to Sparse produces Sparse
    Lx_mat = I + (λ*Dx4)
    Ly_mat = I + (λ*Dy4)

    # O(N) Sparse Back-substitution
    #δx = Lx_mat \ (dt .* ux)
    #δy = Ly_mat \ (dt .* uy)

    δx =  Lx_mat \ (x .+ dt*ux .+ λ*Dx4*x)
    δy =  Ly_mat \ (y .+ dt*uy .+ λ*Dy4*y)

    #Lx_mat \ (x .+ dt*ux .+ λ*Dx4*x)

    return δx, δy #δx, y .+ δy
end


function zwei_step(x, y, dt; abstol = 1e-20)
    x1, y1 = ein_step(x, y, dt)
    x21, y21 = ein_step(x, y, dt/2.0)
    x21, y21 = ein_step(x21, y21, dt/2.0)

    N = length(x)
    new_x = Vector{Float64}(undef, N)
    new_y = Vector{Float64}(undef, N)
    
    error_val = 0.0
    
    @inbounds for i in 1:N
        new_x[i] = 2.0 * x21[i] - x1[i]
        new_y[i] = 2.0 * y21[i] - y1[i]
        
        sx = abstol + max(abs(x21[i]), abs(x1[i]))
        sy = abstol + max(abs(y21[i]), abs(y1[i]))
        
        ex = abs(x21[i] - x1[i]) / (3*sx)
        ey = abs(y21[i] - y1[i]) / (3*sy)
        
        error_val = max(error_val, ex, ey)
    end

    return new_x, new_y, error_val
end

mutable struct state
    x::Vector{Float64}
    y::Vector{Float64}
    t::Float64
    err::Float64
end

function regrid(x, y, κ; r=1.1)
    rB = x[1]
    N = length(x)
    pad_len = 3 

    @views x_start = x[(pad_len+1):-1:2]
    @views y_start = -y[(pad_len+1):-1:2]
    @views x_end = -x[(N-1):-1:(N-pad_len)]
    @views y_end = y[(N-1):-1:(N-pad_len)]

    x_padded = vcat(x_start, x, x_end)
    y_padded = vcat(y_start, y, y_end)

    ds = find_ds(x_padded, y_padded)
    arc_padded = vcat([0.0], cumsum(ds))

    arc_start = arc_padded[pad_len + 1]
    arc_end   = arc_padded[pad_len + N]

    xspline = Dierckx.Spline1D(arc_padded, x_padded; k=3, s=0.0)
    yspline = Dierckx.Spline1D(arc_padded, y_padded; k=3, s=0.0)

    κ_max_dex = argmax(abs.(κ))
    rc_min = 1/abs(κ[κ_max_dex]) 
    s_loc = arc_padded[pad_len+κ_max_dex] 

    ds_max = arc_end/100.0
    ds_min = min(pi*rc_min/16.0, ds_max) 

    new_s = [arc_start, arc_start+ds_min] 
    new_x = [rB, xspline(new_s[2])] 
    new_y = [0.0, yspline(new_s[2])] 

    while new_x[end] < rB + rc_min #new_s[end] < max(s_loc+60*ds_min, arc_start+60*ds_min)
        push!(new_s, new_s[end]+ds_min)
        push!(new_x, xspline(new_s[end]))
        push!(new_y, yspline(new_s[end]))
        if new_y[end] > 1.0 
            break
        end 
    end

    while new_s[end] < arc_end-ds_max 
        ds_1 = min(ds_max, r*(new_s[end]-new_s[end-1]))
        x_test = xspline(new_s[end]+ds_1)
        if x_test <= 0.0
            break 
        elseif new_s[end]+ds_1 > arc_end
            break
        elseif yspline(new_s[end]+ds_1) > y[end]
            break
        else
            push!(new_s, new_s[end] + ds_1)
            push!(new_x, x_test)
            push!(new_y, yspline(new_s[end]))
        end
    end

    if new_x[end] < ds_max/2.0
        new_x = new_x[1:end-1]
        new_y = new_y[1:end-1]
    end

    push!(new_x, 0.0)
    push!(new_y, y[end])
    
    return new_x, new_y
end

function drop_plot(x, y)
    xfull = [x; -reverse(x); -x[2:end]; reverse(x)[1:end-1]]
    yfull = [y; reverse(y); -y[2:end]; -reverse(y)[1:end-1]]

    plot([xfull], [yfull], aspectratio=:equal,
         seriestype = :shape, fillcolor = :skyblue1, fillalpha = 0.7,
         linecolor = :darkblue, linewidth = 2, xlims=(-2.5,2.5), ylims=(-2.5,2.5))
end

function chunk!(data, n_chunks, per_chunk, dt,res)
    error = 0.0
    for _ in 1:n_chunks # Prevent loop shadowing
        t1 = data[end].t 
        x1, y1 = data[end].x, data[end].y
        for j in 1:per_chunk
            testx, testy, error = zwei_step(x1, y1, dt)
            if error > res
                dt /= 2.0 
            else
                x1, y1 = testx, testy
                κ, _, _ = curv_normal(x1, y1)
                x1, y1 = regrid(x1, y1, κ)
                t1 += dt
                dt *= 1.2
            end
            if x1[1] > 1.1 * data[end].x[1]
                break
            end
        end
        if x1[1] != data[end].x[1]
            push!(data, state(x1, y1, t1, copy(error)))
        end
    end
    return dt
end




function tip_plot(x,y)
    scatter(x[1:30],y[1:30],ms=:1,mc=:black,aspectratio=:equal)
    scatter!(x[1:30],-y[1:30],ms=:1,mc=:black)
end

end


