#this module contains what is needed for the linear system
module hele
using Plots, SparseArrays,LaTeXStrings, LinearAlgebra, Dierckx

#This is hoppers conformal map for equal cylinders
#the a parameter = 1 when they touch, and 0 when fully coalesced
#ζ = exp(i*t) runs along the unit disk
function map(a,ζ)
    sqrt(2)*(1-a^4)*ζ/(sqrt(1+a^4)*(1+(a^2)*ζ^2))
end

#Returns an initial shape for the coalescing drops using Hoppers solution
#just gives the top right quarter
#rb is the bridge solution
function hopper_init(rb)
    a=1-rb/2 #this is the approximate conversion for small rb
    dθ = rb/100;
    num = floor(Int,pi/(2*dθ));
    θ = LinRange(0,pi/2,num)
    zvals = [map(a,exp(θ[i]*im)) for i in 1:length(θ)]
    x = real.(zvals)
    y = imag.(zvals)
    return x, y
end
 

#second order accurate
function ∂x1(f,N)
    df = zeros(N)
    fc = [f[2];f;-f[N-1]]
    for i in 2:(N-1)
        df[i] = (f[i+1]-f[i-1])/2
    end
    df[1] = 0.0
    df[N] = - f[N-1]
    return df
end

function ∂y1(f,N)
    df = zeros(N)
    for i in 2:(N-1)
        df[i] = (f[i+1]-f[i-1])/2
    end
    df[1] = f[2]
    df[N] = 0.0
    return df
end

#compute the curvature derivative
function ∂κ_∂s(f,ds)
    N = length(f)
    df = zeros(N)
    for i in 2:(N-1)
        df[i] = (f[i+1] - f[i-1])/(ds[i] + ds[i-1])
    end
    df[1] = 0
    df[N] = 0 
    return df
end

#second order accurate
function ∂x2(f,N)
    df = zeros(N)
    for i in 2:(N-1)
        df[i] = f[i+1]+f[i-1]-2*f[i]
    end
    df[1] = -2*f[1] + 2*f[2]
    df[N] = -2*f[N]

    return df
end

#second order accurate
function ∂y2(f,N)
    df = zeros(N)
    for i in 2:(N-1)
        df[i] = f[i+1]+f[i-1]-2*f[i]
    end
    df[1] = -2*f[1]
    df[N] = -2*f[N] + 2*f[N-1]
    return df
end

#computes the second derivative of the x coordinate with respect to arclength
#Acts on one quarter
function ∂2x_∂2s(f,N,ds)
    df = zeros(N)
    for i in 2:(N-1)
        df[i] = ((f[i+1]-f[i])/ds[i] - (f[i]-f[i-1])/ds[i-1])*2/(ds[i]+ds[i-1])
    end

    df[1] = 2*(f[2]-f[1])/(ds[1]^2)
    df[N] = (-2*f[N]/ds[N-1]^2)
    return df
end

#computes the second derivative of the y coordinate with respect to arclength
function ∂2y_∂2s(f,N,ds)
    df = zeros(N)
    for i in 2:(N-1)
        df[i] = ((f[i+1]-f[i])/ds[i] - (f[i]-f[i-1])/ds[i-1])*2/(ds[i]+ds[i-1])
    end

    df[1] = -2*f[1]/(ds[1]^2)
    df[N] = 2*(f[N-1]-f[N])/(ds[N-1]^2)

    return df
end

# Second derivative matrices
function ∂2x_∂2s_mat(N, ds)
    w  = 2 ./ (ds[1:end-1] .+ ds[2:end])
    dl = w ./ ds[1:end-1]
    du = w ./ ds[2:end]
    d  = -dl .- du
    
    # D_x boundary conditions:
    # Point 1 (x-axis): x is an even function -> x''_1 = 2(x_2 - x_1)/ds_1^2
    # Point N (y-axis): x is exactly 0 -> x''_N = 0 
    
    # We pad the diagonals to exact lengths (N-1 for off-diags, N for main diag)
    diag_minus1 = vcat(dl, 0.0)             # Length N-1 (forces Row N to be 0)
    diag_0      = vcat(-2/ds[1]^2, d, 0.0)  # Length N   (forces Row N to be 0)
    diag_plus1  = vcat(2/ds[1]^2, du)       # Length N-1
    
    D = spdiagm(-1 => diag_minus1, 0 => diag_0, 1 => diag_plus1)
    return D
end

function ∂2y_∂2s_mat(N, ds)
    w  = 2 ./ (ds[1:end-1] .+ ds[2:end])
    dl = w ./ ds[1:end-1]
    du = w ./ ds[2:end]
    d  = -dl .- du
    
    # D_y boundary conditions:
    # Point 1 (x-axis): y is exactly 0 -> y''_1 = 0
    # Point N (y-axis): y is an even function -> y''_N = 2(y_{N-1} - y_N)/ds_{N-1}^2
    
    diag_minus1 = vcat(dl, 2/ds[N-1]^2)     # Length N-1
    diag_0      = vcat(0.0, d, -2/ds[N-1]^2)# Length N   (forces Row 1 to be 0)
    diag_plus1  = vcat(0.0, du)             # Length N-1 (forces Row 1 to be 0)
    
    D = spdiagm(-1 => diag_minus1, 0 => diag_0, 1 => diag_plus1)
    return D
end

function ∂2n_∂2s(f,N,ds)
    df = zeros(N)
    for i in 2:(N-1)
        df[i] = ((f[i+1]-f[i])/ds[i] - (f[i]-f[i-1])/ds[i-1])*2/(ds[i]+ds[i-1])
    end

    df[1] = 2*(f[2]-f[1])/(ds[1]^2)
    df[N] = 2*(f[N-1]-f[N])/(ds[N]^2)
    return df
end


#computes the curvature, as well as the normal vectors, and the arclength elements
function curv_normal(x,y)
    #to compute these we index the coordinates by their number along the curve 
    N = length(x)
    dx = ∂x1(x,N)
    dy = ∂y1(y,N)
    d2x = ∂x2(x,N)
    d2y = ∂y2(y,N)

    #curvature using this indexing
    κ = [(dx[i]*d2y[i] - dy[i]*d2x[i])/ (dx[i]^2 + dy[i]^2)^1.5 for i in 1:N]

    #components of the normal vector
    nx = [(dy[i]) / sqrt(dx[i]^2 + dy[i]^2) for i in 1:N]
    nx[1]=1.0
    nx[N] = 0.0

    ny = [(-dx[i]) / sqrt(dx[i]^2 + dy[i]^2) for i in 1:N]
    ny[1]=0.0
    ny[N] = 1.0


    #compute the arclength elements using trapezium rule 
    #ds[i] equals arclength from point i to point i+1
    #ds = [0.5*sqrt(dx[i]^2 + dy[i]^2) + 0.5*sqrt(dx[i+1]^2 + dy[i+1]^2) for i in 1:(N-1)]

    return κ,nx,ny
end

function find_ds(x,y)
    #to compute these we index the coordinates by their number along the curve 
    N = length(x)
    dx = ∂x1(x,N)
    dy = ∂y1(y,N)
    #compute the arclength elements using trapezium rule 

    #ds[i] equals arclength from point i to point i+1
    #ds = [0.5*sqrt(dx[i]^2 + dy[i]^2) + 0.5*sqrt(dx[i+1]^2 + dy[i+1]^2) for i in 1:(N-1)]

    ds = [sqrt((x[i+1]-x[i])^2 + (y[i+1]-y[i])^2) for i in 1:(N-1)]


    return ds
end




#compute the integral on the right hand side
#We must be careful to integrate around the whole shape, even though we store 1 quarter
function rhs_int(x,y,nx,ny,κ,ds)
    N=length(x)
    B=zeros(N)

    for i in 1:N 
        test = 0
        for j in 2:(N-1)
            #we can ignore the self intersection part
            if j!=i
                test += (κ[j] - κ[i]) * 0.5 * (ds[j]+ds[j-1]) / ((x[i]-x[j])^2 + (y[i]-y[j])^2) * ( (nx[j]*(x[j]-x[i])) + (ny[j]*(y[j]-y[i])) ) / (2*pi)
                #test += (κ[j] - κ[i]) * 0.5 * (ds[j]+ds[j-1]) / ((x[i]+x[j])^2 + (y[i]-y[j])^2) * ( -(nx[j]*(x[j]+x[i])) + (ny[j]*(y[j]-y[i])) ) / (2*pi)
                #test += (κ[j] - κ[i]) * 0.5 * (ds[j]+ds[j-1]) / ((x[i]+x[j])^2 + (y[i]+y[j])^2) * (-(nx[j]*(x[j]+x[i])) - (ny[j]*(y[j]+y[i])) ) / (2*pi)
                #test += (κ[j] - κ[i]) * 0.5 * (ds[j]+ds[j-1]) / ((x[i]-x[j])^2 + (y[i]+y[j])^2) * ( (nx[j]*(x[j]-x[i])) - (ny[j]*(y[j]+y[i])) ) / (2*pi)
                test += (κ[j]-κ[i]) * 0.5*(ds[j]+ds[j-1]) / ((x[i]+x[j])^2+(y[i]-y[j])^2) * (+(nx[j]*(x[i]+x[j])) + ny[j]*(y[j]-y[i])) / (2*pi)
                test += (κ[j]-κ[i]) * 0.5*(ds[j]+ds[j-1]) / ((x[i]-x[j])^2+(y[i]+y[j])^2) * (+(nx[j]*(x[j]-x[i])) + ny[j]*(y[i]+y[j])) / (2*pi)
                test += (κ[j]-κ[i]) * 0.5*(ds[j]+ds[j-1]) / ((x[i]+x[j])^2+(y[i]+y[j])^2) * (+(nx[j]*(x[i]+x[j])) + ny[j]*(y[i]+y[j])) / (2*pi)
            end
        end

        #have to manually do the first and last terms
        if i!=1
            
            test += (κ[1] - κ[i])*(ds[1]) / ((x[i]-x[1])^2 + (y[i]-y[1])^2) * ( (nx[1]*(x[1]-x[i])) + (ny[1]*(y[1]-y[i])) ) / (2*pi)
            test += (κ[1] - κ[i])*(ds[1]) / ((x[i]+x[1])^2 + (y[i]-y[1])^2) * ( -(nx[1]*(-x[1]-x[i])) + (ny[1]*(y[1]-y[i])) ) / (2*pi)
        end

        if i!=N
            test += (κ[N] - κ[i])*(ds[N-1]) / ((x[i]-x[N])^2 + (y[i]-y[N])^2) * ( (nx[N]*(x[N]-x[i])) + (ny[N]*(y[N]-y[i])) ) / (2*pi)
            test += (κ[N] - κ[i])*(ds[N-1]) / ((x[i]-x[N])^2 + (y[i]+y[N])^2) * ( (nx[N]*(x[N]-x[i])) - (ny[N]*(-y[N]-y[i])) ) / (2*pi)
        end
        
        B[i]=test

    end
    
    return B
end



#computes the (negative of the) E matrix in the language of Kelly and Hinch
#This matrix is dense
#We have to be careful to do the full shape, even though we only store 1 quarter
function E_mat(x,y,ds)
    N=length(x)
    E_mat = zeros(N,N)

    for i in 1:N 
        for j in 2:(N-1)
            r1 = (x[i]-x[j])^2 + (y[i]-y[j])^2
            r2 = (x[i]+x[j])^2 + (y[i]-y[j])^2
            r3 = (x[i]-x[j])^2 + (y[i]+y[j])^2
            r4 = (x[i]+x[j])^2 + (y[i]+y[j])^2
            if i!=j
                #integral done using midpoint rule
                #E_mat[i,j] = 0.5*(ds[j]+ds[mod1(j-1,N)])*log((x[i]-x[j])^2 + (y[i]-y[j])^2)/(4*pi)

                E_mat[i,j] = 0.5*(ds[j]+ds[mod1(j-1,N)])*(log(r1)+log(r2) + log(r3) + log(r4))/(4*pi)

            #use the analytic result for the self-interacting one
            else
                E_mat[j,j] = (0.5*ds[j]*log(0.5*ds[j]/exp(1)) + 0.5*ds[j-1]*log(0.5*ds[j-1]/exp(1)) )/(2*pi) + 0.5*(ds[j]+ds[j-1])*(log(r2) + log(r3) + log(r4))/(4*pi)

            end

        end
        E_mat[i,1] = ds[1]*log((x[i]-x[1])^2 + (y[i]-y[1])^2)/(4*pi) + ds[1]*log((x[i]+x[1])^2 + (y[i]-y[1])^2)/(4*pi) 
            
            E_mat[i,N] = ds[N-1]*log((x[i]-x[N])^2 + (y[i]-y[N])^2)/(4*pi) + ds[N-1]*log((x[i]-x[N])^2 + (y[i]+y[N])^2)/(4*pi)
    end

    #The self intersection terms at the end need care and get repaired here
    E_mat[1,1] = (ds[1]*log(0.5*ds[1]/exp(1)))/(2*pi) + ds[1]*log(4*x[1]^2)/(4*pi)
    E_mat[N,N] = (ds[N-1]*log(0.5*ds[N-1]/exp(1)))/(2*pi) + ds[N-1]*log(4*y[N]^2)/(4*pi)

    
    return E_mat

end

#compute the flux through 1 quarter of the drop
function flux(un,ds)
    n = length(un)
    stot = sum(ds)
    mass = 0
    for i in 2:(n-1)
        mass += 0.5*(ds[i]+ds[i-1])*un[i] 
    end
    mass += 0.5*un[1]*ds[1] 
    mass += 0.5*un[n]*ds[n-1]

    return mass/stot

end

#compute the velocity at one time step
function compute_vel(x,y)
    κ,nx,ny = curv_normal(x,y)
    ds = find_ds(x,y)
    E_mat0 = E_mat(x,y,ds)
    b_vec = rhs_int(x,y,nx,ny,κ,ds)
    #normal velocity
    u_normal = - E_mat0 \ b_vec

    #tangent velocity as gradient of curvature
    u_tan = - ∂κ_∂s(κ,ds)


    ux = (u_normal.*nx) .- (u_tan.*ny)
    uy = u_normal.*ny .+ (u_tan.*nx)

    return ux,uy
end

function frw_euler(x,y,scale)
    ux,uy = compute_vel(x,y)
    dt = (scale)*x[1] / maximum(sqrt.(ux.^2 .+ uy.^2))
    return x .+ dt*ux, y .+ dt*uy, dt
end

function frw_euler_dt(x,y,dt)
    ux,uy = compute_vel(x,y)
    return x .+ dt*ux, y .+ dt*uy
end

#Uses the explicit-implicit null method
function ein_step2(x,y,dt)
    N=length(x)
    κ,nx,ny = curv_normal(x,y)
    ds = find_ds(x,y)
    
    ux, uy = compute_vel(x,y)

    #partial derivative matrices
    Dx = ∂2x_∂2s_mat(N,ds)
    Dy = ∂2y_∂2s_mat(N,ds)

    #damping factor
    #changes site to site
    λ = 2*pi*dt*spdiagm(0=>ones(N))./minimum(ds)


    #L matrices 
    Lx_mat = spdiagm(0=>ones(N)) .- λ*Dx
    Ly_mat = spdiagm(0=>ones(N)) .- λ*Dy


    #compute the incrememnts
    δx = dt*(Lx_mat \ ux)
    δy = dt*(Ly_mat \ uy)

    return x .+ δx , y .+ δy
end

function ein_step(x, y, dt)
    N = length(x)
    κ, nx, ny = curv_normal(x, y)
    ds = find_ds(x, y)
    
    ux, uy = compute_vel(x, y)

    # 1. Base 2nd derivative matrices
    Dx2 = ∂2x_∂2s_mat(N, ds)
    Dy2 = ∂2y_∂2s_mat(N, ds) 

    # 2. 4th derivative matrices (Sparse and strictly satisfying quadrant symmetry)
    # Dx4 and Dy4 are positive definite matrices. 
    Dx4 = Dx2 * Dx2
    Dy4 = Dy2 * Dy2

    # 3. Scaling the Artificial Damping
    # To match the k^3 physics at the grid scale, our k^4 damping matrix 
    # needs to be scaled by the minimum grid spacing.
    ds_min = minimum(ds)
    c_damp = 30 # Tunable parameter (0.1 to 1.0 usually works perfectly)
    
    # λ has the correct units to balance the explicit velocity RHS
    λ = dt * c_damp * ds_min

    # 4. Implicit L matrices: (I + λ D^4)
    # Because D^4 is positive definite, adding it to the Identity matrix 
    # creates a highly stable, easily invertible matrix.
    Lx_mat = spdiagm(0 => ones(N)) .+ (λ * Dx4)
    Ly_mat = spdiagm(0 => ones(N)) .+ (λ * Dy4)

    # 5. The EIN Mathematical Cancellation
    # In EIN, we solve: (I - dt*L)*δx = dt*u + dt*L*x^n - dt*L*x^n
    # The L*x^n terms perfectly cancel! We only need to solve for the increment δx.
    δx = Lx_mat \ (dt * ux)
    δy = Ly_mat \ (dt * uy)

    return x .+ δx, y .+ δy
end

function zwei_step(x,y,dt)
    x1, y1 = ein_step(x,y,dt)

    x21, y21 = ein_step(x,y,dt/2)
    x22, y22 = ein_step(x21,y21,dt/2)

    new_x, new_y = 2*x22 .- x1, 2*y22 .- y1 

    # Define your tolerances
    abstol = 1e-20  # The absolute floor (stops division by zero)

    # Compute the acceptable scale for every point
    scale_x = abstol .+  max.(abs.(x22), abs.(x1))
    scale_y = abstol .+ max.(abs.(y22), abs.(y1))

    # Calculate the normalized error at every point
    norm_err_x = abs.(x22 .- x1) ./ scale_x
    norm_err_y = abs.(y22 .- y1) ./ scale_y

    # The total error is the single worst-offending point in the whole domain
    error = max(maximum(norm_err_x), maximum(norm_err_y))
    #error = abs(x22[1]-x1[1])/new_x[1]  #max(maximum(abs.((x22 .- x1)./new_x)) , maximum(abs.((y22 .- y1)./new_y)))

    return new_x, new_y, error
end

function two_step(x,y,dt)
    x1, y1 = frw_euler_dt(x,y,dt)

    x21, y21 = frw_euler_dt(x,y,dt/2)
    x22, y22 = frw_euler_dt(x21,y21,dt/2)

    new_x, new_y = 2*x22 .- x1, 2*y22 .- y1 
    error = abs(x22[1]-x1[1])/new_x[1]   #max(maximum(abs.((x22 .- x1)./new_x)) , maximum(abs.((y22 .- y1)./new_y)))

    return new_x, new_y, error
end




function tip_plot(x,y)
    scatter(x[1:50],y[1:50],ms=:1,mc=:black,aspectratio=:equal)
    scatter!(x[1:50],-y[1:50],ms=:1,mc=:black)
end

mutable struct state
    x::Vector{Float64}
    y::Vector{Float64}
    t::Float64
end

function regrid(x,y,κ; r=2.0)
    rB = x[1]
    N = length(x)

    #we have to pad everything so splining works well
    pad_len = 3 

    x_start = x[(pad_len+1):-1:2]
    y_start = -y[(pad_len+1):-1:2]

    x_end = -x[(N-1):-1:(N-pad_len)]
    y_end = y[(N-1):-1:(N-pad_len)]

    x_padded = vcat(x_start, x, x_end)
    y_padded = vcat(y_start, y, y_end)

    ds=hele.find_ds(x_padded,y_padded)

    arc_padded = vcat([0.0], cumsum(ds))

    # Identify where our 1st quadrant data lives
    arc_start = arc_padded[pad_len + 1]
    arc_end   = arc_padded[pad_len + N]

    #Spline everything 
    xspline = Dierckx.Spline1D(arc_padded,x_padded;k=3,s=0.0)
    yspline = Dierckx.Spline1D(arc_padded,y_padded;k=3,s=0.0)

    #compute the maximum curvature 
    κ_max_dex = argmax(abs.(κ))
    rc_min = 1/abs(κ[κ_max_dex]) #minimum radius of curvature
    s_loc = arc_padded[pad_len+κ_max_dex] #location of minimum

    ds_max = arc_end/100
    ds_min = min(pi*rc_min/16,ds_max) #compute the minimum grid size

    #the initial points
    new_s = [arc_start,arc_start+ds_min] #the arclength is initially zero
    new_x = [rB,xspline(new_s[2])] #start at the bridge 
    new_y = [0.0,yspline(new_s[2])] #the bridge must be at y=0

    #Use a constant fine grid until just past the minimum radius of curvature
    while new_s[end] < max(s_loc+40*ds_min, arc_start+40*ds_min)
        push!(new_s,new_s[end]+ds_min)
        push!(new_x, xspline(new_s[end]))
        push!(new_y, yspline(new_s[end]))

        #break the loop if we go to far out
        if new_y[end] > 1.0 
            break
        end 

    end

      #Increase the grid spacing till we are at the maximum spacing 
    #Then go to the end
    while new_s[end] < arc_end-ds_max 
        ds_1 = min(ds_max, r*(new_s[end]-new_s[end-1]))
        x_test = xspline(new_s[end]+ds_1)
        if x_test <=0
            break 
        elseif new_s[end]+ds_1 > arc_end
            break
        elseif yspline(new_s[end]+ds_1) > y[end]
            break
        else
            push!(new_s,new_s[end] + ds_1)
            push!(new_x, x_test)
            push!(new_y, yspline(new_s[end]))
        end
    end

    if new_x[end]<ds_max/2
        #remove the final elements
        new_x = new_x[1:end-1]
        new_y = new_y[1:end-1]
    end

    #add the points at the top of the corne
    push!(new_x,0.0)
    push!(new_y,y[end])
    
    
    return  new_x, new_y
    
end

function drop_plot(x,y)
    xfull = [x; - reverse(x); -x[2:end]; reverse(x)[1:end-1]]
    yfull = [y; reverse(y); - y[2:end]; -reverse(y)[1:end-1]]

    plot([xfull],[yfull],aspectratio=:equal,
    seriestype = :shape,   # filled polygon
    fillcolor = :skyblue1,
    fillalpha = 0.7,
    linecolor = :darkblue,
    linewidth = 2,xlims=(-2.5,2.5),ylims=(-2.5,2.5))
end

function chunk!(data, n_chunks, per_chunk,dt)
    error=0.
    for j in 1:n_chunks
        t1 = data[end].t 
        x1,y1 = data[end].x, data[end].y
        for j in 1:per_chunk
            testx, testy,error = zwei_step(x1,y1,dt)
            if error > 1e-3
                dt/=2 
            else
                x1, y1 = testx, testy
                κ,nx,ny = curv_normal(x1,y1)
                x1,y1 = regrid(x1, y1,κ);
                t1+=dt
                dt*=1.2
            end
            #break every 10 % increase and store a point
            #otherwise we may skip some important ones
            if x1[1] > 1.1*data[end].x[1]
                break
            end
        end
        #save the point if we have an increase
        if x1[1]!= data[end].x[1]
            push!(data,state(x1,y1,t1))
        end
    end
end

end
