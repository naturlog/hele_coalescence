cd(@__DIR__)
include("hele_faster.jl"), include("hele_linsys.jl")
using .hele,  .hele_fast , Plots, JLD2, BenchmarkTools, CurveFit



rb = 1e-1;
x0,y0=hele_fast.hopper_init(rb);
κ,nx,ny= hele.curv_normal(x0,y0);
x0,y0 = hele.regrid(x0,y0,κ)
state0 = hele_fast.state(x0,y0,0.0,0.0);
data = [state0];
dt=1e-12*rb^2;
res=1e-3;


@time dt = hele_fast.chunk!(data,5,100,dt,res)

data[end].x[1]/data[1].x[1], length(data[end].x)

hele_fast.tip_plot(data[end].x,data[end].y)
hele.drop_plot(data[end].x,data[end].y)



#@save "rb_1e_6.JLD2" data

rbs = [data[i].x[1] for i in 5:length(data)]
times = [data[i].t for i in 5:length(data)]
rdot = [ (rbs[i+1]-rbs[i])/(times[i+1]-times[i]) for i in 1:length(rbs)-1 ];
R0c = 2.35;

plot(rbs[sdex:end], (3/13)*(R0c^(3/13)) .* rbs[sdex:end] .^ (-10/3),lw=:3,linecolor=:black,label="Theory")
scatter!(rbs[sdex:2:end-1],rdot[sdex:2:end-1],yaxis=:log,xaxis=:log,ylabel=L"\dot{r}_0",xlabel=L"r_0",ms=:4,markercolor=:red,markerstrokecolor=:red,markerstrokewidth=:0,label="Simulation",legendposition=:topright)
savefig("r0_rdot.pdf")

plot(ti[sdex:end], (3/13)*(R0c^(3/13)) .* rbs[sdex:end] .^ (-10/13),lw=:3,linecolor=:black,label="Theory")
scatter(times[sdex:2:end],rdot[sdex:2:end],yaxis=:log,xaxis=:log,ylabel=L"r_0",xlabel=L"t",ms=:4,markercolor=:red,markerstrokecolor=:red,markerstrokewidth=:0,label="Simulation",legendposition=:topright)
savefig("r0_rdot.pdf")

power_fit(times[sdex:fdex],rdot[sdex:fdex])



sdex = findfirst(x->x>2e-5,rbs)
fdex = findfirst(x->x>5e-3,rbs)
fit_p = power_fit(times[sdex:fdex],rbs[sdex:fdex])
plot(times[sdex:end],2.35 .* times[sdex:end] .^ (3/13),lw=:3,linecolor=:black,label="Theory")
scatter!(times[sdex:2:end],rbs[sdex:2:end],yaxis=:log,xaxis=:log,ylabel=L"r_0",xlabel=L"t",ms=:4,markercolor=:red,markerstrokecolor=:red,markerstrokewidth=:0,label="Simulation",legendposition=:topleft)
savefig("r0_scaling_2.pdf")

3/13

rbs[sdex+70]



anim = @animate for i in 1:length(data)
    hele.drop_plot(data[i].x,data[i].y)
    end 
gif(anim, "coalesce.gif", fps=12)

