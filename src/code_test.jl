using FileIO 
using NPZ
using CairoMakie

x, y, z, u, v, w, flags = load("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/piv_results_2.jld2", 
                              "x", "y", "z", "u", "v", "w", "flags")
x_p = npzread("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/x.npy")
y_p = npzread("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/y.npy")
z_p = npzread("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/z.npy")
u_p = npzread("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/u.npy")
v_p = npzread("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/v.npy")
w_p = npzread("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/w.npy")
flags_p = npzread("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/flags.npy") 

flags_qplot = transpose(flags[:,:,8])
u_qplot = transpose(u[:,:,8])
v_qplot = transpose(v[:,:,8])
@show size(u_qplot)
@show size(flags_qplot)
#u_qplot[flags_qplot] .= 0.0
#v_qplot[flags_qplot] .= 0.0

u_p = transpose(u_p[2,:,:])
v_p = transpose(v_p[2,:,:])
flags_p = transpose(flags_p[2,:,:])
u_p[flags_p] .= 0.0  
v_p[flags_p] .= 0.0  

# 2D quiver plot of u_qplot, v_qplot
fig = arrows((x .- 1) ./ 8, (y .- 1) ./ 8, u_p, v_p)
save("/Users/jojo/Downloads/JuliaPIV/tests/Displacements/quiver_plot_python_postinterp.png", fig)
