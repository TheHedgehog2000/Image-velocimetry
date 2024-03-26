using FileIO 
using NPZ
using CairoMakie

x, y, z, u, v, w, s2n, flags = load("../tests/Displacements/piv_results_2.jld2", 
                              "x", "y", "z", "u", "v", "w", "s2n", "flags")
x_p = npzread("../tests/Displacements/x.npy")
y_p = npzread("../tests/Displacements/y.npy")
z_p = npzread("../tests/Displacements/z.npy")
u_p = npzread("../tests/Displacements/u.npy")
v_p = npzread("../tests/Displacements/v.npy")
w_p = npzread("../tests/Displacements/w.npy")
s2n_p = npzread("../tests/Displacements/s2n.npy")
flags_p = npzread("../tests/Displacements/flags.npy") 

flags_qplot = transpose(flags[:,:,2])
u_qplot = transpose(u[:,:,2])
v_qplot = transpose(v[:,:,2])
u_qplot[flags_qplot] .= 0.0
v_qplot[flags_qplot] .= 0.0

u_p = transpose(u_p[2,:,:])
v_p = transpose(v_p[2,:,:])
flags_p = transpose(flags_p[2,:,:])
u_p[flags_p] .= 0.0  
v_p[flags_p] .= 0.0  

# 2D quiver plot of u_qplot, v_qplot
fig = arrows((x .- 1) ./ 8, (y .- 1) ./ 8, u_p, v_p)
save("../tests/Displacements/quiver_plot_python_postinterp.png", fig)
