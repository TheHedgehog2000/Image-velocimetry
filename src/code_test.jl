using FileIO 
using NPZ
using CairoMakie
using NaturalSort

#x, y, z, u, v, w, s2n, flags = load("../tests/Displacements/piv_results_2.jld2", 
#                              "x", "y", "z", "u", "v", "w", "s2n", "flags")
#x_p = npzread("../tests/Displacements/x.npy")
#y_p = npzread("../tests/Displacements/y.npy")
#z_p = npzread("../tests/Displacements/z.npy")
#u_p = npzread("../tests/Displacements/u.npy")
#v_p = npzread("../tests/Displacements/v.npy")
#w_p = npzread("../tests/Displacements/w.npy")
#s2n_p = npzread("../tests/Displacements/s2n.npy")
#flags_p = npzread("../tests/Displacements/flags.npy") 

#flags_qplot = transpose(flags[:,:,8])
#u_qplot = transpose(u[:,:,8])
#v_qplot = transpose(v[:,:,8])
#u_qplot[flags_qplot] .= 0.0
#v_qplot[flags_qplot] .= 0.0

#u_p = transpose(u_p[2,:,:])
#v_p = transpose(v_p[2,:,:])
#flags_p = transpose(flags_p[2,:,:])
#u_p[flags_p] .= 0.0  
#v_p[flags_p] .= 0.0  

# 2D quiver plot of u_qplot, v_qplot
#fig = arrows((x .- 1) ./ 8, (y .- 1) ./ 8, u_p, v_p)
#save("../tests/Displacements/quiver_plot_python_postinterp.png", fig)

function main()
    folder = "/mnt/h/Dispersal/WT_replicate1_processed/Displacements/"
    files = sort([f for f in readdir(folder, join=true) if occursin("piv_results", f)], lt=natural)

    x, y, u_dummy = load(files[1], "x", "y", "u")

    u_p = zeros(Float32, size(u_dummy))
    v_p = zeros(Float32, size(u_dummy)) 

    for i in 1:10
        u, v, flags = load(files[i], "u", "v", "flags")
        u_p += u
        v_p += v
    end

    fig = arrows((x .- 1) ./ 8, (y .- 1) ./ 8, transpose(u_p[:,:,8]), transpose(v_p[:,:,8]))
    save("$folder/quiver_test_inverted.png", fig)
end
main()
