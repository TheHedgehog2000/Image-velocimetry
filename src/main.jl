include("piv.jl")
using .PIV
using ProfileView

params = PIV.pivSettings(image_folder="/mnt/h/Dispersal/WT_replicate1_processed/", mpass=3, 
                         windows=[(64,64,16), (32,32,8), (16,16,4)], 
                         overlaps=[(32,32,8), (16,16,4), (8,8,2)], s2n_validate=true, median_validate=true, global_validate=true,
                        std_validate=false, replace_method="zero")
PIV.piv(params)
#params_test = PIV.pivSettings(image_folder="../tests/", mpass=3, 
#                         windows=[(64,64,16), (32,32,8), (16,16,4)], 
#                         overlaps=[(32,32,8), (16,16,4), (8,8,2)], s2n_validate=true, median_validate=true, global_validate=true,
#                        std_validate=false, replace_method="zero")
#@profview PIV.piv(params_test)
#@profview PIV.piv(params_test)
