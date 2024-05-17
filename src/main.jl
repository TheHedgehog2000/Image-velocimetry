include("piv.jl")
using .PIV
using ProfileView

folders = ["/mnt/h/Dispersal/rbmB_replicate1_processed/", "/mnt/h/Dispersal/rbmB_replicate2_processed/", 
    "/mnt/h/Dispersal/rbmB_replicate3_processed/", "/mnt/h/Dispersal/rbmB_replicate4_processed/", 
    "/mnt/h/Dispersal/rbmB_replicate5_processed/",  "/mnt/h/Dispersal/cheY_replicate1_processed/", 
    "/mnt/h/Dispersal/cheY_replicate2_processed/", "/mnt/h/Dispersal/cheY_replicate3_processed/",
   "/mnt/h/Dispersal/cheY_replicate4_processed/", "/mnt/h/Dispersal/cheY_replicate5_processed/"]

for folder in folders
    params = PIV.pivSettings(image_folder=folder, mpass=3, 
                             windows=[(64,64,16), (32,32,8), (16,16,4)], 
                             overlaps=[(32,32,8), (16,16,4), (8,8,2)], s2n_validate=true, median_validate=true, global_validate=true,
                            std_validate=false, replace_method="zero")
    PIV.piv(params)
end
