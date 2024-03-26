include("piv.jl")
using .PIV

params = PIV.pivSettings(image_folder="/Users/jojo/Downloads/JuliaPIV/tests", mpass=3, 
                         windows=[(64,64,16), (32,32,8), (16,16,4)], 
                         overlaps=[(32,32,8), (16,16,4), (8,8,2)], s2n_validate=false)
PIV.piv(params)
