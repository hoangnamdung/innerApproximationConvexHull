#=
    Copyright © 2025, Nam-Dũng Hoang, Nguyen Kieu Linh, and Hoang Xuan Phu
    This code was implemented for the paper titled "Inner δ-Approximation of
    the Convex Hull of Finite Sets"
=#

using JLD2

function arrayCompare(first, second)
    first_set = Set(eachrow(first))
    second_set = Set(eachrow(second))

    return issetequal(first_set, second_set)
end

function main()
    sizes = [100_000, 215_000]

    setNumbers = 10

    # dataType: 1 for ellipse type, 2 for rectangle type, 3 for nonagon type
    dataType = 1
    dataTypeName = ["ellipse", "rectangle", "nonagon"]

    resultDirectory = string("results/",dataTypeName[dataType],"/")
    for i in eachindex(sizes)
        for k in 1:setNumbers
            println()
            instanceName = string(dataTypeName[dataType],"_",sizes[i],"_", k)

            inner_file = string(resultDirectory, instanceName, "_Alg_2_rho_0.jld2")
            qHull_file = string(resultDirectory, instanceName, "_QHull.jld2")

            inner_vertices = load(inner_file, "V")
            qHull_vertices = load(qHull_file, "V")

            println("Checking ", instanceName)
            println("\t innerCH_Alg_2 vs QHull, results are identical: ", arrayCompare(inner_vertices, qHull_vertices))
        end
    end
end

main()
