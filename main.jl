#=
    Copyright © 2025, Nam-Dũng Hoang, Nguyen Kieu Linh, and Hoang Xuan Phu
    This code was implemented for the paper titled "Inner δ-Approximation of
    the Convex Hull of Finite Sets"
=#

using Random
using Distributions
using CGAL
using IterTools
using BenchmarkTools
using TimerOutputs

include("utils.jl")
include("innerCH.jl")
include("quickHull.jl")

function main()
    sizes = [100_000]
    # sizes = [100_000, 215_000, 462_000, 1_000_000, 2_150_000, 4_620_000, 10_000_000, 21_500_000, 46_200_000]
    rho = 0
    setNumbers = 10

    # set benchmarking = true if want to benchmark
    benchmarking = true
    # only export results if exportResult = true and benchmarking = false
    exportResult = true

    # dataType: 1 for ellipse type, 2 for rectangle type, 3 for nonagon type
    dataType = 1
    dataTypeName = ["ellipse", "rectangle", "nonagon"]

    resultDirectory = string("results/",dataTypeName[dataType],"/")
    dataDirectory = string("data/",dataTypeName[dataType],"/")

    # diamX is not important when rho = 0
    diamX = 2000           #for dataType = 1
    if dataType == 2
        diamX = 2332.38    #for dataType = 2
    elseif dataType == 3
        diamX = 2059.1263  #for dataType = 3
    end
    delta = rho * diamX

    instanceNames = Vector{String}(undef, length(sizes)*setNumbers)
    runningTimeQH = Matrix{Float64}(undef, length(instanceNames), 7)
    runningTimeInnerCH_Alg_2 = Matrix{Float64}(undef, length(instanceNames),7)
    runningTimeAklToussaint = Matrix{Float64}(undef, length(instanceNames), 7)
    runningTimeBykat = Matrix{Float64}(undef, length(instanceNames), 7)
    runningTimeEddy = Matrix{Float64}(undef, length(instanceNames), 7)
    runningTimeGrahamAndrew = Matrix{Float64}(undef, length(instanceNames), 7)

    index = 0
    for i in eachindex(sizes)
        for k in 1:setNumbers
            filename = string(dataDirectory, dataTypeName[dataType],"_",sizes[i],"_",k,".jld2")
            points = loadData(filename)

            pointsQH = permutedims(points)
            points_CGAL = Vector{Point2}(undef, size(pointsQH, 1))

            for t in 1:length(points_CGAL)
                points_CGAL[t] = Point2(pointsQH[t,1],pointsQH[t,2])
            end

            index = index+1
            instanceNames[index] = string(dataTypeName[dataType],"_",sizes[i],"_",k)

            println()
            println("Consider instance ", instanceNames[index])

            if benchmarking
                println("Qhull")
                bmQH = run(@benchmarkable callQHull($pointsQH) samples= 13 seconds=10000)
                runningTimeQH[index,:] = report(bmQH, 3)

                println("Akl Toussaint heuristic")
                bmAklToussaint = run(@benchmarkable ch_akl_toussaint($points_CGAL) samples=13 seconds=10000)
                runningTimeAklToussaint[index,:] = report(bmAklToussaint, 3)

                println("Bykat-quickHull-non-recursive version")
                bmBykat = run(@benchmarkable ch_bykat($points_CGAL) samples=13 seconds=10000)
                runningTimeBykat[index,:] = report(bmBykat, 3)

                println("Eddy - a version of quickHull algorithm")
                bmEddy = run(@benchmarkable ch_eddy($points_CGAL) samples=13 seconds=10000)
                runningTimeEddy[index,:] = report(bmEddy, 3)

                println("Graham Andrew - Graham' scan")
                bmGrahamAndrew = run(@benchmarkable ch_graham_andrew($points_CGAL) samples=13 seconds=10000)
                runningTimeGrahamAndrew[index,:] = report(bmGrahamAndrew, 3)

                println("innerCH_Alg_2:")
                bmInnerCH_Alg_2 = run(@benchmarkable inner_convex_2($points,$delta) samples=13 seconds=10000)
                runningTimeInnerCH_Alg_2[index,:] = report(bmInnerCH_Alg_2, 3)
            else
                exportFileQH = string(resultDirectory, instanceNames[index], "_QHull")
                callQHull(pointsQH, exportResult, exportFileQH)

                exportFileInnerCH_Alg_2 = string(resultDirectory, instanceNames[index],"_Alg_2_rho_",rho)
                inner_convex_2(points, delta, exportResult, exportFileInnerCH_Alg_2)
            end
        end
    end

    if benchmarking
        baseName = string(resultDirectory)
        # Algorithm 2
        exportReport(instanceNames, runningTimeInnerCH_Alg_2, string(baseName, "1_innerCH_Alg2_rho","_",rho,".csv"))
        # Quick hull
        exportReport(instanceNames, runningTimeQH, string(baseName,"2_qHull_running_time.csv"))
        # Akl Toussaint
        exportReport(instanceNames, runningTimeAklToussaint, string(baseName,"3_Akl_Toussaint_running_time.csv"))
        # Bykat
        exportReport(instanceNames, runningTimeBykat, string(baseName,"4_Bykat_running_time.csv"))
        # Eddy
        exportReport(instanceNames, runningTimeEddy, string(baseName,"5_Eddy_running_time.csv"))
        # Graham Andrew
        exportReport(instanceNames, runningTimeGrahamAndrew, string(baseName,"6_Graham_Andrew_running_time.csv"))
    end
end

main()
