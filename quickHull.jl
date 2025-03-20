#=
    Copyright © 2025, Nam-Dũng Hoang, Nguyen Kieu Linh, and Hoang Xuan Phu
    This code was implemented for the paper titled "Inner δ-Approximation of
    the Convex Hull of Finite Sets"
=#

using TimerOutputs
using QHull
using BenchmarkTools
using CSV
using DataFrames
# using PackageCompiler

include("utils.jl")

const TimeOutputQH = TimerOutput()

function callQHull(points, exportCH =false, exportFile="")
    @timeit TimeOutputQH "calling QHull" begin
        convexHull = chull(points)
    end
    if exportCH
        exportResult(points, convexHull.vertices, exportFile)
    end
end
