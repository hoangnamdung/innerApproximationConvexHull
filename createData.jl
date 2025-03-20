#=
    Copyright © 2025, Nam-Dũng Hoang, Nguyen Kieu Linh, and Hoang Xuan Phu
    This code was implemented for the paper titled "Inner δ-Approximation of
    the Convex Hull of Finite Sets"
=#

using Random
using Distributions
using JLD2
include("utils.jl")

function createEllipseData(n::Int64, a::Number, b::Number)
    θ = 2π * rand(n)
    r = sqrt.(rand(n))
    x = a .* r .* cos.(θ)
    y = b .* r .* sin.(θ)
    points = vcat(x', y')
    return points
end

function createRectangleData(n::Int64, x_min::Number, x_max::Number,
                                       y_min::Number, y_max::Number)
    @assert x_min < x_max
    @assert y_min < y_max

    x_mean = (x_max + x_min) / 2
    y_mean = (y_max + y_min) / 2

    ratio = (x_max - x_min) / (y_max - y_min)

    points = rand(Uniform(y_min - y_mean, y_max - y_mean), 2, n)

    for i in 1:n
        points[1, i] *= ratio
        points[1, i] += x_mean
        points[2, i] += y_mean
    end

    return points
end

function rotatePoints(points::Matrix{Float64}, theta::Float64)
    rotation_matrix = [cos(theta) -sin(theta);
                        sin(theta)  cos(theta)]
    return rotation_matrix * points
end

nonagon = [
    -900 -600;
    -600 -900;
    100 -750;
    600 -400;
    900 100;
    800 500;
    400 900;
    -300 800;
    -800.0 0.0
]

function triangleArea(A::Matrix{Float64})
    M = [A[1,1] A[2,1] A[3,1];
         A[1,2] A[2,2] A[3,2];
         1       1       1]
    return 0.5 * abs(det(M))
end

function triangulate(polygon::Matrix{Float64})
    n = size(polygon, 1)
    triangles = Matrix{Float64}[]
    for i in 2:(n-1)
        push!(triangles, vcat(polygon[1, :]', polygon[i, :]', polygon[i+1, :]'))
    end
    return triangles
end

function randomPointInTriangle(T::Matrix{Float64})
    r1, r2 = rand(), rand()
    sqrt_r1 = sqrt(r1)
    x = (1 - sqrt_r1) * T[1,1] + (sqrt_r1 * (1 - r2)) * T[2,1] + (sqrt_r1 * r2) * T[3,1]
    y = (1 - sqrt_r1) * T[1,2] + (sqrt_r1 * (1 - r2)) * T[2,2] + (sqrt_r1 * r2) * T[3,2]
    return [x, y]
end

function createNonagonData(nonagon::Matrix{Float64}, N::Int)
    triangles = triangulate(nonagon)
    areas = [triangleArea(T) for T in triangles]
    total_area = sum(areas)
    proportions = areas ./ total_area
    num_points = round.(Int, N * proportions)

    points = Matrix{Float64}(undef, 2, 0)

    for (T, count) in zip(triangles, num_points)
        new_points = hcat([randomPointInTriangle(T) for _ in 1:count]...)
        points = hcat(points, new_points)
    end

    return points
end

function main()
    sizes = [100_000]
    # sizes = [100_000, 215_000, 462_000, 1_000_000, 2_150_000, 4_620_000, 10_000_000, 21_500_000, 46_200_000]
    setNumbers = 10

    dataTypeName = ["ellipse", "rectangle", "nonagon"]
    theta = π/4
    Random.seed!(42)

    for dataType in 1:3
        dataDirectory = string("data/", dataTypeName[dataType], "/")
        for i in eachindex(sizes)
            for k in 1:setNumbers
                if (dataType == 1)
                    a = 1000
                    b = 600
                    println("create data in an ellipse")
                    points1 = createEllipseData(sizes[i], a, b)
                    points = rotatePoints(points1, theta)
                elseif (dataType == 2)
                    x_min = -1000
                    x_max = 1000
                    y_min = -600
                    y_max = 600
                    println("create data in a rectangle")
                    points1 = createRectangleData(sizes[i], x_min, x_max, y_min, y_max)
                    points= rotatePoints(points1, theta)
                else
                    println("create data in a nonagon")
                    points = createNonagonData(nonagon, sizes[i])
                end
                dataFile = string(dataDirectory, dataTypeName[dataType], "_", sizes[i], "_", k, ".jld2")
                println("\t Exporting ", dataFile)
                jldsave(dataFile; points)
            end
        end
    end
end

main()
