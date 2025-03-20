#=
    Copyright © 2025, Nam-Dũng Hoang, Nguyen Kieu Linh, and Hoang Xuan Phu
    This code was implemented for the paper titled "Inner δ-Approximation of
    the Convex Hull of Finite Sets"
=#

using Random
using LinearAlgebra
using TimerOutputs
using LoopVectorization
using Distributions

include("utils.jl")

# Create a TimerOutput, this is the main type that keeps track of everything.
const TimeOutput = TimerOutput()

@inline function creat_V_Etest(X)
    min_x = X[1, 1]
    max_x = X[1, 1]
    min_y = X[2, 1]
    max_y = X[2, 1]

    q12 = X[2, 1]
    q21 = X[1, 1]
    q32 = X[2, 1]
    q41 = X[1, 1]
    local len = size(X,2)
    @avx for i in 2:len
        x = X[1, i]
        y = X[2, i]

        x_lt_min_x = x < min_x
        x_eq_min_x = x == min_x

        x_gt_max_x = x > max_x
        x_eq_max_x = x == max_x

        y_lt_min_y = y < min_y
        y_eq_min_y = y == min_y

        y_gt_max_y = y > max_y
        y_eq_max_y = y == max_y

        min_x = x_lt_min_x ? x : min_x
        q32 = x_lt_min_x ? y : q32
        q32 = x_eq_min_x ? min(q32, y) : q32

        max_x = x_gt_max_x ? x : max_x
        q12 = x_gt_max_x ? y : q12
        q12 = x_eq_max_x ? max(q12, y) : q12

        min_y = y_lt_min_y ? y : min_y
        q41 = y_lt_min_y ? x : q41
        q41 = y_eq_min_y ? max(q41, x) : q41

        max_y = y_gt_max_y ? y : max_y
        q21 = y_gt_max_y ? x : q21
        q21 = y_eq_max_y ? min(q21, x) : q21
    end
    V = Vector{Vector{Float64}}(undef, 4)
    V[1] = [max_x, q12]
    V[2] = [q21, max_y]
    V[3] = [min_x, q32]
    V[4] = [q41, min_y]

    E_test = Vector{Matrix{Float64}}(undef,4)

    E_test[1] = [V[1] V[2]]
    E_test[2] = [V[2] V[3]]
    E_test[3] = [V[3] V[4]]
    E_test[4] = [V[4] V[1]]

    return V, E_test
end

#Functions prepared for the first 4 sets

@inline function find_first_4_sets(X, V)
    local epsilon = 10^-8
    local d = [V[2] - V[1], V[3] - V[2], V[4] - V[3], V[1] - V[4]]
    local dbot = [[d[1][2] -d[1][1]];
            [d[2][2] -d[2][1]];
            [d[3][2] -d[3][1]];
            [d[4][2] -d[4][1]]]
    local b = [dot(dbot[i, :], V[i]) for i in 1:4]
    local X_sets = Vector{Matrix{Float64}}(undef,4)
    mask = [Vector{Bool}(undef,size(X,2)) for _ in 1:4]
    local len = size(X,2)
    @avx for j in 1:len
        a1 = dbot[1,1]*X[1,j] + dbot[1,2]*X[2,j]
        ta1 = a1 - b[1]
        mask[1][j] = ta1 > epsilon

        a2 = dbot[2,1]*X[1,j] + dbot[2,2]*X[2,j]
        ta2 = a2 - b[2]
        mask[2][j] = ta2 > epsilon

        a3 = dbot[3,1]*X[1,j] + dbot[3,2]*X[2,j]
        ta3 = a3 - b[3]
        mask[3][j] = ta3 > epsilon

        a4 = dbot[4,1]*X[1,j] + dbot[4,2]*X[2,j]
        ta4 = a4 - b[4]
        mask[4][j] = ta4 > epsilon
    end
    X_sets = [X[:,mask[i]] for i in 1:4]

    local f3_delta = Vector{Tuple{Int,Float64}}()
    for i in 1:3
        push!(f3_delta, f1f2f3(X_sets[i],[V[i] V[i+1]]))
    end
    push!(f3_delta, f1f2f3(X_sets[4],[V[4] V[1]]))

    return X_sets, f3_delta
end

@inline function both_X_v_vplus_vhat(X_v_vplus, v, vplus, vhat)
    local epsilon = 10^-8
    local d = [vhat - v, vplus - vhat]
    local dbot_v_vhat = [d[1][2] -d[1][1]]
    local dbot_vhat_vplus = [d[2][2] -d[2][1]]

    local b = [dot(dbot_v_vhat,v), dot(dbot_vhat_vplus , vhat)]
    # local a = [dbot_v_vhat;dbot_vhat_vplus] * X_v_vplus

    local X_sets = Vector{Matrix{Float64}}(undef,2)
    mask = [Vector{Bool}(undef,size(X_v_vplus,2)) for _ in 1:2]
    local len = size(X_v_vplus,2)
    @avx for j in 1:len
        a1 = dbot_v_vhat[1]*X_v_vplus[1,j] + dbot_v_vhat[2]*X_v_vplus[2,j]
        ta1 = a1 - b[1]
        mask[1][j] = ta1 > epsilon

        a2 = dbot_vhat_vplus[1]*X_v_vplus[1,j] + dbot_vhat_vplus[2]*X_v_vplus[2,j]
        ta2 = a2 - b[2]
        mask[2][j] = ta2 > epsilon
    end
    X_sets = [X_v_vplus[:,mask[i]] for i in 1:2]

    # local X_sets = [X_v_vplus[:,a[i,:].-b[i] .> epsilon] for i in 1:2]
    local f3_delta = Vector{Tuple{Int,Float64}}()
    push!(f3_delta, f1f2f3(X_sets[1],[v vhat]))
    push!(f3_delta, f1f2f3(X_sets[2],[vhat vplus]))
    return X_sets,f3_delta
end

# this function computes h1, h2 and h3
@inline function f1f2f3(X, egde_vvpus)
    local d = egde_vvpus[:, 2] - egde_vvpus[:, 1]
    local dbot = [d[2] -d[1]]
    local f2 = Int[1]
    local f1 = -Inf
    # local a = dbot * X
    local len = size(X,2)

    @inbounds @simd for i in 1:len
        a = dbot[1]*X[1,i] + dbot[2]*X[2,i]
        if a > f1
            f1 = a
            f2 = [i]
        elseif a== f1
            push!(f2, i)
        end
    end
    @assert length(f2) >= 1

    local i = pop!(f2)
    if length(f2) == 0
        f3 = i
    else
        maxf3 = d[1]*X[1,i]+d[2]*X[2,i]
        f3 = i
        len = length(f2)
        @avx for j in 1:len
            f2_ = f2[j]
            dot_products = d[1]*X[1,f2_]+d[2]*X[2,f2_]
            com = dot_products > maxf3
            f3 = com ? f2[j] : f3
            maxf3 = com ? dot_products : maxf3
        end
    end
    local Delta = (f1 - (dbot * egde_vvpus[:, 1])[1]) / norm(d)
    return f3, Delta
end

#=======================Algorithm 1=============================
==============================================================#
@inline function find_inner_Alg_1(points, delta)
    #Initialize sets E and X_v_vplus
    E = Vector{Matrix{Float64}}()
    # Define V and E_test
    V, E_test = creat_V_Etest(points)
    # Determine the sets X_v_v_plus corresponding to the edges of Etest
    X_v_vplus1, f3_delta1 = find_first_4_sets(points, V)

    D_test = Set{Tuple{Matrix{Float64}, Matrix{Float64}, Int, Float64}}()
    for i in 1:4
        if f3_delta1[i][2] > delta
            push!(D_test, (E_test[i], X_v_vplus1[i], f3_delta1[i][1],f3_delta1[i][2]))
        else
            push!(E, E_test[i])
        end
    end
    #Step 2
    while !isempty(D_test)
        d = first(D_test)
        delete!(D_test, d)

        point_f3 = d[2][:,d[3]]
        push!(V, point_f3)
        X_vs, f3_delta_ = both_X_v_vplus_vhat(d[2], d[1][:,1], d[1][:,2], point_f3)
        if f3_delta_[1][2] <= delta
            push!(E, [d[1][:,1] point_f3])
        else
            push!(D_test, ([d[1][:,1] point_f3], X_vs[1],f3_delta_[1][1],f3_delta_[1][2]))
        end
        if f3_delta_[2][2] <= delta
            push!(E, [d[1][:,2] point_f3])
        else
            push!(D_test, ([point_f3 d[1][:,2]], X_vs[2],f3_delta_[2][1],f3_delta_[2][2]))
        end
    end
    return V, E
end

#=======================Algorithm 2=============================
==============================================================#
@inline function find_inner_Alg_2(points, delta)
    #Initialize sets E and X_v_vplus
    E = Vector{Matrix{Float64}}()
    # Define V and E_test
    V, E_test = creat_V_Etest(points)
    # Determine the sets X_v_v_plus corresponding to the edges of Etest
    X_v_vplus1, f3_delta1 = find_first_4_sets(points, V)

    D_test = Set{Tuple{Matrix{Float64}, Matrix{Float64}, Int, Float64}}()
    for i in 1:4
        if f3_delta1[i][2] > delta
            push!(D_test, (E_test[i], X_v_vplus1[i], f3_delta1[i][1],f3_delta1[i][2]))
        else
            push!(E, E_test[i])
        end
    end

    #Step 2
    while !isempty(D_test)
        d = first(D_test)
        delete!(D_test, d)

        while true
            point_f3 = d[2][:,d[3]]
            push!(V, point_f3)
            X_vs, f3_delta_ = both_X_v_vplus_vhat(d[2], d[1][:,1], d[1][:,2], point_f3)
            if f3_delta_[1][2] <= delta && f3_delta_[2][2] <= delta
                push!(E, [d[1][:,1] point_f3])
                push!(E, [d[1][:,2] point_f3])
                break
            elseif f3_delta_[1][2] > delta && f3_delta_[2][2] <= delta
                push!(E, [d[1][:,2] point_f3])
                d = ([d[1][:,1] point_f3], X_vs[1],f3_delta_[1][1],f3_delta_[1][2])
            elseif f3_delta_[1][2] <= delta && f3_delta_[2][2] > delta
                push!(E, [d[1][:,1] point_f3])
                d = ([point_f3 d[1][:,2]], X_vs[2],f3_delta_[2][1],f3_delta_[2][2])
            elseif  f3_delta_[1][2] > delta && f3_delta_[2][2] > delta
                push!(D_test, ([point_f3 d[1][:,2]], X_vs[2],f3_delta_[2][1],f3_delta_[2][2]))
                d = ([d[1][:,1] point_f3], X_vs[1],f3_delta_[1][1],f3_delta_[1][2])
            end

        end
    end
    return V, E
end

#==============================================================
==============================================================#

@inline function inner_convex_1(points, sigma, exportCH = false, exportFileInnerCH_Alg_1 = "")
    @timeit TimeOutput "find inner convex Algorithm 1" begin
        V, E = find_inner_Alg_1(points, sigma)
    end
    if exportCH
        exportResult(V, exportFileInnerCH_Alg_1)
    end
end

@inline function inner_convex_2(points, sigma, exportCH = false, exportFileInnerCH_Alg_2 = "")
    @timeit TimeOutput "find inner convex Algorithm 2" begin
        V, E = find_inner_Alg_2(points, sigma)
    end
    if exportCH
        exportResult(V, exportFileInnerCH_Alg_2)
    end
end
