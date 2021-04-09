using LoopVectorization

# addprocs(6)

function compute_array(m, n)
    @inbounds x = zeros(Int32, (m, n))
    @inbounds for i = 0:m - 1
        for j = 0:n - 1
            x[i+1, j+1] = i*i + j*j
        end
    end
    return x
end

function compute_array_threaded(m, n)
    @inbounds x = zeros(Int32, (m, n))
    @inbounds Threads.@threads for i = 0:m - 1
        for j = 0:n - 1
            x[i+1, j+1] = i*i + j*j
        end
    end
    return x
end


function compute_array_list(m, n)
    x = @inbounds [Int32(i*i + j*j) for i in 0:m-1, j in 0:n-1]
    return x
end


function compute_array_fill(m, n)
    @inbounds x = [Int32(i) for i in 0:m-1].^2
    @inbounds y = [Int32(j) for j in 0:n-1].^2
    @inbounds return broadcast(+, x, y')
end

function compute_array_collect(m, n)
    return collect(Int32, 0:m-1).^2 .+ (collect(Int32, 0:n-1).^2)'
end

function compute_array_strided(m, n)
    x = collect(Int32, 0:m-1).^2 .+ zeros(Int8, n)'
    y = zeros(Int8, m) .+ (collect(Int32, 0:n-1).^2)'
    t = x .+ y
    t = reshape(t, (m, n))
    return t
end


# any type of of Int 
function compute_array_normal_swap_lv_threaded_generic(m::T, n::T) where T<:Integer
    x = Matrix{T}(undef,m,n)
    Threads.@threads for j = 0:n - 1
        @avx for i = 0:m - 1
            @inbounds x[i+1, j+1] = i*i + j*j
        end
    end
    return x
end
#
# function compute_array_list_unitrange(m::Int16, n::Int16)
#     m_loop = Array(UnitRange{Int32}.(Int16(0),m-Int16(1)))
#     n_loop = Array(UnitRange{Int32}.(Int16(0),n-Int16(1)))
#     x = [Int32(i*i + j*j) for i in m_loop, j in n_loop]
#     return x
# end
#
# function compute_array_list_steprange(m::Int16, n::Int16)
#     m_loop = Array{Int32}((Int16(0):m-Int16(1)))
#     n_loop = Array{Int32}((Int16(0):n-Int16(1)))
#     x = [Int32(i*i + j*j) for i in m_loop, j in n_loop]
#     return x
# end


# function compute_array_list(m, n)
#     x = collect(1:m+1) .* reshape(collect(1:n+1), (n+1, 1))
#     return x
# end

l = Int32(10000)
k = Int32(10000)
t0 = Int32(150)
n_loop = 5
compute_array(Int32(150), Int32(150))
# compute_array_threaded(Int32(150), Int32(150))
# compute_array_distributed(t0, t0)
# # println(p)
# compute_array_list(Int32(150), Int32(150))
compute_array_fill(Int32(150), Int32(150))
compute_array_collect(Int32(150), Int32(150))
compute_array_normal_swap_lv_threaded_generic(Int32(150), Int32(150))
# compute_array_strided(Int32(150), Int32(150))


# compute_array_list_unitrange(Int16(150), Int16(150))
# compute_array_list_steprange(Int16(150), Int16(150))


# s = time()
# @time for i = 1:n_loop
#     p = compute_array(l, k)
#     println(p[l, k])
# #     println(typeof(p[l, k]))
# end
# println(time() - s)

# @time for i = 1:n_loop
#     p = compute_array_threaded(l, k)
#     println(p[l, k])
#     # println(typeof(p[l, k]))
# end

# s = time()
# @time for i = 1:n_loop
#     p = compute_array_list(l, k)
#     println(p[l, k])
#     # println(typeof(p[l, k]))
# end
# println(time() - s)

@time for i = 1:n_loop
    p = compute_array_fill(l, k)
    println(p[l, k])
    # println(typeof(p[l, k]))
end
#
@time for i = 1:n_loop
    p = compute_array_collect(l, k)
    println(p[l, k])
    # println(typeof(p[l, k]))
end

@time for i = 1:n_loop
    p = compute_array_normal_swap_lv_threaded_generic(l, k)
    println(p[l, k])
    # println(typeof(p[l, k]))
end



# @time for i = 1:n_loop
#     p = compute_array_strided(l, k)
#     println(p[l, k])
#     # println(typeof(p[l, k]))
# end

# s = time()
# for i = 1:n_loop
#     p = compute_array_list_unitrange(l, k)
#     println(p[l, k])
#     # println(typeof(p[l, k]))
# end
# println(time() - s)
#
# s = time()
# for i = 1:n_loop
#     p = compute_array_list_steprange(l, k)
#     println(p[l, k])
#     # println(typeof(p[l, k]))
# end
# println(time() - s)
