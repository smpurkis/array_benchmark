using Distributed
# using SharedArrays


function compute_array(m, n)
    x = zeros(m, n)
    for i = 0:m - 1
        for j = 0:n - 1
            x[i+1, j+1] = Int32(i*i + j*j)
        end
    end
    return x
end

function compute_array_threaded(m, n)
    x = zeros(m, n)
    Threads.@threads for i = 0:m - 1
        for j = 0:n - 1
            x[i+1, j+1] = Int32(i*i + j*j)
        end
    end
    return x
end

# function compute_array_distributed(m, n)
#     x = zero(SharedArray{Int64}((m, n)))
#     @distributed for i = 0:m - 1
#         for j = 0:n - 1
#             x[i+1, j+1] = Int32(i*i + j*j)
#         end
#     end
#     return x
# end


function compute_array_list(m, n)
    x = [Int32(i*i + j*j) for i in 0:m-1, j in 0:n-1]
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

l = Int16(15000)
k = Int16(15000)
n_loop = 5
# compute_array(10, 10)
# compute_array_threaded(10, 10)
# p = compute_array_distributed(10, 10)
# println(p)
compute_array_list(Int16(150), Int16(150))
# compute_array_list_unitrange(Int16(150), Int16(150))
# compute_array_list_steprange(Int16(150), Int16(150))


# s = time()
# for i = 1:n_loop
#     p = compute_array(l, k)
#     println(p[l, k])
#     println(typeof(p[l, k]))
# end
# println(time() - s)

# s = time()
# for i = 1:n_loop
#     p = compute_array_threaded(l, k)
#     println(p[l, k])
#     println(typeof(p[l, k]))
# end
# println(time() - s)

# s = time()
# for i = 1:n_loop
#     p = compute_array_distributed(Int64(l), Int64(k))
# #     println(p)
# #     println(p[l, k])
#     println(typeof(p[l, k]))
#     end
# println(time() - s)

s = time()
for i = 1:n_loop
    p = compute_array_list(l, k)
    println(p[l, k])
    # println(typeof(p[l, k]))
end
println(time() - s)
#
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
