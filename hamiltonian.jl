# Calculate the partial derivatives of the Hamiltonian for the n-body problem
#
# TODO: Make sure that these methods are optimised

function Hp(p)
    """Return ∂H/∂p"""
    return p ./ m
end

function norm(x)
    """Return sqrt norm of the vector x"""
    return sqrt(sum([xx^2 for xx in x]))
end

function Hq(q)
    """Return ∂H/∂q"""
    hq = zeros((6,3))
    for k in 1:6
        kth = zeros(3)
        for i in vcat(1:k-1,k+1:6)
            #println("k=$k, i=$i")
            kth += (q[i,:]-q[k,:]) .* (m[i] / norm(q[i,:]-q[k,:])^3)
        end
        kth *= G * m[k]
        hq[k,:] = kth
    end
    return -hq
end

