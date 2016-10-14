# This file is a part of Julia. License is MIT: http://julialang.org/license
import Base: size, copy, \, inv, cholfact, cholfact!
import Base.LinAlg: BlasInt, BlasFloat, checksquare
export SymmetricRFP, TriangularRFP, CholeskyDenseRFP

# Rectangular Full Packed Matrices
type SymmetricRFP{T<:BlasFloat} <: AbstractMatrix{T}
    data::Vector{T}
    transr::Char
    uplo::Char
    N::BlasInt
end

function Ac_mul_A_RFP{T<:BlasFloat}(A::Matrix{T})
    n = size(A, 2)
    C = LAPACK2.sfrk!('N', 'U', 'T', 1.0, A, 0.0, Array(T, div(n*(n+1),2)))
    SymmetricRFP(C, 'N', 'U')
end

function SymmetricRFP(x::Vector, transr = 'N', uplo = 'U')
    N = try
        Int(sqrt(1/4+2*length(x))-1/2)
    catch
        throw(DimensionMismatch("x has length $(length) which is not N*(N+1)/2 for any integer N"))
    end
    SymmetricRFP(x, transr, uplo, N)
end

size(A::SymmetricRFP) = (A.N, A.N)

#TODO see if this can be done better, and account for transpose
function Base.getindex(A::SymmetricRFP, i::Int, j::Int)
    n = A.N
    if A.uplo == 'L'
        (i,j) = ifelse(i >= j, (i,j), (j,i))
        A.data[(n*(n+1)-(n-j+1)*(n-j+2)) >> 1 + i - j + 1]
    else
        (i,j) = ifelse(j >= i, (i,j), (j,i))
        A.data[j*(j-1) >> 1 + i]
    end
end

# function Base.getindex(A::SymmetricRFP, i)
#     if A.uplo == 'L'
#         A.data[i+j*(j-1)]
#     else
#
#     end
# end

type TriangularRFP{T<:BlasFloat} <: AbstractMatrix{T}
    data::Vector{T}
    transr::Char
    uplo::Char
end
TriangularRFP(A::Matrix) = TriangularRFP(trttf!('N', 'U', A), 'N', 'U')

full(A::TriangularRFP) = (A.uplo=='U' ? triu! : tril!)(LAPACK2.tfttr!(A.transr, A.uplo, A.data))

type CholeskyDenseRFP{T<:BlasFloat} <: Factorization{T}
    data::Vector{T}
    transr::Char
    uplo::Char
end

cholfact!{T<:BlasFloat}(A::SymmetricRFP{T}) = CholeskyDenseRFP(LAPACK2.pftrf!(A.transr, A.uplo, copy(A.data)), A.transr, A.uplo)
cholfact{T<:BlasFloat}(A::SymmetricRFP{T}) = cholfact!(copy(A))

copy(A::SymmetricRFP) = SymmetricRFP(copy(A.data), A.transr, A.uplo)

# Least squares
\(A::CholeskyDenseRFP, B::VecOrMat) = LAPACK2.pftrs!(A.transr, A.uplo, A.data, copy(B))

inv(A::CholeskyDenseRFP)=LAPACK2.pftri!(A.transr, A.uplo, copy(A.data))
