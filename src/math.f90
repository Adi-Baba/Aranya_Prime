module prime_math
    use iso_c_binding
    implicit none
contains
    subroutine prime_math_sum(n, res, x, y) bind(c, name="prime_math_sum")
        integer(c_long_long), value :: n
        real(c_double), dimension(n) :: res, x, y
        !GCC$ ATTRIBUTES DLLEXPORT :: prime_math_sum
        integer :: i
        
        !$omp parallel do
        do i = 1, n
            res(i) = x(i) + y(i)
        end do
    end subroutine prime_math_sum
end module prime_math
