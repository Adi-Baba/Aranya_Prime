module main

@[export: 'prime_v_mul']
fn prime_v_mul(n i64, res &f64, x &f64, y f64) {
    unsafe {
        for i in 0 .. n {
            res[i] = x[i] * y
        }
    }
}
