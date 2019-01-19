use std::f64;

extern crate time;
use time::PreciseTime;


fn main() {
    let t1 = PreciseTime::now();
    let integral_of_function = composite_simpsons(&function, 0., 2.*f64::consts::PI, 100000);
    let t2 = PreciseTime::now();
    println!("Integration took {} seconds", t1.to(t2));
    println!("F approx: {}", integral_of_function);
}


fn function(a: f64) -> f64 {
    a.sin()
}


fn composite_simpsons(f: &Fn(f64) -> f64, a: f64, b:f64, n: u64) -> f64 {
    let step_size = (b - a) / n as f64;
    let mut x_k;
    let mut x_k1;
    let mut integral = 0.0;
    for i in 0..n {
        let k = &(i as f64);
        x_k = a + k * step_size;
        x_k1 = a + (k + 1.) * step_size;

        let simpson = step_size / 6. * (f(x_k) + 4. * f((x_k + x_k1) / 2. ) + f(x_k1));
        integral += simpson;
    }
    integral
}