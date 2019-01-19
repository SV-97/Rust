use std::f64;

extern crate num;
extern crate time;
use time::PreciseTime;
use num::Complex;

const PI: f64 = f64::consts::PI;
const E: f64 = f64::consts::E;


fn fft(f: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = f.len();
    if n == 1 {
        f.to_vec()
    } else {
        let half_n = n/2;
        let g = fft(&(f[0..].iter().step_by(2).cloned().collect::<Vec<Complex<f64>>>()));
        let u = fft(&(f[1..].iter().step_by(2).cloned().collect::<Vec<Complex<f64>>>()));
        let mut c_k_1 = vec![Complex {re: 0., im: 0.}; half_n];
        let mut c_k_2 = vec![Complex {re: 0., im: 0.}; half_n];
        for k in 0..half_n {
            let k_f = k as f64;
            let n_f = n as f64;
            c_k_1[k] = g[k] + u[k] * (-2. * PI * Complex::i() * k_f / n_f).exp();
            c_k_2[k] = g[k] + u[k] * (-2. *  PI * Complex::i() * k_f / n_f).exp();
        };
        c_k_1.append(&mut c_k_2);
        c_k_1
    }
}


fn function(x: f64) -> f64{
    (2. * PI * x).sin()
}


fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64>{
    let step_size = (stop - start) / n as f64;
    let mut space = vec![];
    for i in 0..n {
        space.push(start + step_size * i as f64);
    }
    space
}


fn main() {
    let t = linspace(0., 5., (2. as f64).powf(15.) as usize);
    let mut wave = vec![];
    for &x in &t {
        wave.push(Complex{re: function(x), im: 0.});
    }
    
    let t1 = PreciseTime::now();
    let wave_fft = fft(&wave);
    let t2 = PreciseTime::now();
    dbg!(&wave_fft);
    println!("Fourier transform took {} seconds", t1.to(t2));
}
