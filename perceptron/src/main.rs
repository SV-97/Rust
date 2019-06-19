extern crate num;
extern crate rand;

pub mod activation;
pub mod linalg;
pub mod perceptron;

fn main() {
    let seed = [1; 32];
    let random_gen: rand::rngs::StdRng = rand::SeedableRng::from_seed(seed);
    let mut p = perceptron::Perceptron::<_, f32, _>::new(500 as usize, random_gen);
    p.fit(
        vec![
            vec![1., 0., 0.].into(),
            vec![1., 0., 1.].into(),
            vec![1., 1., 0.].into(),
            vec![1., 1., 1.].into(),
        ],
        vec![0., 1., 1., 1.].into(),
    );
    dbg!(p.w);
}
