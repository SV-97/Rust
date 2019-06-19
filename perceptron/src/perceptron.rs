extern crate num;
extern crate rand;

use rand::Rng;

use super::activation;
use super::linalg;
use linalg::*;

/// Perceptron model
/// modeled after the sklearn Perceptron
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct Perceptron<R, N, U>
where
    R: rand::SeedableRng + Rng,
    N: num::Float,
    U: num::Unsigned,
{
    /// Number of iterations
    pub n_iterations: U,
    /// Vector holding all the errors of a learning pass
    pub errors: Vec<N>,
    /// Generator that's used to initialize the weights
    pub random_gen: R,
    /// Weights
    pub w: Vector<N>,
    /// Training inputs
    pub X: Vec<Vector<N>>,
    /// Expected training outputs
    pub y: Vector<N>,
}
#[allow(non_snake_case)]
impl<R, N, U> Perceptron<R, N, U>
where
    R: rand::SeedableRng + Rng,
    N: num::Float + std::fmt::Debug,
    U: num::Unsigned + num::ToPrimitive + Copy + PartialOrd,
    rand::distributions::Standard: rand::distributions::Distribution<N>,
{
    pub fn new(n_iterations: U, random_gen: R) -> Self {
        Perceptron {
            n_iterations,
            errors: Vec::new(),
            random_gen,
            w: Vector::new(),
            X: Vec::new(),
            y: Vector::new(),
        }
    }

    /// Make the Perceptron learn
    /// # Arguments
    ///
    /// * `X` - Matrix with training data
    /// * `y` - Vector of expected outputs
    pub fn fit(&mut self, X: Vec<Vector<N>>, y: Vector<N>) {
        if X.len() != y.len() {
            panic!("Input data and corresponding output data don't match up - they have different dimensions")
        }
        self.w = Vector::from_fn(X[0].len(), |_| self.random_gen.gen::<N>()); // initialize weights with random values
        self.X = X;
        self.y = y;
        for _ in num::range(U::zero(), self.n_iterations) {
            let index = self.random_gen.gen_range::<usize, _, _>(0, self.y.len());
            let x_ = &self.X[index];
            let y_ = self.y[index];
            let y_hat = activation::heaviside((&self.w).dot(&x_));
            let error = y_ - y_hat;
            self.errors.push(error);
            self.w += x_ * error;
        }
    }

    /// Predict the output of a given input
    /// Model has to be trained using fit beforehand
    /// # Arguments
    /// 
    /// * `x` - Input vector
    pub fn predict(&self, x: &Vector<N>) -> N {
        activation::heaviside((&self.w).dot(x))
    }
}

/// Function that's -1 for x<0, 1 for x>0 and 0 otherwise
pub fn ternarize<N>(x: &N) -> N
where N: num::Float
{
    if x < &N::zero() {
        - N::one()
    } else if x > &N::zero() {
        N::one()
    } else {
        N::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron() {
        let seed = [1; 32];
        let random_gen: rand::rngs::StdRng = rand::SeedableRng::from_seed(seed);
        let mut p = Perceptron::<_, f32, _>::new(500 as usize, random_gen);
    }
    
    #[test]
    fn test_fit() {
        let seed = [1; 32];
        let random_gen: rand::rngs::StdRng = rand::SeedableRng::from_seed(seed);
        let mut p = Perceptron::<_, f32, _>::new(500 as usize, random_gen);
        p.fit(
            vec![
                vec![1., 0., 0.].into(),
                vec![1., 0., 1.].into(),
                vec![1., 1., 0.].into(),
                vec![1., 1., 1.].into(),
            ],
            vec![0., 1., 1., 1.].into(),
        );
        let w_out = Vector::from(p.w.iter().map(|x| ternarize(x)).collect::<Vec<_>>());
        let w_expected = Vector::from(vec![-1., 1., 1.]);
        assert_eq!(w_out, w_expected);
    }

    #[test]
    fn test_predict() {
        let seed = [1; 32];
        let random_gen: rand::rngs::StdRng = rand::SeedableRng::from_seed(seed);
        let mut p = Perceptron::<_, f32, _>::new(500 as usize, random_gen);
        p.fit(
            vec![
                vec![1., 0., 0.].into(),
                vec![1., 0., 1.].into(),
                vec![1., 1., 0.].into(),
                vec![1., 1., 1.].into(),
            ],
            vec![0., 1., 1., 1.].into(),
        );

        let output = p.predict(&Vector::from(vec![1., 0., 0.1]));
        assert_eq!(ternarize(&output), 0.)
    }
}