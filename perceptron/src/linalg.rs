extern crate num;

use std::ops::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Vector<T>
where
    T: Clone,
{
    v: Vec<T>,
}

impl<T> Vector<T>
where
    T: Clone,
{
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.v.iter()
    }

    pub fn new() -> Self {
        Vector { v: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn from_fn<U, F>(size: U, f: F) -> Self
    where
        U: Into<usize>,
        F: FnMut(usize) -> T,
    {
        let s: usize = size.into();
        (0..s).map(f).collect::<Vec<T>>().into()
    }
}

pub trait DotProduct<Rhs> {
    type Output;

    fn dot(&self, other: Rhs) -> Self::Output;
}

impl<N> DotProduct<&Self> for &Vector<N>
where
    N: num::Float,
{
    type Output = N;

    fn dot(&self, other: &Self) -> Self::Output {
        self.iter()
            .zip(other.iter())
            .fold(N::zero(), |i, r| i + ((*r.0) * (*r.1)))
    }
}

impl<N> Add for &Vector<N>
where
    N: num::Float + Add<Output = N>,
{
    type Output = Vector<N>;

    fn add(self, other: Self) -> Self::Output {
        self.iter()
            .zip(other.iter())
            .map(|(l, r)| *l + *r)
            .collect::<Vec<N>>()
            .into()
    }
}

impl<N> AddAssign for Vector<N>
where
    N: num::Float + Add<Output = N>,
{
    fn add_assign(&mut self, other: Self) {
        *self = &*self + &other;
    }
}

impl<N> Sub for &Vector<N>
where
    N: num::Float + Sub<Output = N>,
{
    type Output = Vector<N>;

    fn sub(self, other: Self) -> Self::Output {
        self.iter()
            .zip(other.iter())
            .map(|(l, r)| *l - *r)
            .collect::<Vec<N>>()
            .into()
    }
}

impl<N> SubAssign for Vector<N>
where
    N: num::Float + Sub<Output = N>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = &*self - &other;
    }
}

/// Vector scalar multiplication
impl<N> Mul<N> for &Vector<N>
where
    N: num::Float,
{
    type Output = Vector<N>;

    fn mul(self, other: N) -> Self::Output {
        self.iter()
            .map(|v_i| *v_i * other)
            .collect::<Vec<N>>()
            .into()
    }
}

/// Vector scalar division
impl<N> Div<N> for &Vector<N>
where
    N: num::Float,
{
    type Output = Vector<N>;

    fn div(self, other: N) -> Self::Output {
        self.iter()
            .map(|v_i| *v_i / other)
            .collect::<Vec<N>>()
            .into()
    }
}

impl<T> std::convert::From<Vec<T>> for Vector<T>
where
    T: Clone,
{
    fn from(v: Vec<T>) -> Self {
        Vector { v: v }
    }
}

impl<T> Index<usize> for Vector<T>
where
    T: Clone,
{
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        &self.v[i]
    }
}

impl<U> Vector<U>
where
    U: num::Float,
{
    /// A vector full of ones
    pub fn ones(size: usize) -> Self {
        Vector::from_fn(size, |_| U::one())
    }

    /// Additive identity - Vector full of zeros
    pub fn add_ident(size: usize) -> Self {
        Vector::from_fn(size, |_| U::zero())
    }

    /// Alias for add_ident
    pub fn zeros(size: usize) -> Self {
        Vector::add_ident(size)
    }

    /// Euclidian norm of the vector
    pub fn norm(&self) -> U {
        self.dot(&self).sqrt()
    }

    /// Normalize a vector so that it's euclidian norm is 1
    /// 
    /// # Example
    /// 
    /// ```
    /// let mut a = Vector::from(vec![1., 2., 3.]);
    /// assert_eq!(a.normalize.abs(), 1.);
    /// ```
    /// 
    pub fn normalize(&self) -> Self {
        let abs = self.norm();
        self.iter().map(|x| *x / abs).collect::<Vec<U>>().into()
    }

    /// Elementwise `abs`
    pub fn abs(&self) -> Self {
        self.iter().map(|x| x.abs()).collect::<Vec<U>>().into()
    }

    /// Check if two vectors are approximately equal
    /// # Arguments
    ///
    /// * `treshold` - Optional treshold for what is considered equal. Defaults to 1.0e-9
    /// 
    pub fn approx_eq(&self, other: Self, treshold: Option<U>) -> bool {
        /*
        let error = (self - &other).norm().abs();
        if let Some(treshold) = treshold {
            error < treshold
        } else {
            error < U::one() * U::from(10e-9).expect("Can't compare with default treshold")
        }
        */
        let error = self - &other;

        let treshold = match treshold {
            Some(treshold) => treshold,
            None => U::one() * U::from(10e-9).expect("Can't compare with default treshold"),
        };
        for b in error.iter().map(|x| x.abs() < treshold) {
            if !b {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let v1 = Vector::from(vec![1., 2.]);
        let v2 = Vector::from(vec![3., 5.]);
        assert_eq!(&v1 + &v2, Vector::from(vec![4., 7.]));
    }

    #[test]
    fn test_add_assign() {
        let mut v1 = Vector::from(vec![1., 2.]);
        let v2 = Vector::from(vec![3., 5.]);
        v1 += v2;
        assert_eq!(v1, Vector::from(vec![4., 7.]));
    }

    #[test]
    fn test_sub() {
        let v1 = Vector::from(vec![1., 2.]);
        let v2 = Vector::from(vec![3., 5.]);
        assert_eq!(&v1 - &v2, Vector::from(vec![-2., -3.]));
    }

    #[test]
    fn test_sub_assign() {
        let mut v1 = Vector::from(vec![1., 2.]);
        let v2 = Vector::from(vec![3., 5.]);
        v1 -= v2;
        assert_eq!(v1, Vector::from(vec![-2., -3.]));
    }

    #[test]
    fn test_norm() {
        let a = Vector::from(vec![3., 4.]);
        assert_eq!(a.norm(), 5.);
    }

    #[test]
    fn test_normalize() {
        let a = Vector::from(vec![1., 1., 1.]);
        let b = (1. / 3. as f64).sqrt();
        let error = &a.normalize() - &Vector::from(vec![b, b, b]);
        let l = error.len();
        assert!(error.approx_eq(Vector::zeros(l), None));
        assert!(error.norm().abs() < 1.0e-9);
    }
}
