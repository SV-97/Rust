extern crate num;

pub fn heaviside<N>(x: N) -> N
where
    N: num::Num + PartialOrd,
{
    if x < N::zero() {
        N::zero()
    } else {
        N::one()
    }
}

#[test]
fn test_heaviside() {
    assert_eq!(heaviside(5), 1);
    assert_eq!(heaviside(-5), 0);
    assert_eq!(heaviside(0.000001), 1.);
    assert_eq!(heaviside(-0.0000001), 0.);
    assert_eq!(heaviside(0), 1);
}
