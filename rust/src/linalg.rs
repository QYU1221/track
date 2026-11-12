use nalgebra::SMatrix;
pub(crate) fn mat<const R: usize, const C: usize>(a: [[f64; C]; R]) -> SMatrix<f64, R, C> {
    SMatrix::from_fn(|i, j| a[i][j])
}

pub(crate) fn array<const R: usize, const C: usize>(a: SMatrix<f64, R, C>) -> [[f64; C]; R] {
    std::array::from_fn(|i| std::array::from_fn(|j| a[(i, j)]))
}
