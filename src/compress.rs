use crate::imagewrapper::{GreyImageWrapper, RgbImageWrapper};
use faer_core::{Mat, MatRef, Parallelism, dyn_stack::PodStack};
use faer_svd::*;
use rayon::prelude::*;

fn svdapprox(mat: MatRef<f32>, rank: usize, bad: bool) -> Mat<f32> {
    let m = mat.nrows();
    let n = mat.ncols();

    let mut s = Mat::zeros(m, n);
    let s = s.as_mut();
    let mut u = Mat::zeros(m, m);
    let u = u.as_mut();
    let mut v = Mat::zeros(n, n);
    let v = v.as_mut();

    let parallelism = Parallelism::None;
    let params = SvdParams::default();

    let stack_req = compute_svd_req::<f32>(
        m,
        n,
        ComputeVectors::Full,
        ComputeVectors::Full,
        parallelism,
        params,
    )
    .unwrap();

    let mut buffer = vec![0u8; stack_req.size_bytes()];
    let stack = PodStack::new(&mut buffer);

    compute_svd(mat, s, Some(u), Some(v), parallelism, stack, params);
    // TODO: Implement. Need to figure out if `compute_svd` automatically sorts the singular values.
    Mat::zeros(mat.nrows(), mat.ncols()) // Just a placeholder
}

pub trait Compressible {
    fn compress(&self, rank: usize) -> Self;
    fn compress_bad(&self, rank: usize) -> Self;
}

impl Compressible for GreyImageWrapper {
    fn compress(&self, rank: usize) -> Self {
        GreyImageWrapper {
            mat: svdapprox(self.mat.as_ref(), rank, false),
            width: self.width,
            height: self.height,
        }
    }

    fn compress_bad(&self, rank: usize) -> Self {
        GreyImageWrapper {
            mat: svdapprox(self.mat.as_ref(), rank, true),
            width: self.width,
            height: self.height,
        }
    }
}

impl Compressible for RgbImageWrapper {
    fn compress(&self, rank: usize) -> Self {
        let compressed_mats: [Mat<f32>; 3] = self
            .mats
            .par_iter()
            .map(|mat| svdapprox(mat.as_ref(), rank, false))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        RgbImageWrapper {
            mats: compressed_mats,
            width: self.width,
            height: self.height,
        }
    }

    fn compress_bad(&self, rank: usize) -> Self {
        let compressed_mats: [Mat<f32>; 3] = self
            .mats
            .par_iter()
            .map(|mat| svdapprox(mat.as_ref(), rank, true))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        RgbImageWrapper {
            mats: compressed_mats,
            width: self.width,
            height: self.height,
        }
    }
}

#[cfg(test)]
fn test_svd() {
    // TODO: Just want to figure out whether `compute_svd` sorts the singular values or not.
}
