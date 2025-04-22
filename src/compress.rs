use crate::imagewrapper::{GreyImageWrapper, RgbImageWrapper};
use faer_core::{Mat, MatRef, Parallelism, dyn_stack::PodStack};
use faer_svd::*;
use rayon::prelude::*;

#[derive(Debug)]
pub enum SvdApproxError {
    InvalidRank(usize, usize),
    ComputeReqFailed,
}

impl std::fmt::Display for SvdApproxError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SvdApproxError::InvalidRank(k, rank) => {
                write!(f, "`rank` must be between 0 and {}, got {}.", k, rank)
            }
            SvdApproxError::ComputeReqFailed => {
                write!(f, "Failed to compute buffer requirements for SVD.")
            }
        }
    }
}

fn svdapprox(mat: MatRef<f32>, rank: usize, bad: bool) -> Result<Mat<f32>, SvdApproxError> {
    let m = mat.nrows();
    let n = mat.ncols();
    let k = m.min(n);

    if rank <= 0 || rank > k {
        return Err(SvdApproxError::InvalidRank(k, rank));
    }

    if rank == k {
        return Ok(mat.to_owned());
    }

    let mut s = Mat::zeros(k, 1);
    let s_mut = s.as_mut();
    let mut u = Mat::zeros(m, m);
    let u_mut = u.as_mut();
    let mut v = Mat::zeros(n, n);
    let v_mut = v.as_mut();

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
    .map_err(|_| SvdApproxError::ComputeReqFailed)?;

    // Multiply by 1.5 to allocate a bit more space for the PodStack
    let required_size = (1.5 * stack_req.size_bytes() as f32) as usize;
    let mut buffer = vec![0u8; required_size];
    let stack = PodStack::new(&mut buffer);

    // `compute_svd` automatically sorts the singular values in descending order
    compute_svd(
        mat,
        s_mut,
        Some(u_mut),
        Some(v_mut),
        parallelism,
        stack,
        params,
    );

    // If `bad` is false, apply the Eckart-Young-Mirsky theorem to get the best low-rank
    // approximation, using the `rank` largest singular values and corresponding singular vectors.
    // Otherwise, use the smallest singular pairs to get the worst low-rank approximation.
    let (s_new, u_new, v_new) = if bad {
        (
            Mat::from_fn(
                rank,
                rank,
                |i, j| {
                    if i == j { s[(k - rank + i, 0)] } else { 0.0 }
                },
            ),
            u.as_ref().submatrix(0, m - rank, m, rank),
            v.as_ref().submatrix(0, n - rank, n, rank),
        )
    } else {
        (
            Mat::from_fn(rank, rank, |i, j| if i == j { s[(i, 0)] } else { 0.0 }),
            u.as_ref().submatrix(0, 0, m, rank),
            v.as_ref().submatrix(0, 0, n, rank),
        )
    };

    Ok(u_new * s_new * v_new.transpose())
}

pub trait Compressible {
    type Error;
    fn compress(&self, rank: usize) -> Result<Self, Self::Error>
    where
        Self: Sized;
    fn compress_bad(&self, rank: usize) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

impl Compressible for GreyImageWrapper {
    type Error = SvdApproxError;

    fn compress(&self, rank: usize) -> Result<Self, Self::Error> {
        let mat = svdapprox(self.mat.as_ref(), rank, false)?;
        Ok(GreyImageWrapper {
            mat,
            width: self.width,
            height: self.height,
        })
    }

    fn compress_bad(&self, rank: usize) -> Result<Self, Self::Error> {
        let mat = svdapprox(self.mat.as_ref(), rank, true)?;
        Ok(GreyImageWrapper {
            mat,
            width: self.width,
            height: self.height,
        })
    }
}

impl Compressible for RgbImageWrapper {
    type Error = SvdApproxError;

    fn compress(&self, rank: usize) -> Result<Self, Self::Error> {
        let compressed_mats: [Mat<f32>; 3] = self
            .mats
            .par_iter()
            .map(|mat| svdapprox(mat.as_ref(), rank, false))
            .collect::<Result<Vec<_>, SvdApproxError>>()?
            .try_into()
            .unwrap();

        Ok(RgbImageWrapper {
            mats: compressed_mats,
            width: self.width,
            height: self.height,
        })
    }

    fn compress_bad(&self, rank: usize) -> Result<Self, Self::Error> {
        let compressed_mats: [Mat<f32>; 3] = self
            .mats
            .par_iter()
            .map(|mat| svdapprox(mat.as_ref(), rank, true))
            .collect::<Result<Vec<_>, SvdApproxError>>()?
            .try_into()
            .unwrap();

        Ok(RgbImageWrapper {
            mats: compressed_mats,
            width: self.width,
            height: self.height,
        })
    }
}
