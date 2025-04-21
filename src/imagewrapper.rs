use faer_core::Mat;
use image::*;
use std::array;
use std::io::{BufReader, Read, Seek, Write};

pub trait ImageWrapper {
    fn load<R: Read + Seek>(reader: R) -> ImageResult<Self>
    where
        Self: Sized;

    fn save<W: Write + Seek>(&self, writer: W, format: ImageFormat) -> ImageResult<()>;
}

pub struct GreyImageWrapper {
    pub mat: Mat<f32>,
    pub width: usize,
    pub height: usize,
}

impl ImageWrapper for GreyImageWrapper {
    fn load<R: Read + Seek>(reader: R) -> ImageResult<Self> {
        let mut reader = BufReader::new(reader);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        let format = guess_format(&buf)?;

        let dyn_img = load_from_memory_with_format(&buf, format)?.into_luma8();
        let (width, height) = dyn_img.dimensions();
        let width = width as usize;
        let height = height as usize;

        let mat = Mat::from_fn(height, width, |i, j| {
            let pixel = dyn_img.get_pixel(j as u32, i as u32);
            pixel[0] as f32
        });

        Ok(Self { mat, width, height })
    }

    fn save<W: Write + Seek>(&self, mut writer: W, format: ImageFormat) -> ImageResult<()> {
        let img = GrayImage::from_fn(self.width as u32, self.height as u32, |i, j| {
            let pixel_value = self.mat.get(j as usize, i as usize).clamp(0.0, 255.0) as u8;
            Luma([pixel_value])
        });

        img.write_to(&mut writer, format)?;
        Ok(())
    }
}

pub struct RgbImageWrapper {
    pub mats: [Mat<f32>; 3],
    pub width: usize,
    pub height: usize,
}

impl ImageWrapper for RgbImageWrapper {
    fn load<R: Read + Seek>(reader: R) -> ImageResult<Self> {
        let mut reader = BufReader::new(reader);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        let format = guess_format(&buf)?;

        let dyn_img = load_from_memory_with_format(&buf, format)?.into_rgb8();
        let (width, height) = dyn_img.dimensions();
        let width = width as usize;
        let height = height as usize;

        let mats = array::from_fn(|k| {
            Mat::from_fn(height, width, |j, i| {
                let pixel = dyn_img.get_pixel(i as u32, j as u32);
                pixel[k] as f32
            })
        });

        Ok(Self {
            mats,
            width,
            height,
        })
    }

    fn save<W: Write + Seek>(&self, mut writer: W, format: ImageFormat) -> ImageResult<()> {
        let image = RgbImage::from_fn(self.width as u32, self.height as u32, |i, j| {
            let pixel_values = array::from_fn(|k| {
                self.mats[k].get(j as usize, i as usize).clamp(0.0, 255.0) as u8
            });
            Rgb(pixel_values)
        });

        image.write_to(&mut writer, format)?;
        Ok(())
    }
}
