extern crate cpython;

use cpython::{PyResult, Python};
use image::GenericImageView;
use std::{fs, path};
use std::str::FromStr;

static SUPPORTED_EXTENSIONS: [&str; 2] = [".jpg", ".jpeg"];


fn get_pixels_vec(file_path: path::PathBuf, expected_dimensions: (usize, usize)) -> Result<Vec<f64>, String> {
    let img = match image::open(&file_path) {
        Ok(v) => v,
        Err(_) => return Err(format!("Can't open {:?}.", file_path)),
    };

    let mut pixel_vec: Vec<f64> = Vec::with_capacity(expected_dimensions.0 * expected_dimensions.1 * 3);
    for p in img.pixels() {
        pixel_vec.push(p.2.data[0] as f64);
        pixel_vec.push(p.2.data[1] as f64);
        pixel_vec.push(p.2.data[2] as f64);
    }
    Ok(pixel_vec)
}


fn has_right_dimensions(dimensions: (u32, u32), expected_dimensions: (usize, usize)) -> bool {
    dimensions.0 as usize == expected_dimensions.0 && dimensions.1 as usize == expected_dimensions.1
}


fn is_supported_format(file_path: &fs::DirEntry) -> bool {
    let file_name: String = file_path.file_name().into_string().unwrap().to_lowercase();
    for e in SUPPORTED_EXTENSIONS.iter().map(|x| x.to_string()) {
        if file_name.ends_with(&e) {
            return true;
        }
    }
    return false;
}


fn get_supported_images(images_path: &path::PathBuf, expected_dimensions: (usize, usize)) -> Result<Vec<fs::DirEntry>, String> {
    let mut images: Vec<fs::DirEntry> = vec![];
    let mut not_readable: Vec<path::PathBuf> = vec![];

    for f in fs::read_dir(images_path).expect(&format!("Can't open directory {:?}", images_path)).map(|x| x.unwrap()) {
        if f.file_type().unwrap().is_file() && is_supported_format(&f) {
            match image::open(f.path()) {
                Ok(img) => {
                    if has_right_dimensions(img.dimensions(), expected_dimensions) {
                        images.push(f)
                    }
                }
                Err(_) => not_readable.push(f.path()),
            };
        }
    }

    if not_readable.len() > 0 {
        Err(format!("Can't read {} files : {:?}", not_readable.len(), not_readable))
    } else {
        Ok(images)
    }
}


#[allow(dead_code)]
pub fn load_image(path: &str, expected_dimensions: (usize, usize)) -> Vec<f64> {
    match image::open(path) {
        Ok(img) => {
            if has_right_dimensions(img.dimensions(), expected_dimensions) {
                get_pixels_vec(path::PathBuf::from_str(path).unwrap(), expected_dimensions).unwrap()
            } else {
                panic!("The image provided \"{}\" has no the right dimensions {:?}", path, expected_dimensions)
            }
        }
        Err(e) => panic!("{}", e),
    }
}


pub fn load_from_images_path(_: Python, path: &str, images_dimensions: (usize, usize), y_train_value: f64) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let images_path: path::PathBuf = path::PathBuf::from_str(path).unwrap(); // set paths in functions of flags and fill y_train
    let supported_images: Vec<fs::DirEntry> = match get_supported_images(&images_path, images_dimensions) {
        Ok(v) => v,
        Err(e) => panic!("{}", e),
    };
    let mut x_train: Vec<Vec<f64>> = vec![];

    for file in supported_images {
        match get_pixels_vec(file.path(), images_dimensions) {
            Ok(v) => x_train.push(v),
            Err(e) => println!("{}", e),
        };
    }

    let y_train: Vec<f64> = vec![y_train_value; x_train.len()];
    Ok((x_train, y_train))
}
