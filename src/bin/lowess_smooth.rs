use auto_pid_paper::{lowess::Lowess, param_manager::ParamManager};

use clap::Parser;
use indicatif::ProgressIterator;
use itertools_num::linspace;
use polars::prelude::*;
use rayon::prelude::*;
use std::{
    fs::{self, File},
    path::Path,
};

use anyhow::Result;
use nalgebra::DVector;

fn curve_from_df(df: &DataFrame, idx: usize) -> Result<(DVector<f64>, DVector<f64>)> {
    let col_name = format!("curve_{}_contains_pnt", idx);
    let df = df.clone().lazy().filter(col(&col_name)).collect()?;

    let xs = DVector::from_vec(
        df["x"]
            .f64()?
            .into_iter()
            .map(|val| val.unwrap())
            .collect::<Vec<f64>>(),
    );
    let ys = DVector::from_vec(
        df["y"]
            .f64()?
            .into_iter()
            .map(|val| val.unwrap())
            .collect::<Vec<f64>>(),
    );

    Ok((xs, ys))
}

/// extend curvelus and linearize data
#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    /// Data dir
    #[arg(short, long, default_value_t=String::from("data/faust_det_60"))]
    directory: String,

    /// Input file
    #[arg(short, long, default_value_t=String::from("clustered_points.parquet"))]
    input_file: String,
}
fn main() -> Result<()> {
    println!("\nExecuting {}\n", file!());

    let args = Args::parse();

    let params = ParamManager::new(&args.directory)?;

    let timer = std::time::Instant::now();

    let file = std::path::Path::new(&args.directory).join("raw.parquet");
    let mut file = File::open(file)?;
    let raw_df = ParquetReader::new(&mut file).finish()?;
    let x_min: f64 = raw_df[params.x_col()].min()?.unwrap();
    let x_max: f64 = raw_df[params.x_col()].max()?.unwrap();

    // reset output directory
    let output_dir = Path::new(&args.directory).join("smoothed_lines/");
    if fs::exists(Path::new(&output_dir))? {
        fs::remove_dir_all(&output_dir)?;
    }
    let file = std::path::Path::new(&args.directory).join(&args.input_file);
    let mut file = File::open(file)?;
    let df = ParquetReader::new(&mut file).finish()?;
    fs::create_dir_all(&output_dir)?;

    let mut max_idx = 0;
    for idx in 0.. {
        let col_name = format!("curve_{}_contains_pnt", idx);

        if !df.get_column_names().contains(&col_name.as_str()) {
            break;
        }
        max_idx = idx;
    }

    for idx in (0..max_idx + 1).progress() {
        let data = curve_from_df(&df, idx);
        match data {
            Ok((xs, ys)) => {
                let window_size = (((xs.len() as f64) * params.bandwidth()) as usize)
                    .max(params.polynomial_order() as usize);
                let loess = Lowess::new(&xs, &ys);

                let xs_model = &xs.iter().cloned().collect::<Vec<_>>();
                //[window_size / 4..xs.len() - window_size / 4];
                let xs = linspace(x_min, x_max, 500).collect::<Vec<_>>();
                let y_estimate = xs
                    .par_iter()
                    .map(|&x| {
                        loess.estimate(x, window_size, false, params.polynomial_order() as usize)
                    })
                    .collect::<Vec<_>>();

                let extrapolated_right = xs
                    .par_iter()
                    .map(|&x| x > *xs_model.last().unwrap())
                    .collect::<Vec<_>>();
                let extrapolated_left = xs
                    .par_iter()
                    .map(|&x| x < *xs_model.first().unwrap())
                    .collect::<Vec<_>>();

                let out_df = df!(
                    "x"=>xs,
                    "y"=>y_estimate,
                    "extrapolated_left"=>extrapolated_left,
                    "extrapolated_right"=>extrapolated_right,
                )?;

                let mut file = std::fs::File::create(
                    Path::new(&output_dir).join(format!("curve_{}.parquet", idx)),
                )?;

                let mut out_df = out_df
                    .unique(Some(&["x".into()]), UniqueKeepStrategy::Any, None)?
                    .sort(["x"], SortMultipleOptions::default())?
                    .drop_nulls::<String>(None)?;

                ParquetWriter::new(&mut file).finish(&mut out_df)?;
            }
            _ => break,
        }
    }

    println!("\ttime: {} s", timer.elapsed().as_secs_f32());
    Ok(())
}
