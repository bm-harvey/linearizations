use anyhow::Result;
use auto_pid_paper::lowess::Lowess;
use auto_pid_paper::param_manager::ParamManager;

use clap::Parser;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use polars::prelude::*;
use rayon::prelude::*;
use std::fs::{self, File};
use std::path::Path;

#[derive(Default, Debug, Clone)]
struct Curve {
    points: Vec<(f64, f64)>,
}

enum UseExtrapolated {
    Left,
    Right,
    None,
    Both,
}

impl Curve {
    /// expects a simple df with an "x" an "y" col
    fn from_parquet_file(file: &str, extrapolation_use: UseExtrapolated) -> Result<Self> {
        let mut file = File::open(file)?;
        let df = ParquetReader::new(&mut file).finish()?;
        let df = df.lazy();
        let df = match extrapolation_use {
            UseExtrapolated::None => df
                .filter(col("extrapolated_left").not())
                .filter(col("extrapolated_right").not()),
            UseExtrapolated::Left => df.filter(col("extrapolated_right").not()),
            UseExtrapolated::Right => df.filter(col("extrapolated_left").not()),
            UseExtrapolated::Both => df,
        }
        .collect()?;

        let mut points = Vec::<(f64, f64)>::with_capacity(df.height());

        for (x, y) in df["x"].f64()?.into_iter().zip(df["y"].f64()?.into_iter()) {
            let (x, y) = (x.unwrap(), y.unwrap());
            points.push((x, y));
        }
        Ok(Self { points })
    }

    fn points(&self) -> &Vec<(f64, f64)> {
        &self.points
    }

    fn x_at(&self, idx: usize) -> f64 {
        self.points[idx].0
    }

    fn add_point(&mut self, x: f64, y: f64) {
        self.points.push((x, y));
    }
    fn insert_point(&mut self, idx: usize, x: f64, y: f64) {
        self.points.insert(idx, (x, y));
    }

    fn x_lims(&self) -> (f64, f64) {
        (
            self.points.first().unwrap().0,
            self.points.last().unwrap().0,
        )
    }

    fn contains_x(&self, x: f64) -> bool {
        self.x_lims().0 <= x && x <= self.x_lims().1
    }

    fn y_lims(&self) -> (f64, f64) {
        (
            self.points.first().unwrap().1,
            self.points.last().unwrap().1,
        )
    }

    fn y_at(&self, idx: usize) -> f64 {
        self.points[idx].1
    }

    fn evaluate(&self, x: f64) -> f64 {
        let idx_l = self.points.iter().position(|pnt| x < pnt.0);

        if idx_l.is_none() {
            let pnt_1 = self.points[self.points.len() - 2];
            let pnt_2 = self.points[self.points.len() - 1];

            let slope = (pnt_2.1 - pnt_1.1) / (pnt_2.0 - pnt_1.0);

            return pnt_2.1 + slope * (x - pnt_2.0);
        }

        let idx_l = idx_l.unwrap(); // none case checked above;

        if idx_l == 0 {
            let pnt_1 = self.points[0];
            let pnt_2 = self.points[1];

            let slope = (pnt_2.1 - pnt_1.1) / (pnt_2.0 - pnt_1.0);

            return pnt_1.1 + slope * (x - pnt_1.0);
        }

        // at this point idx_l and idx_r should be valid indices.
        let idx_r = idx_l;
        let idx_l = idx_r - 1;

        let dx = self.x_at(idx_r) - self.x_at(idx_l);
        ((x - self.x_at(idx_l)) * self.y_at(idx_r) + (self.x_at(idx_r) - x) * self.y_at(idx_l)) / dx
    }

    fn extend_from_single_reference_curve(
        &mut self,
        ref_x: f64,
        new_x: f64,
        reference_curve: &Curve,
        left_insert: bool,
    ) -> &mut Self {
        // get xs of overlap
        let x_low = self.x_lims().0.max(reference_curve.x_lims().0);
        let x_high = self.x_lims().1.min(reference_curve.x_lims().1);

        let x_range = x_high - x_low;

        let dx = x_range / 200.;

        let mut x = x_low;
        let mut xs = vec![];
        let mut dys = vec![];
        while x < x_high {
            xs.push(x);
            dys.push(self.evaluate(x) - reference_curve.evaluate(x));
            x += dx;
        }

        let mut dys_frac = vec![];
        if false {
            for idx in 1..dys.len() {
                dys_frac.push(dys[idx - 1] / dys[idx]);
            }
        } else {
            for idx in 1..dys.len() {
                dys_frac.push(dys[idx] / dys[idx - 1]);
            }
        }

        let mut avg_frac = dys_frac.iter().sum::<f64>() / dys_frac.len() as f64;
        if avg_frac < 0.0 {
            avg_frac = -1. * avg_frac;
        }

        let dy_ref = self.evaluate(ref_x) - reference_curve.evaluate(ref_x);
        let dy_new = dy_ref * avg_frac.powf((new_x - ref_x) / dx);
        let new_y = reference_curve.evaluate(new_x) + dy_new;

        if new_y.is_nan() {
            panic!()
        }
        if left_insert {
            self.insert_point(0, new_x, new_y);
        } else {
            self.add_point(new_x, new_y);
        }

        self
    }

    fn extend_from_lowess(
        &mut self,
        new_x: f64,
        left_insert: bool,
        params: &ParamManager,
    ) -> &mut Self {
        let xs = nalgebra::DVector::from_vec(self.points.iter().map(|(x, _)| *x).collect_vec());
        let ys = nalgebra::DVector::from_vec(self.points.iter().map(|(_, y)| *y).collect_vec());
        let loess = Lowess::new(&xs, &ys);

        let window_size = (((self.points.len() as f64) * params.bandwidth()) as usize)
            .max(params.polynomial_order() as usize);

        let new_y = loess.estimate(
            new_x,
            window_size,
            false,
            params.polynomial_order() as usize,
        );

        if new_y.is_nan() {
            panic!()
        }

        if left_insert {
            self.insert_point(0, new_x, new_y);
        } else {
            self.add_point(new_x, new_y);
        }

        self
    }

    fn extend_from_double_reference_curve(
        &mut self,
        ref_x: f64,
        new_x: f64,
        reference_curve_1: &Curve,
        reference_curve_2: &Curve,
        left_insert: bool,
    ) -> &mut Self {
        let x_low = self.x_lims().0.max(reference_curve_1.x_lims().0);
        let x_high = self.x_lims().1.min(reference_curve_1.x_lims().1);

        let x_range = x_high - x_low;

        let dx = x_range / 20.;

        let mut x = x_low;
        let mut xs = vec![];
        let mut dys = vec![];
        while x < x_high {
            xs.push(x);
            dys.push(self.evaluate(x) - reference_curve_1.evaluate(x));
            x += dx;
        }

        let mut dys_frac = vec![];
        if false {
            for idx in 1..dys.len() {
                dys_frac.push(dys[idx - 1] / dys[idx]);
            }
        } else {
            for idx in 1..dys.len() {
                dys_frac.push(dys[idx] / dys[idx - 1]);
            }
        }

        let mut avg_frac = dys_frac.iter().sum::<f64>() / dys_frac.len() as f64;
        if avg_frac < 0.0 {
            avg_frac = -1. * avg_frac;
        }

        let dy_ref = self.evaluate(ref_x) - reference_curve_1.evaluate(ref_x);
        let dy_new = dy_ref * avg_frac.powf((new_x - ref_x) / dx);
        let new_y_1 = reference_curve_1.evaluate(new_x) + dy_new;

        let x_low = self.x_lims().0.max(reference_curve_2.x_lims().0);
        let x_high = self.x_lims().1.min(reference_curve_2.x_lims().1);

        let x_range = x_high - x_low;

        let dx = x_range / 20.;

        let mut x = x_low;
        let mut xs = vec![];
        let mut dys = vec![];
        while x < x_high {
            xs.push(x);
            dys.push(self.evaluate(x) - reference_curve_2.evaluate(x));
            x += dx;
        }

        let mut dys_frac = vec![];
        if false {
            for idx in 1..dys.len() {
                dys_frac.push(dys[idx - 1] / dys[idx]);
            }
        } else {
            for idx in 1..dys.len() {
                dys_frac.push(dys[idx] / dys[idx - 1]);
            }
        }

        let mut avg_frac = dys_frac.iter().sum::<f64>() / dys_frac.len() as f64;
        if avg_frac < 0.0 {
            avg_frac = -1. * avg_frac;
        }

        let dy_ref = self.evaluate(ref_x) - reference_curve_2.evaluate(ref_x);
        let dy_new = dy_ref * avg_frac.powf((new_x - ref_x) / dx);
        let new_y_2 = reference_curve_2.evaluate(new_x) + dy_new;

        //let new_y = 50.;
        let new_y = 0.5 * (new_y_1 + new_y_2);
        //let new_y = new_y_2;

        if new_y.is_nan() {
            panic!()
        }
        if left_insert {
            self.insert_point(0, new_x, new_y);
        } else {
            self.add_point(new_x, new_y);
        }

        self
    }
}

fn create_extended_curves(
    curves: Vec<Curve>,
    dx: f64,
    x_range: Option<(f64, f64)>,
    params: &ParamManager,
) -> Vec<Curve> {
    // fix the passed parameters so the user doesn't need to be as careful
    let dx = dx.abs();
    let mut org_curves = curves.clone();
    // this sort is arranging the curves by their rightmost point (in descending order)
    org_curves.sort_by(|a, b| {
        let b_x = b.x_lims().1;
        let a_x = a.x_lims().1;
        b_x.partial_cmp(&a_x).unwrap()
    });

    let mut established_curves = Vec::<Curve>::new();

    let mut target_x = 0.;

    for curve in org_curves.iter_mut() {
        // the first curve should just simply be added.
        if established_curves.is_empty() {
            //established_curves.push(curve.clone());
            target_x = match x_range {
                None => curve.x_lims().1,
                Some((_, x_high)) => x_high,
            };
            //continue;
        }
        // following curves use preestablished curves to guide their extensions
        let mut ref_x = curve.x_lims().1;

        while ref_x < target_x {
            let mut new_x = ref_x + dx;
            let ref_y = curve.y_lims().1;
            if new_x > target_x {
                new_x = target_x;
            }

            // search for the closest spanning curve above and below
            // brute force approach
            let mut upper_spanning_curve: Option<Curve> = None;
            let mut lower_spanning_curve: Option<Curve> = None;
            let mut dy_upper = 0.0;
            let mut dy_lower = 0.0;

            for est_curve in established_curves.iter() {
                if !est_curve.contains_x(ref_x) {
                    continue;
                }

                let est_y = est_curve.evaluate(ref_x);
                let dy = ref_y - est_y;
                if dy >= 0. {
                    match lower_spanning_curve.clone() {
                        None => {
                            lower_spanning_curve = Some(est_curve.clone());
                            dy_lower = dy;
                        }
                        Some(_) => {
                            if dy < dy_lower {
                                lower_spanning_curve = Some(est_curve.clone());
                                dy_lower = dy;
                            }
                        }
                    }
                } else {
                    match upper_spanning_curve.clone() {
                        None => {
                            upper_spanning_curve = Some(est_curve.clone());
                            dy_upper = dy;
                        }
                        Some(_) => {
                            if dy > dy_upper {
                                upper_spanning_curve = Some(est_curve.clone());
                                dy_upper = dy;
                            }
                        }
                    }
                }
            } // for est_curve

            match (upper_spanning_curve, lower_spanning_curve) {
                // just linearly extend the current curve
                (None, None) => {
                    curve.extend_from_lowess(new_x, false, params);
                }
                // keep the delta between  this curve and the curve above constant
                (Some(upper_curve), None) => {
                    curve.extend_from_single_reference_curve(ref_x, new_x, &upper_curve, false);
                }
                // keep the delta between  this curve and the curve below constant
                (None, Some(lower_curve)) => {
                    curve.extend_from_single_reference_curve(ref_x, new_x, &lower_curve, false);
                }
                // keep the fractional distance between the lower curve and upper curve constant
                (Some(upper_curve), Some(lower_curve)) => {
                    curve.extend_from_double_reference_curve(
                        ref_x,
                        new_x,
                        &lower_curve,
                        &upper_curve,
                        false,
                    );
                }
            };

            ref_x = new_x;
        } // while ref_x
        established_curves.push(curve.clone());
    } // for curve

    let mut org_curves = established_curves.clone();
    // this sort is arranging the curves by their rightmost point
    org_curves.sort_by(|a, b| {
        let a_x = a.x_lims().0;
        let b_x = b.x_lims().0;
        a_x.partial_cmp(&b_x).unwrap()
    });
    established_curves.clear();

    for curve in org_curves.iter_mut() {
        // the first curve should just simply be added.
        if established_curves.is_empty() {
            //established_curves.push(curve.clone());
            target_x = match x_range {
                None => curve.x_lims().0,
                Some((x_low, _)) => x_low,
            };
            //continue;
        }

        // following curves use preestablished curves to guide their extensions
        let mut ref_x = curve.x_lims().0;

        while ref_x > target_x {
            let mut new_x = ref_x - dx;
            let ref_y = curve.y_lims().0;
            if new_x < target_x {
                new_x = target_x;
            }

            // search for the closest spanning curve above and below
            // brute force approach
            let mut upper_spanning_curve: Option<Curve> = None;
            let mut lower_spanning_curve: Option<Curve> = None;
            let mut dy_upper = 0.0;
            let mut dy_lower = 0.0;

            for est_curve in established_curves.iter() {
                if !est_curve.contains_x(ref_x) {
                    continue;
                }

                let est_y = est_curve.evaluate(ref_x);
                let dy = ref_y - est_y;
                if dy >= 0. {
                    match lower_spanning_curve.clone() {
                        None => {
                            lower_spanning_curve = Some(est_curve.clone());
                            dy_lower = dy;
                        }
                        Some(_) => {
                            if dy < dy_lower {
                                lower_spanning_curve = Some(est_curve.clone());
                                dy_lower = dy;
                            }
                        }
                    }
                } else if dy < 0.0 {
                    match upper_spanning_curve.clone() {
                        None => {
                            upper_spanning_curve = Some(est_curve.clone());
                            dy_upper = dy;
                        }
                        Some(_) => {
                            if dy > dy_upper {
                                upper_spanning_curve = Some(est_curve.clone());
                                dy_upper = dy;
                            }
                        }
                    }
                }
            } // for est_curve

            match (upper_spanning_curve, lower_spanning_curve) {
                // just linearly extend the current curve
                (None, None) => {
                    curve.extend_from_lowess(new_x, true, params);
                }
                //(None, None) => curve.insert_point(0, new_x, curve.evaluate(new_x)),
                // keep the delta between  this curve and the curve above constant
                (Some(upper_curve), None) => {
                    curve.extend_from_single_reference_curve(ref_x, new_x, &upper_curve, true);
                }
                // keep the delta between  this curve and the curve below constant
                (None, Some(lower_curve)) => {
                    curve.extend_from_single_reference_curve(ref_x, new_x, &lower_curve, true);
                }
                // keep the fractional distance between the lower curve and upper curve constant
                (Some(upper_curve), Some(lower_curve)) => {
                    curve.extend_from_double_reference_curve(
                        ref_x,
                        new_x,
                        &lower_curve,
                        &upper_curve,
                        true,
                    );
                }
            };
            ref_x = new_x;
        } // while ref_x
        established_curves.push(curve.clone());
    } // for curve

    established_curves
}
fn linearize(curves: &[Curve], xs: &[Option<f64>], ys: &[Option<f64>]) -> Vec<f64> {
    let z_lin = xs
        .par_iter()
        .zip(ys.par_iter())
        .progress()
        //.par_bridge()
        .map(|(x, y)| {
            //for (x, y) in xs.iter().zip(ys.iter()).progress() {
            let mut upper_spanning_curve: Option<Curve> = None;
            let mut lower_spanning_curve: Option<Curve> = None;
            let mut dy_upper = 0.0;
            let mut dy_lower = 0.0;

            let (x, y) = (x.unwrap(), y.unwrap());

            for curve in curves.iter() {
                if !curve.contains_x(x) {
                    continue;
                }

                let est_y = curve.evaluate(x);

                let dy = y - est_y;
                if dy >= 0. {
                    match lower_spanning_curve.clone() {
                        None => {
                            lower_spanning_curve = Some(curve.clone());
                            dy_lower = dy;
                        }
                        Some(_) => {
                            if dy < dy_lower {
                                lower_spanning_curve = Some(curve.clone());
                                dy_lower = dy;
                            }
                        }
                    }
                } else if dy < 0.0 {
                    match upper_spanning_curve.clone() {
                        None => {
                            upper_spanning_curve = Some(curve.clone());
                            dy_upper = dy;
                        }
                        Some(_) => {
                            if dy > dy_upper {
                                upper_spanning_curve = Some(curve.clone());
                                dy_upper = dy;
                            }
                        }
                    }
                }
            }

            match (upper_spanning_curve, lower_spanning_curve) {
                // just linearly extend the current curve
                (None, None) => f64::NAN,
                // keep the delta between  this curve and the curve above constant
                (Some(upper_curve), None) => upper_curve.y_lims().0 + dy_upper,
                // keep the delta between  this curve and the curve below constant
                (None, Some(lower_curve)) => lower_curve.y_lims().0 + dy_lower,
                // keep the fractional distance between the lower curve and upper curve constant
                (Some(upper_curve), Some(lower_curve)) => {
                    (upper_curve.y_lims().0 * dy_lower - lower_curve.y_lims().0 * dy_upper)
                        / (dy_lower - dy_upper)
                }
            }
        })
        .collect::<Vec<_>>();
    z_lin
}

fn predict_next_line(curves: &[Curve]) -> Curve {
    let mut curves: Vec<_> = Vec::from(curves);
    curves.sort_by(|c1, c2| c1.y_at(0).partial_cmp(&c2.y_at(0)).unwrap());

    let curve_2 = &curves[curves.len() - 1];
    let curve_1 = &curves[curves.len() - 2];

    let mut curve_3 = Curve::default();
    let mut x = curve_1.x_lims().0;
    let dx = (curve_1.x_lims().1 - curve_1.x_lims().0) / 500.;
    while x < curve_1.x_lims().1 {
        if x > curve_1.x_lims().1 {
            x = curve_1.x_lims().1;
        }
        let y = 2. * curve_2.evaluate(x) - curve_1.evaluate(x);
        curve_3.add_point(x, y);
        x += dx;
    }

    curve_3
}

/// extend curves and linearize data
#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    /// Data dir
    #[arg(short, long, default_value_t=String::from("data/faust_det_60"))]
    directory: String,
}

fn main() -> Result<()> {
    println!("\nExecuting {}\n", file!());
    let timer = std::time::Instant::now();

    let args = Args::parse();

    let params = ParamManager::new(&args.directory)?;

    let enumerated_curves = (0..fs::read_dir(Path::new(&args.directory).join("smoothed_lines/"))?
        .count())
        //.unwrap()
        //.enumerate()
        .map(|file_idx| {
            let path = Path::new(&args.directory)
                .join("smoothed_lines/")
                .join(format!("curve_{}.parquet", file_idx));
            //let file =
            //let path = file.unwrap().path();
            let file_name = path.to_str().unwrap();
            (
                file_idx,
                Curve::from_parquet_file(file_name, UseExtrapolated::None).unwrap(),
            )
        })
        .filter(|(_, curve)| curve.points().iter().map(|p| p.1).all(|y| !y.is_nan()))
        .filter(|(_, curve)| curve.points().iter().map(|p| p.0).all(|x| !x.is_nan()))
        .collect::<Vec<_>>();

    let file_name = Path::new(&args.directory).join("raw.parquet");
    let mut file = File::open(file_name)?;
    let df = ParquetReader::new(&mut file).finish()?;
    let mut df = df
        .lazy()
        .select(&[col(params.x_col()), col(params.y_col())])
        .with_column(col(params.x_col()).alias("x"))
        .with_column(col(params.y_col()).alias("y"))
        .drop_nulls(Some(vec!["x".into(), "y".into()]))
        .collect()?;
    let float_cmp = |a: f64, b: f64| a.partial_cmp(&b).unwrap();

    let right_extend_idx = enumerated_curves
        .iter()
        .max_by(|c1, c2| float_cmp(c1.1.x_lims().1, c2.1.x_lims().1))
        .unwrap()
        .0;
    let left_extend_idx = enumerated_curves
        .iter()
        .min_by(|c1, c2| float_cmp(c1.1.x_lims().0, c2.1.x_lims().0))
        .unwrap()
        .0;

    let curves = (0..fs::read_dir(Path::new(&args.directory).join("smoothed_lines/"))?.count())
        .map(|file_idx| {
            let path = Path::new(&args.directory)
                .join("smoothed_lines/")
                .join(format!("curve_{}.parquet", file_idx));
            //let file =
            //let path = file.unwrap().path();
            let file_name = path.to_str().unwrap();

            let use_extrapolated = if file_idx == left_extend_idx && file_idx == right_extend_idx {
                UseExtrapolated::Both
            } else if file_idx == left_extend_idx {
                UseExtrapolated::Left
            } else if file_idx == right_extend_idx {
                UseExtrapolated::Right
            } else {
                UseExtrapolated::None
            };
            Curve::from_parquet_file(file_name, use_extrapolated).unwrap()
        })
        .filter(|curve| curve.points().iter().map(|p| p.1).all(|y| !y.is_nan()))
        .filter(|curve| curve.points().iter().map(|p| p.0).all(|x| !x.is_nan()))
        .collect::<Vec<_>>();

    let (x_low, x_high) = (df["x"].min()?.unwrap(), df["x"].max()?.unwrap());
    let dx = (x_high - x_low) / 1000.;
    let mut extended_curves = create_extended_curves(curves, dx, Some((x_low, x_high)), &params);

    for _ in 0..params.num_extra_lines() {
        extended_curves.push(predict_next_line(&extended_curves));
    }

    let output_dir = Path::new(&args.directory).join("extended_curves");
    if fs::exists(Path::new(&output_dir))? {
        fs::remove_dir_all(&output_dir)?;
    }
    fs::create_dir_all(&output_dir)?;

    for (idx, curve) in extended_curves.iter().enumerate() {
        let xs = curve.points.iter().map(|(x, _)| *x).collect::<Vec<_>>();
        let ys = curve.points.iter().map(|(_, y)| *y).collect::<Vec<_>>();

        let mut df = df!(
            "x"=>&xs,
            "y"=>&ys,
        )?;
        let file = Path::new(&output_dir).join(format!("curve_{}.parquet", idx));
        let mut file = std::fs::File::create(file)?;

        ParquetWriter::new(&mut file).finish(&mut df)?;
    }

    let xs = df["x"].f64()?.into_iter().collect::<Vec<_>>();
    let ys = df["y"].f64()?.into_iter().collect::<Vec<_>>();
    let z_lin = linearize(&extended_curves, &xs, &ys);

    let df = df.with_column(Series::new("z_lin", z_lin))?;

    let output_dir = Path::new(&args.directory).join("linearized.parquet");
    let mut file = std::fs::File::create(output_dir)?;

    ParquetWriter::new(&mut file).finish(df)?;

    println!("\ttime: {} s", timer.elapsed().as_secs_f32());
    Ok(())
}
