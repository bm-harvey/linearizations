use std::path::Path;

pub struct ParamManager {
    x_col: String,
    y_col: String,
    sigma: f64,
    threshold: f64,
    bins_x: u64,
    bins_y: u64,
    bandwidth: f64,
    polynomial_order: u64,
    num_extra_lines: u64,
}

impl Default for ParamManager {
    fn default() -> Self {
        Self {
            // gloabal
            x_col: "x".into(),
            y_col: "y".into(),
            // ridge finding
            bins_x: 1_024,
            bins_y: 1_024,
            sigma: 1.0,
            threshold: 5e-4,
            // lowess
            polynomial_order: 1,
            bandwidth: 0.1,
            // linearization
            num_extra_lines: 2,
        }
    }
}

// probably should not be using anyhow here but whatever tbh
use anyhow::Result;

impl ParamManager {
    pub fn new(dir_path: &str) -> Result<Self> {
        let mut result = Self::default();

        let param_file = Path::new(dir_path).join("params.dat");
        let param_data = std::fs::read_to_string(param_file)?;
        for (idx, line) in param_data.lines().enumerate() {
            if line.trim().starts_with("#") {
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }
            let words = line.split(":").collect::<Vec<_>>();
            if words.len() < 2 {
                println!(
                    "Warning: Line {} : \"{}\" is improperly formatted ",
                    idx, line
                );
            }
            let param_name = words[0].trim();
            let param_value = words[1].trim();
            match param_name {
                "x_col" => result.x_col = param_value.into(),
                "y_col" => result.y_col = param_value.into(),
                "bins_x" => result.bins_x = param_value.parse::<u64>()?,
                "bins_y" => result.bins_y = param_value.parse::<u64>()?,
                "sigma" => result.sigma = param_value.parse::<f64>()?,
                "threshold" => result.threshold = param_value.parse::<f64>()?,
                "bandwidth" => result.bandwidth = param_value.parse::<f64>()?,
                "polynomial_order" => result.polynomial_order = param_value.parse::<u64>()?,
                "num_extra_lines" => result.num_extra_lines = param_value.parse::<u64>()?,
                _ => {
                    println!("Parameter \"{}\" not recognized and was ignored", param_name)
                }
            }
        }

        //for line in f

        Ok(result)
    }

    pub fn x_col(&self) -> &str {
        &self.x_col
    }
    pub fn y_col(&self) -> &str {
        &self.y_col
    }
    pub fn num_bins_x(&self) -> u64 {
        self.bins_x
    }
    pub fn num_bins_y(&self) -> u64 {
        self.bins_y
    }
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }
    pub fn polynomial_order(&self) -> u64 {
        self.polynomial_order
    }
    pub fn num_extra_lines(&self) -> u64 {
        self.num_extra_lines
    }
}
