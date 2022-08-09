pub type Point = (f64, f64);
use serde::ser::{Serialize, SerializeStruct, Serializer};

pub struct TrainingResult {
    min_error: f64,
    intercept: f64,
    slope: f64,
}

impl Serialize for TrainingResult{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer
    {
        let mut s = serializer.serialize_struct("TrainingResult", 3)?;
        s.serialize_field("min_error", &self.min_error)?;
        s.serialize_field("intercept", &self.intercept)?;
        s.serialize_field("slope", &self.slope)?;
        s.end()
        
    }
}

impl TrainingResult {
    pub fn new() -> Self {
        TrainingResult {
            min_error: 9999999999999999999999999.,
            intercept: 0.,
            slope: 0.,
        }
    }
}

pub struct LinearRegressionModel {
    epochs: usize,
    pub graph: Vec<Point>,
    pub best_result: TrainingResult,
}

impl LinearRegressionModel {
    fn round_to_n(number: f64, digits: u32) -> f64 {
        let exp = 10_u32.pow(digits) as f64;
        ((number * exp).round()) / exp
    }

    fn close_enough(number: &mut f64) {
        if 1. - (*number % 1.) <= 0.0001 {
            *number = number.round();
        }
    }

    pub fn new_uninit() -> Self {
        LinearRegressionModel {
            epochs: 100,
            graph: vec![],
            best_result: TrainingResult::new(),
        }
    }

    pub fn new(input_graph: Vec<Point>) -> Self {
        LinearRegressionModel {
            epochs: 100,
            graph: input_graph,
            best_result: TrainingResult::new(),
        }
    }

    pub fn set_epochs(&mut self, value: usize) {
        self.epochs = value;
    }

    pub fn add_points(&mut self, points: &[Point]) {
        for point in points {
            self.graph.push(*point);
        }
    }

    pub fn train(&mut self) {
        for x in (1..=100000).rev() {
            let mut tr = TrainingResult::new();
            let learn_rate: f64 = x as f64 / 1000000.;

            for i in 0..self.graph.len() * self.epochs {
                let prediction = tr.intercept + tr.slope * self.graph[i % self.graph.len()].0;
                let error = prediction - self.graph[i % self.graph.len()].1;
                tr.intercept -= learn_rate * error;
                tr.slope -= (learn_rate * error) * self.graph[i % self.graph.len()].0;
                if error.abs() < tr.min_error / i as f64{
                    tr.min_error += error.abs();
                }
            }
            tr.min_error /= (self.graph.len() - 1) as f64;
            if tr.min_error.abs() < self.best_result.min_error {
                tr.min_error = tr.min_error.abs();
                Self::close_enough(&mut tr.slope);
                Self::close_enough(&mut tr.intercept);
                self.best_result = tr;
            }
        }
        println!(
            "best result:\nslope: {}\nintercept: {}\nerror: {}",
            self.best_result.slope, self.best_result.intercept, self.best_result.min_error
        );
    }

    pub fn predict(self, input: f64) -> f64 {
        Self::round_to_n(
            self.best_result.intercept + self.best_result.slope * input,
            2,
        )
    }
}
