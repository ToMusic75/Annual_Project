extern crate rulinalg;
extern crate rand;

pub mod model {
    pub trait SupervisedLearning<I, O> {
        fn fit(&mut self, inputs: &I, outputs: &O) -> Result<(), ()>;

        fn predict(&self, inputs: &I) -> Result<O, ()>;
    }

    pub trait UnsupervisedLearning<I, O> {
        fn fit(&mut self, inputs: &I) -> Result<(), ()>;

        fn predict(&self, inputs: &I) -> Result<O, ()>;
    }

    pub mod linear_regression;
    pub mod perceptron;
    pub mod mlp;
    pub mod rbf;
    pub mod kmeans;
    pub mod externs;
    pub mod utils;
}