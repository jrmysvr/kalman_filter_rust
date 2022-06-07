#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(deprecated)]

use rand::Rng;
use rand::{thread_rng};
use gnuplot::{Figure, Graph, Caption, LineStyle, Dash};
use nalgebra::SMatrix;
use nalgebra::linalg::try_invert_to;

const DT: f64 = 1.0;

type M3x3 = SMatrix<f64, 3, 3>;
type M1 = SMatrix<f64, 1, 1>;
type M1x3 = SMatrix<f64, 1, 3>;
type M3x1 = SMatrix<f64, 3, 1>;

type F = M3x3;
type H = M1x3;
type Q = M3x3;
type R = M1;
type P = M3x3;
type X = M3x1;
type S = M1;


// https://en.wikipedia.org/wiki/KalmanFilter
struct KalmanFilter {
    f: F,
    h: H,
    q: Q,
    r: R,
    p: P,
    x: X,
}

impl KalmanFilter {
    fn new() -> KalmanFilter {
        KalmanFilter {
            f: F::new(1.0, DT, 0.0,
                      0.0, 1.0, DT,
                      0.0, 0.0, 1.0),

            h: H::new(1.0, 0.0, 0.0),

            q: Q::new(0.05, 0.05, 0.0,
                      0.05, 0.05, 0.0,
                      0.0, 0.0, 0.0),

            r: R::new(0.5),

            p: P::identity(),

            x: X::repeat(0.0),
        }
    }

    fn predict(&mut self) -> X {
        self.x = self.f * self.x;
        self.p = (self.f * self.p) * self.f.transpose() + self.q;
        self.x
    }

    fn update(&mut self, value: f64) {
        let y = value - (self.h.dot(&self.x.transpose()));

        let s = self.r + self.h * (self.p * self.h.transpose());
        let mut s_inv = S::new(0.0);
        if ! try_invert_to(s, &mut s_inv) { panic!("S was not inverted"); };

        let K = (self.p * self.h.transpose()) * s_inv;
        self.x = self.x + K * y;
        let I = M3x3::identity();
        self.p = ((I - K * self.h) * self.p) *
                  (I - (K * self.h).transpose()) +
                  ((K * self.r) * K.transpose());
    }
}

// https://en.wikipedia.org/wiki/Alpha_beta_filter
struct AlphaBetaFilter {
    a: f64,
    b: f64,
    xk_1: f64,
    vk_1: f64,
}

impl AlphaBetaFilter {
    fn new(a: f64, b: f64) -> AlphaBetaFilter {
        AlphaBetaFilter {
            a: a,
            b: b,
            xk_1: 0.0,
            vk_1: 0.0,
        }
    }
}


trait Filter {
    fn do_filter(&mut self, value: f64) -> f64;
}


impl Filter for KalmanFilter {
    fn do_filter(&mut self, value: f64) -> f64 {
        let prediction = self.predict();
        self.update(value);
        self.h.dot(&prediction.transpose())
    }
}

impl Filter for AlphaBetaFilter {
    fn do_filter(&mut self, value: f64) -> f64 {
        let mut xk = self.xk_1 + (self.vk_1 * DT);
        let mut vk = self.vk_1;
        let rk = value - xk;
        xk += self.a * rk;
        vk += (self.b * rk) / DT;

        self.xk_1 = xk;
        self.vk_1 = vk;

        xk
    }
}

fn apply_filter<F: Filter>(value: &f64, filter: &mut F) -> f64 {
    filter.do_filter(*value)
}

fn main() {
    /*
     * Inspired by: https://github.com/zziz/kalman-filter
     */
    let mut ab_filter = AlphaBetaFilter::new(0.5, 0.1);
    let mut kalman_filter = KalmanFilter::new();

    let mut rng = thread_rng();
    let n_data = 100;
    let mx = 10_f64;
    let mx2 = 100;

    // let data: Vec<f64> = (0..n_data).map(|d| ((d as f64)/mx).sin()).collect();
    let data: Vec<f64> = (-mx2..mx2).map(|d|
        // -(x**2 + 2x - 2)
        -((((d as f64)/mx).powf(2.0)) + 2.0 * ((d as f64)/mx) - 2.0)
        ).collect();

    let noisy_data: Vec<f64> = data.iter()
                                //.map(|d| d + rng.gen_range(-0.2, 0.2))
                                .map(|d| d + rng.gen_range(-5.0, 5.0))
                                .collect();

    let filtered_data: Vec<f64> = noisy_data.iter()
                                // .map(|d| apply_filter(d, &mut ab_filter))
                                .map(|d| apply_filter(d, &mut kalman_filter))
                                .collect();

    // let range: Vec<usize> = (0..n_data).collect();
    let range: Vec<usize> = (0..2*n_data).collect();

    let mut fig = Figure::new();
    fig.axes2d()
        .set_legend(Graph(0.5), Graph(0.9), &[], &[])
        .lines(
            &range[..],
            &data[..],
            &[Caption("Data"),
              LineStyle(Dash)]
        )
        .lines(
            &range[..],
            &noisy_data[..],
            &[Caption("Noisy Data")]
        )
        .lines(
            &range[..],
            &filtered_data[..],
            &[Caption("Filtered Data")]
        );

    fig.show().unwrap();
}


