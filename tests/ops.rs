//! The kernels that everything else is made of, against the same arithmetic on the CPU.

use std::rc::Rc;

use pufferl::gpu::{Gpu, Program};
use pufferl::ops::{self, matmul, matmul_batched, Out, Reduce, Sum, View};
use pufferl::rng::Rng;

fn randoms(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| 2.0 * rng.uniform() - 1.0).collect()
}

fn run(gpu: &Gpu, ops: Vec<pufferl::gpu::Op>) {
    let mut program = Program::new();
    program.extend(ops);
    gpu.submit(&[&program]);
}

/// Every way the trainer multiplies: plain, either side transposed, a block of a wider
/// matrix on either side, into a block of a wider result, added to what is there, and
/// with a k long enough to be cut into layers.
#[test]
fn matmul_against_cpu() {
    let gpu = Rc::new(Gpu::new().expect("a GPU"));
    let mut rng = Rng::new(1);
    // m, n, k, a transposed, b transposed, accumulate
    let cases = [
        (1, 1, 1, false, false, false),
        (5, 7, 3, false, false, false),
        (33, 65, 17, false, true, false),
        (64, 32, 128, true, false, true),
        (7, 130, 1500, true, false, false),
        (130, 9, 2049, false, true, true),
        (3, 5, 4096, true, true, false),
    ];
    for (m, n, k, a_t, b_t, accumulate) in cases {
        // Both inputs are blocks of wider matrices, some way into their buffers
        let (a_off, b_off, c_off, pad) = (3, 5, 2, 4);
        let (a_rows, a_cols) = if a_t { (k, m + pad) } else { (m, k + pad) };
        let (b_rows, b_cols) = if b_t { (n, k + pad) } else { (k, n + pad) };
        let a_data = randoms(&mut rng, a_off + a_rows * a_cols);
        let b_data = randoms(&mut rng, b_off + b_rows * b_cols);
        let c_stride = n + pad;
        let c_data = randoms(&mut rng, c_off + m * c_stride);

        let a = gpu.storage_from("a", &a_data);
        let b = gpu.storage_from("b", &b_data);
        let c = gpu.storage_from("c", &c_data);
        let view = |buf, off, cols, transposed| if transposed { View::transposed(buf, off, cols) } else { View::rows(buf, off, cols) };
        let mut out = Out::at(&c, c_off, c_stride);
        if accumulate {
            out = out.accumulate();
        }
        run(&gpu, matmul(&gpu, m, n, k, view(&a, a_off, a_cols, a_t), view(&b, b_off, b_cols, b_t), out));
        let got: Vec<f32> = gpu.read(&c, 0, c_data.len());

        let a_at = |i: usize, x: usize| a_data[a_off + if a_t { x * a_cols + i } else { i * a_cols + x }] as f64;
        let b_at = |x: usize, j: usize| b_data[b_off + if b_t { j * b_cols + x } else { x * b_cols + j }] as f64;
        for (at, (&got, &before)) in got.iter().zip(&c_data).enumerate() {
            let inside = at >= c_off && (at - c_off) % c_stride < n && (at - c_off) / c_stride < m;
            let want = if inside {
                let (i, j) = ((at - c_off) / c_stride, (at - c_off) % c_stride);
                let product: f64 = (0..k).map(|x| a_at(i, x) * b_at(x, j)).sum();
                product + if accumulate { before as f64 } else { 0.0 }
            } else {
                before as f64
            };
            assert!((got as f64 - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "{m}x{n} over {k}: c[{at}] is {got}, not {want}");
        }
    }
}

/// Matrices of one shape multiplied together, as Muon does the MinGRU's layers: with a k
/// short enough to go in one piece, one cut fine because the product is small, and a long one.
#[test]
fn batched_matmul_against_cpu() {
    let gpu = Rc::new(Gpu::new().expect("a GPU"));
    let mut rng = Rng::new(3);
    for (batches, m, n, k) in [(4, 12, 36, 12), (3, 128, 128, 384), (2, 9, 5, 1300), (5, 300, 300, 40)] {
        // Each batch's matrices are a little further apart than they are big
        let (a_bs, b_bs, c_bs) = (m * k + 3, k * n + 1, m * n + 2);
        let a_data = randoms(&mut rng, batches * a_bs);
        let b_data = randoms(&mut rng, batches * b_bs);
        let a = gpu.storage_from("a", &a_data);
        let b = gpu.storage_from("b", &b_data);
        let c = gpu.storage("c", batches * c_bs);
        // b is stored transposed, as x is for x x^T
        run(&gpu, matmul_batched(&gpu, batches, m, n, k, View::rows(&a, 0, k).batched(a_bs),
            View::transposed(&b, 0, k).batched(b_bs), Out::new(&c, n).batched(c_bs)));
        let got: Vec<f32> = gpu.read(&c, 0, batches * c_bs);
        for batch in 0..batches {
            for (i, j) in [(0, 0), (m - 1, n - 1), (m / 2, n / 3), (1, n - 1)] {
                let want: f64 = (0..k).map(|x| a_data[batch * a_bs + i * k + x] as f64 * b_data[batch * b_bs + j * k + x] as f64).sum();
                let value = got[batch * c_bs + i * n + j];
                assert!((value as f64 - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "batch {batch} of {m}x{n} over {k}: c[{i}][{j}] is {value}, not {want}");
            }
            assert!(got[batch * c_bs + m * n..(batch + 1) * c_bs].iter().all(|&v| v == 0.0), "wrote between batches");
        }
    }
}

#[test]
fn sums_against_cpu() {
    let gpu = Rc::new(Gpu::new().expect("a GPU"));
    let mut rng = Rng::new(2);
    for (n, columns) in [(1, 1), (8, 8), (1000, 1), (4096, 8), (100_003, 1), (80_000, 8)] {
        let data = randoms(&mut rng, n);
        let input = gpu.storage_from("input", &data);
        let scratch = gpu.storage("scratch", ops::scratch_len(columns));
        for sum in [Sum::Values, Sum::Squares] {
            let output = gpu.storage_from("output", &vec![1.0; columns + 1]);
            run(&gpu, ops::reduce(&gpu, Reduce { input: &input, in_off: 0, n_in: n, columns, sum,
                output: &output, out_off: 1, accumulate: true, scratch: &scratch }));
            let got: Vec<f32> = gpu.read(&output, 0, columns + 1);
            assert_eq!(got[0], 1.0, "a sum wrote before where it was told to");
            for column in 0..columns {
                let want: f64 = 1.0 + data.iter().skip(column).step_by(columns)
                    .map(|&v| if sum == Sum::Squares { (v * v) as f64 } else { v as f64 }).sum::<f64>();
                assert!((got[column + 1] as f64 - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "{n} numbers in {columns} columns: sum {column} is {}, not {want}", got[column + 1]);
            }
        }
    }

    // A mean and a deviation: advantages with a mean far from 0, where a sum of squares
    // less a squared mean would lose the deviation to rounding
    let n = 8192;
    let data: Vec<f32> = randoms(&mut rng, n).into_iter().map(|v| 100.0 + 0.01 * v).collect();
    let input = gpu.storage_from("input", &data);
    let sums = gpu.storage("sums", 2);
    let scratch = gpu.storage("scratch", ops::scratch_len(1));
    run(&gpu, ops::mean_and_deviation(&gpu, &input, n, &sums, &scratch));
    let got: Vec<f32> = gpu.read(&sums, 0, 2);
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let deviation = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>();
    assert!((got[0] as f64 / n as f64 - mean).abs() < 1e-3, "mean {} is not {mean}", got[0] / n as f32);
    assert!((got[1] as f64 - deviation).abs() < 0.05 * deviation, "deviation {} is not {deviation}", got[1]);
}
