//! How fast a matmul kernel is on this GPU, at the shapes the trainer multiplies:
//!
//!     cargo run --release --example bench_matmul [KERNEL.wgsl TX TY OX OY]
//!
//! A kernel is given with the threads of its workgroup across and down, and the outputs
//! each thread computes across and down. With none, the trainer's own.

use std::time::Instant;

use pufferl::gpu::{Gpu, Params, Program};

struct Shape {
    name: &'static str,
    m: usize, n: usize, k: usize,
    a: (usize, usize), b: (usize, usize),
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| args[i].parse::<usize>().unwrap();
    let (source, tx, ty, ox, oy) = if args.len() >= 5 {
        (std::fs::read_to_string(&args[0]).unwrap(), num(1), num(2), num(3), num(4))
    } else {
        (include_str!("../src/kernels/matmul.wgsl").to_string(), 8, 8, 4, 4)
    };
    // A sixth: k is cut into chunks this long, a layer of the dispatch for each
    let chunk = if args.len() >= 6 { num(5) } else { 512 };

    let gpu = Gpu::new().unwrap();
    println!("{} ({:?})", gpu.info.name, gpu.info.backend);
    let (rows, hid, obs) = (8192, 128, 413);
    let shapes = [
        // y = x w^T: a is x (rows, in), b is w^T
        Shape { name: "layer forward   8192x128 -> 384", m: rows, n: 3 * hid, k: hid, a: (hid, 1), b: (1, hid) },
        // gx = gy w
        Shape { name: "layer grad x    8192x384 -> 128", m: rows, n: hid, k: 3 * hid, a: (3 * hid, 1), b: (hid, 1) },
        // gw = gy^T x
        Shape { name: "layer grad w    384x128 over 8192", m: 3 * hid, n: hid, k: rows, a: (1, 3 * hid), b: (hid, 1) },
        Shape { name: "encoder forward 8192x413 -> 128", m: rows, n: hid, k: obs, a: (obs, 1), b: (1, obs) },
        Shape { name: "encoder grad w  128x413 over 8192", m: hid, n: obs, k: rows, a: (1, hid), b: (obs, 1) },
        // The same two with gy and x transposed first, so that both run along k
        Shape { name: "layer grad w    transposed inputs", m: 3 * hid, n: hid, k: rows, a: (rows, 1), b: (1, rows) },
        Shape { name: "encoder grad w  transposed inputs", m: hid, n: obs, k: rows, a: (rows, 1), b: (1, rows) },
        Shape { name: "rollout layer   1024x128 -> 384", m: 1024, n: 3 * hid, k: hid, a: (hid, 1), b: (1, hid) },
        Shape { name: "newton-schulz   z x: 128x128 x 128x384", m: hid, n: 3 * hid, k: hid, a: (hid, 1), b: (3 * hid, 1) },
        Shape { name: "newton-schulz   x x^T: 128x384 x 384x128", m: hid, n: hid, k: 3 * hid, a: (3 * hid, 1), b: (1, 3 * hid) },
        Shape { name: "newton-schulz   y s: 128x128 x 128x128", m: hid, n: hid, k: hid, a: (hid, 1), b: (hid, 1) },
    ];

    for shape in &shapes {
        let (m, n, k) = (shape.m, shape.n, shape.k);
        let a_len = (m - 1) * shape.a.0 + (k - 1) * shape.a.1 + 1;
        let b_len = (k - 1) * shape.b.0 + (n - 1) * shape.b.1 + 1;
        let a_data: Vec<f32> = (0..a_len).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
        let b_data: Vec<f32> = (0..b_len).map(|i| ((i * 104729) % 1000) as f32 / 1000.0 - 0.5).collect();
        let a = gpu.storage_from("a", &a_data);
        let b = gpu.storage_from("b", &b_data);
        let layers = k.div_ceil(chunk.min(k));
        let c = gpu.storage("c", m * n * layers);
        let params = Params::new().u(m).u(n).u(k).u(0).u(shape.a.0).u(shape.a.1).u(0).u(shape.b.0).u(shape.b.1)
            .u(0).u(n).u(0).u(chunk.min(k));
        let groups = [n.div_ceil(tx * ox) as u32, m.div_ceil(ty * oy) as u32, layers as u32];
        let op = gpu.op("matmul", &source, &params, &[&a.buffer, &b.buffer, &c.buffer], groups);
        let mut program = Program::new();
        program.push(op.clone());

        gpu.submit(&[&program]);
        gpu.wait();
        // Against the CPU, on a few numbers of the result
        let mut got: Vec<f32> = gpu.read(&c, 0, m * n * layers);
        for layer in 1..layers {
            for i in 0..m * n {
                got[i] += got[layer * m * n + i];
            }
        }
        let mut worst = 0f32;
        for &(i, j) in &[(0, 0), (m - 1, n - 1), (m / 2, n / 3), (1, n - 2), (m - 2, 1)] {
            let want: f64 = (0..k).map(|x| a_data[i * shape.a.0 + x * shape.a.1] as f64 * b_data[x * shape.b.0 + j * shape.b.1] as f64).sum();
            worst = worst.max((got[i * n + j] as f64 - want).abs() as f32 / (1.0 + want.abs() as f32));
        }

        // Many of it in one submit, so that what is timed is the GPU and not the way to it
        let reps = 50;
        let mut repeated = Program::new();
        for _ in 0..reps {
            repeated.push(op.clone());
        }
        gpu.submit(&[&repeated]);
        gpu.wait();
        let start = Instant::now();
        gpu.submit(&[&repeated]);
        gpu.wait();
        let ms = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let gflops = 2.0 * (m * n * k) as f64 / 1e9 / (ms / 1000.0);
        println!("{:<42} {ms:8.3} ms  {gflops:7.1} GFLOPS   error {worst:.1e}", shape.name);
    }
}
