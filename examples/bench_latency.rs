//! What a round trip to this GPU costs with nothing much to do there: a small kernel, a copy
//! to where the CPU can read, and the read. Acting pays this once a step.

use std::time::Instant;

use pufferl::gpu::{Gpu, Params, Program};

const KERNEL: &str = "
struct P { n: u32, gx: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i < p.n) { x[i] += 1u; }
}";

fn main() {
    let gpu = Gpu::new().unwrap();
    let n = 2048;
    let x = gpu.storage("x", n);
    let staging = gpu.staging("staging", n);
    for ops in [1usize, 12, 40] {
        let mut program = Program::new();
        for _ in 0..ops {
            program.push(gpu.op_flat("bump", KERNEL, n, Params::new(), &[&x.buffer]));
        }
        program.copy(&x, 0, &staging, 0, n);
        let reps = 300;
        let mut last = 0;
        let start = Instant::now();
        for _ in 0..reps {
            gpu.submit(&[&program]);
            last = gpu.read_staging(&staging, n, |data: &[u32]| data[0]);
        }
        println!("{ops:>3} small dispatches and a read: {:.3} ms a round trip (counted to {last})",
            start.elapsed().as_secs_f64() * 1000.0 / reps as f64);
    }
}
