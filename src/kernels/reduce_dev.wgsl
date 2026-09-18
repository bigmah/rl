// Sums of squared distances from a mean, for a standard deviation: reduce.wgsl's strided
// sums of (x - mean)^2, where the mean is total[0] / count and total is a sum already made.

struct P {
    n: u32, gx: u32,
    n_in: u32,
    inv_count: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> total: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let mean = total[0] * p.inv_count;
    var sum = 0.0;
    for (var j = i; j < p.n_in; j += p.n) {
        let d = input[j] - mean;
        sum += d * d;
    }
    output[i] = sum;
}
