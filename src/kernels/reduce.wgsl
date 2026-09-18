// Sums. Thread i adds up every stride'th number of the input from i on, so n threads turn
// n_in numbers into n sums, and a second pass with a smaller stride turns those into fewer.
// Numbers that share a column of rows `stride` wide stay together, which is how eight
// loss statistics laid out side by side come out as eight sums.
//
// mode 0 sums x, mode 1 sums x * x.

struct P {
    n: u32, gx: u32,
    n_in: u32,
    in_off: u32,
    out_off: u32,
    mode: u32,
    accumulate: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    var sum = 0.0;
    for (var j = i; j < p.n_in; j += p.n) {
        let v = input[p.in_off + j];
        sum += select(v, v * v, p.mode == 1u);
    }
    sum *= p.scale;
    if (p.accumulate != 0u) {
        output[p.out_off + i] += sum;
    } else {
        output[p.out_off + i] = sum;
    }
}
