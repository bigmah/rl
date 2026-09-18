// z = c * s^2 + b * s + a * I, the polynomial of a Newton-Schulz step, for square s: what
// multiplies x to make the next one.

struct P {
    n: u32, gx: u32,
    size: u32,  // the side of one square
    a: f32,
    b: f32,
    c: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> s_squared: array<f32>;
@group(0) @binding(3) var<storage, read_write> z: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let e = i % (p.size * p.size);
    let diagonal = (e / p.size) == (e % p.size);
    z[i] = p.c * s_squared[i] + p.b * s[i] + select(0.0, p.a, diagonal);
}
