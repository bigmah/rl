// Keep this step's observations in the rollout, which is laid out agent by agent: every
// agent's horizon of steps together, as the minibatches that train on it are cut.

struct P {
    n: u32, gx: u32,
    obs_size: u32,
    horizon: u32,
}

struct Step {
    t: u32,
    counter: u32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> tick: Step;
@group(0) @binding(2) var<storage, read> obs: array<f32>;
@group(0) @binding(3) var<storage, read_write> roll_obs: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let agent = i / p.obs_size;
    let j = i % p.obs_size;
    roll_obs[(agent * p.horizon + tick.t) * p.obs_size + j] = obs[i];
}
