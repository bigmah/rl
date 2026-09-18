// pufferl.cu's sample_logits: an action for each head of each agent, drawn from the
// softmax of the head's logits, with the log-probability of the draw and the value beside
// it. All three go into the rollout at this step, and the actions out to the env.
//
// The draw is by inverse CDF on a number hashed from the seed, the step and the agent,
// so a run is the same run again from the same seed.

//HEADS

struct P {
    n: u32, gx: u32,
    horizon: u32,
}

struct Step {
    t: u32,
    counter: u32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> tick: Step;
@group(0) @binding(2) var<storage, read> dec: array<f32>;
@group(0) @binding(3) var<storage, read_write> act_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> roll_actions: array<u32>;
@group(0) @binding(5) var<storage, read_write> roll_logprobs: array<f32>;
@group(0) @binding(6) var<storage, read_write> roll_values: array<f32>;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn uniform01(agent: u32, head: u32) -> f32 {
    var h = pcg(tick.seed ^ 0x9E3779B9u);
    h = pcg(h ^ tick.counter);
    h = pcg(h ^ (agent * 0x85EBCA6Bu + 1u));
    h = pcg(h + head);
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x + gid.y * p.gx;
    if (agent >= p.n) {
        return;
    }
    let base = agent * (NUM_LOGITS + 1u);
    let slot = agent * p.horizon + tick.t;
    var offset = 0u;
    var logprob = 0.0;
    for (var h = 0u; h < NUM_HEADS; h++) {
        let A = ACT_SIZES[h];
        var top = dec[base + offset];
        for (var j = 1u; j < A; j++) {
            top = max(top, dec[base + offset + j]);
        }
        var total = 0.0;
        for (var j = 0u; j < A; j++) {
            total += exp(dec[base + offset + j] - top);
        }
        let lse = top + log(total);

        let u = uniform01(agent, h);
        var cumulative = 0.0;
        var chosen = A - 1u;
        var found = false;
        for (var j = 0u; j < A; j++) {
            cumulative += exp(dec[base + offset + j] - lse);
            if (!found && u < cumulative) {
                chosen = j;
                found = true;
            }
        }
        logprob += dec[base + offset + chosen] - lse;
        act_out[agent * NUM_HEADS + h] = chosen;
        roll_actions[slot * NUM_HEADS + h] = chosen;
        offset += A;
    }
    roll_logprobs[slot] = logprob;
    roll_values[slot] = dec[base + NUM_LOGITS];
}
