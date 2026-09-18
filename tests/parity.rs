//! The Rust trainer against the MLX one, on the numbers tests/parity_dump.py wrote down:
//! the same policy and the same minibatch have to give the same loss, the same gradients,
//! and the same weights after Muon has stepped.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use pufferl::actor::Rollout;
use pufferl::config::Ini;
use pufferl::gpu::Gpu;
use pufferl::learner::{Hyper, Learner, LOSS_KEYS};
use pufferl::model::{Model, Shapes};

fn golden(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(case)
}

fn floats(dir: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn words(dir: &Path, name: &str) -> Vec<u32> {
    floats(dir, name).into_iter().map(f32::to_bits).collect()
}

/// The largest |got - want| / (atol + rtol * |want|), and where: at most 1 when they agree.
fn worst(got: &[f32], want: &[f32], rtol: f32, atol: f32) -> (f32, usize) {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).enumerate()
        .map(|(i, (g, w))| ((g - w).abs() / (atol + rtol * w.abs()), i))
        .fold((0.0, 0), |a, b| if b.0 > a.0 || b.0.is_nan() { b } else { a })
}

fn check(what: &str, got: &[f32], want: &[f32], rtol: f32, atol: f32) {
    let (error, at) = worst(got, want, rtol, atol);
    println!("  {what}: worst {error:.3} of tolerance, at {at}: {} against {}", got[at], want[at]);
    assert!(error <= 1.0, "{what} is off at {at}: {} against {}", got[at], want[at]);
}

fn run(case: &str) {
    let dir = golden(case);
    let mut ini = Ini::default();
    ini.load(&dir.join("case.ini"));
    let size = |key: &str| ini.size("case", key);
    let act_sizes: Vec<usize> = ini.text("case", "act_sizes").split(',')
        .filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
    let (height, width) = (size("picture_height"), size("picture_width"));
    let shapes = Shapes {
        obs_size: size("obs_size"),
        act_sizes,
        hidden: size("hidden"),
        layers: size("layers"),
        picture: (height > 0).then_some((height, width, 3)),
    };
    let (segments, horizon) = (size("segments"), size("horizon"));
    let f = |key: &str| ini.num("train", key) as f32;
    let hyper = Hyper {
        gamma: f("gamma"), gae_lambda: f("gae_lambda"), clip_coef: f("clip_coef"), vf_coef: f("vf_coef"),
        vf_clip_coef: f("vf_clip_coef"), max_grad_norm: f("max_grad_norm"), momentum: f("momentum"),
        weight_decay: f("weight_decay"), vtrace: ini.flag("train", "vtrace"),
        vtrace_rho_clip: f("vtrace_rho_clip"), vtrace_c_clip: f("vtrace_c_clip"), norm_adv: ini.flag("train", "norm_adv"),
    };

    let gpu = Rc::new(Gpu::new().expect("a GPU"));
    println!("{case} on {} ({:?})", gpu.info.name, gpu.info.backend);
    let model = Model::new(gpu.clone(), shapes, 0);
    let weights = floats(&dir, "weights.bin");
    assert_eq!(weights.len(), model.flat_len, "a checkpoint from MLX is laid out as the weights are here");
    model.write_weights(&weights);

    // The minibatch is the whole rollout: one segment for every agent
    let rollout = Rollout {
        agents: segments,
        horizon,
        obs: gpu.storage_from("obs", &floats(&dir, "obs.bin")),
        actions: gpu.storage("actions", segments * horizon * model.shapes.num_heads()),
        logprobs: gpu.storage_from("logprobs", &floats(&dir, "old_logprobs.bin")),
        values: gpu.storage_from("values", &floats(&dir, "old_values.bin")),
        rewards: gpu.storage_from("rewards", &floats(&dir, "rewards.bin")),
        terminals: gpu.storage_from("terminals", &floats(&dir, "terminals.bin")),
        initial_state: gpu.storage_from("state", &floats(&dir, "state.bin")),
    };
    gpu.write(&rollout.actions, 0, &words(&dir, "actions.bin"));
    let learner = Learner::new(&model, &rollout, segments, &hyper);
    learner.set_hyper(ini.num("case", "learning_rate") as f32, ini.num("case", "ent_coef") as f32);

    for step in 1.. {
        if !dir.join(format!("stats{step}.bin")).exists() {
            assert!(step > 1, "no golden numbers for {case}: run tests/parity_dump.py");
            break;
        }
        println!(" step {step}");
        learner.clear_stats();
        gpu.submit(&[learner.cut(0), &learner.forward_backward]);
        let stats = learner.read_stats();
        let want = floats(&dir, &format!("stats{step}.bin"));
        for (key, (got, want)) in LOSS_KEYS.iter().zip(stats.iter().zip(&want)) {
            println!("  {key}: {got} against {want}");
        }
        check("the loss statistics", &stats, &want, 1e-4, 1e-5);
        let grads: Vec<f32> = gpu.read(&learner.grads, 0, model.flat_len);
        check("the gradients", &grads, &floats(&dir, &format!("grads{step}.bin")), 1e-3, 1e-6);

        gpu.submit(&[&learner.optimizer]);
        check("the weights after Muon", &model.read_weights(), &floats(&dir, &format!("weights{step}.bin")), 1e-3, 1e-5);
    }
}

#[test]
fn plain() {
    run("plain");
}

#[test]
fn all() {
    run("all");
}

#[test]
fn picture() {
    run("picture");
}
