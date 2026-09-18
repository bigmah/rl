//! Where a training epoch's time goes on this GPU, at platformer's shapes and with no env:
//!
//!     cargo run --release --example bench_train [AGENTS HORIZON MINIBATCH OBS_SIZE [PICTURE_WIDTH PICTURE_HEIGHT]]
//!
//! OBS_SIZE is what comes before a picture, if there is one.

use std::rc::Rc;
use std::time::Instant;

use pufferl::actor::Actor;
use pufferl::gpu::{Gpu, Program};
use pufferl::learner::{Hyper, Learner};
use pufferl::model::{Model, Shapes};

fn time(gpu: &Gpu, what: &str, reps: usize, programs: &[&Program]) {
    gpu.submit(programs);
    gpu.wait();
    let start = Instant::now();
    for _ in 0..reps {
        gpu.submit(programs);
    }
    let encoded = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    gpu.wait();
    let steps: usize = programs.iter().map(|p| p.len()).sum();
    println!("{what:<28} {:8.3} ms, of which the CPU encoding {steps} steps {encoded:.3} ms",
        start.elapsed().as_secs_f64() * 1000.0 / reps as f64);
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let (agents, horizon, minibatch, features, picture) = match args[..] {
        [a, h, m, o] => (a, h, m, o, None),
        [a, h, m, o, width, height] => (a, h, m, o, Some((height, width, 3))),
        _ => (1024, 64, 8192, 413, None),
    };
    let obs_size = features + picture.map_or(0, |(h, w, c)| h * w * c);
    let gpu = Rc::new(Gpu::new().unwrap());
    println!("{} ({:?}): {agents} agents, horizon {horizon}, minibatch {minibatch}, observations of {obs_size}",
        gpu.info.name, gpu.info.backend);
    let shapes = Shapes { obs_size, act_sizes: vec![3, 3], hidden: 128, layers: 4, picture };
    let model = Model::new(gpu.clone(), shapes, 1);
    let mut actor = Actor::new(&model, agents, horizon, 1);
    let hyper = Hyper { gamma: 0.995, gae_lambda: 0.9, clip_coef: 0.2, vf_coef: 2.0, vf_clip_coef: 0.2,
        max_grad_norm: 1.5, momentum: 0.95, weight_decay: 0.0, vtrace: false, vtrace_rho_clip: 1.0,
        vtrace_c_clip: 1.0, norm_adv: false };
    let learner = Learner::new(&model, &actor.rollout, minibatch / horizon, &hyper);
    learner.set_hyper(0.0, 0.001);

    let obs = vec![0.1f32; agents * obs_size];
    let terminals = vec![0f32; agents];
    for t in 0..horizon {
        actor.upload(&obs, &terminals, t);
        actor.run(t, true);
        actor.actions();
    }
    let (mut upload, mut run, mut read) = (0.0, 0.0, 0.0);
    for t in 0..horizon {
        let t0 = Instant::now();
        actor.upload(&obs, &terminals, t);
        let t1 = Instant::now();
        actor.run(t, true);
        let t2 = Instant::now();
        actor.actions();
        upload += (t1 - t0).as_secs_f64();
        run += (t2 - t1).as_secs_f64();
        read += t2.elapsed().as_secs_f64();
    }
    let ms = |seconds: f64| seconds * 1000.0 / horizon as f64;
    println!("{:<28} {:8.3} ms: upload {:.3}, encode and submit {:.3}, wait and read {:.3}", "a step of acting",
        ms(upload + run + read), ms(upload), ms(run), ms(read));

    time(&gpu, "a step's work on the GPU", 50, &[actor.step_program()]);
    time(&gpu, "cutting a minibatch", 20, &[learner.cut(0)]);
    time(&gpu, "forward, loss and backward", 20, &[&learner.forward_backward]);
    time(&gpu, "Muon", 20, &[&learner.optimizer]);
    time(&gpu, "a whole minibatch", 20, &[learner.cut(0), &learner.forward_backward, &learner.optimizer]);
}
