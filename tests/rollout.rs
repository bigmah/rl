//! Acting and training are different kernels over the same policy, a step at a time and a
//! segment at a time. These hold them to each other.

use std::collections::HashMap;
use std::rc::Rc;

use pufferl::actor::Actor;
use pufferl::gpu::Gpu;
use pufferl::learner::{Hyper, Learner};
use pufferl::model::{Model, Shapes};
use pufferl::rng::Rng;

fn hyper() -> Hyper {
    Hyper { gamma: 0.995, gae_lambda: 0.9, clip_coef: 0.2, vf_coef: 2.0, vf_clip_coef: 0.2, max_grad_norm: 1.5,
        momentum: 0.95, weight_decay: 0.0, vtrace: false, vtrace_rho_clip: 1.0, vtrace_c_clip: 1.0, norm_adv: false }
}

/// A rollout trained on before the weights have moved is on-policy: the learner has to find
/// the log-probabilities and values the actor wrote down, episode ends, carried state and all.
fn on_policy(picture: Option<(usize, usize, usize)>) {
    let gpu = Rc::new(Gpu::new().expect("a GPU"));
    let (agents, horizon, features) = (6, 9, 7);
    let obs_size = features + picture.map_or(0, |(h, w, c)| h * w * c);
    let shapes = Shapes { obs_size, act_sizes: vec![5, 2, 3], hidden: 16, layers: 3, picture };
    let model = Model::new(gpu.clone(), shapes, 11);
    let mut actor = Actor::new(&model, agents, horizon, 5);
    let learner = Learner::new(&model, &actor.rollout, agents, &hyper());

    let mut rng = Rng::new(4);
    let (mut rewards, mut terminals) = (vec![0f32; agents * horizon], vec![0f32; agents * horizon]);
    // Two horizons, so that the second starts from state the first left
    for _ in 0..2 {
        for t in 0..horizon {
            let obs: Vec<f32> = (0..agents * obs_size).map(|_| rng.uniform() - 0.5).collect();
            let ended: Vec<f32> = (0..agents).map(|_| (rng.uniform() < 0.25) as u8 as f32).collect();
            for agent in 0..agents {
                rewards[agent * horizon + t] = rng.uniform() - 0.5;
                terminals[agent * horizon + t] = ended[agent];
            }
            actor.upload(&obs, &ended, t);
            actor.run(t, true);
            let actions = actor.actions();
            assert!(actions.chunks(3).all(|a| a[0] < 5 && a[1] < 2 && a[2] < 3), "an action its head has not got");
        }
    }
    actor.finish(&rewards, &terminals);

    learner.set_hyper(0.0, 0.001);
    learner.clear_stats();
    gpu.submit(&[learner.cut(0), &learner.forward_backward]);
    let stats = learner.read_stats();
    println!("old_kl {:e}, kl {:e}, clipfrac {}, importance {}", stats[4], stats[5], stats[6], stats[7]);
    assert!(stats[4].abs() < 1e-5 && stats[5].abs() < 1e-5, "the learner's log-probabilities are not the actor's");
    assert_eq!(stats[6], 0.0);
    assert!((stats[7] - 1.0).abs() < 1e-5);
}

#[test]
fn on_policy_from_features() {
    on_policy(None);
}

#[test]
fn on_policy_from_a_picture() {
    on_policy(Some((36, 36, 3)));
}

/// Agents that all see the same thing draw from one distribution, so how often each
/// action comes up has to be the probability the sampler says it drew it with.
#[test]
fn samples_follow_their_probabilities() {
    let gpu = Rc::new(Gpu::new().expect("a GPU"));
    let (agents, obs_size) = (20_000, 5);
    let shapes = Shapes { obs_size, act_sizes: vec![3, 2], hidden: 8, layers: 1, picture: None };
    let model = Model::new(gpu.clone(), shapes, 3);
    let mut actor = Actor::new(&model, agents, 1, 9);
    let obs: Vec<f32> = (0..agents).flat_map(|_| [2.0, -3.0, 1.5, 0.5, -2.5]).collect();
    actor.upload(&obs, &vec![0.0; agents], 0);
    actor.run(0, true);
    let actions = actor.actions().to_vec();
    let logprobs: Vec<f32> = gpu.read(&actor.rollout.logprobs, 0, agents);

    let mut seen: HashMap<(u32, u32), (usize, f32)> = HashMap::new();
    for (action, logprob) in actions.chunks(2).zip(&logprobs) {
        let entry = seen.entry((action[0], action[1])).or_insert((0, *logprob));
        entry.0 += 1;
        assert!((entry.1 - logprob).abs() < 1e-5, "one action, two log-probabilities");
    }
    let total: f64 = seen.values().map(|(_, logprob)| (*logprob as f64).exp()).sum();
    assert!(seen.len() == 6 && (total - 1.0).abs() < 1e-4, "probabilities of {} actions sum to {total}", seen.len());
    for (action, (count, logprob)) in &seen {
        let p = (*logprob as f64).exp();
        let frequency = *count as f64 / agents as f64;
        let sigma = (p * (1.0 - p) / agents as f64).sqrt();
        println!("{action:?}: drawn {frequency:.4} of the time, with probability {p:.4}");
        assert!((frequency - p).abs() < 5.0 * sigma + 1e-4, "{action:?} came up {frequency} of the time, not {p}");
    }
}
