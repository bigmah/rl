'''Golden numbers for tests/parity.rs, from mlx_pufferl.py: a policy, a minibatch, and what
the MLX trainer makes of them -- the loss statistics, every gradient, and the weights after
one and two steps of Muon. The Rust trainer has to land on the same numbers.

    uv run python tests/parity_dump.py        # rewrites tests/golden/
'''

import math
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mlx_pufferl as M

CASES = {
    # 5.0 as it is: GAE, advantages as they come
    'plain': dict(obs_size=11, act_sizes=[3, 2], hidden=8, layers=2, segments=3, horizon=5, picture=None,
        train=dict(gamma=0.995, gae_lambda=0.9, clip_coef=0.2, vf_coef=2.0, vf_clip_coef=0.2, max_grad_norm=1.5,
            momentum=0.95, weight_decay=0.0, vtrace=0, vtrace_rho_clip=1.0, vtrace_c_clip=1.0, norm_adv=0),
        learning_rate=0.015, ent_coef=0.001),
    # Everything a config can turn on, and shapes that are not multiples of anything
    'all': dict(obs_size=19, act_sizes=[17, 2, 2, 2], hidden=12, layers=3, segments=5, horizon=7, picture=None,
        train=dict(gamma=0.999, gae_lambda=0.95, clip_coef=0.1, vf_coef=0.5, vf_clip_coef=0.1, max_grad_norm=0.3,
            momentum=0.9, weight_decay=0.01, vtrace=1, vtrace_rho_clip=0.9, vtrace_c_clip=0.8, norm_adv=1),
        learning_rate=0.005, ent_coef=0.01),
    # A picture at the end of the observation, through the convolutions
    'picture': dict(obs_size=5 + 36 * 40 * 3, act_sizes=[4, 2], hidden=8, layers=1, segments=2, horizon=3,
        picture=(36, 40, 3),
        train=dict(gamma=0.995, gae_lambda=0.9, clip_coef=0.2, vf_coef=2.0, vf_clip_coef=0.2, max_grad_norm=1.5,
            momentum=0.95, weight_decay=0.0, vtrace=0, vtrace_rho_clip=1.0, vtrace_c_clip=1.0, norm_adv=0),
        learning_rate=0.015, ent_coef=0.001),
}


def flat(policy, tree):
    '''A tree shaped like the policy's parameters, flattened the way a checkpoint is'''
    names = [name for name, _ in M._checkpoint_order(policy)]
    values = dict(tree_flatten(tree))
    chunks = []
    for name in names:
        value = np.array(values[name], dtype=np.float32).ravel()
        chunks += [value, np.zeros(-value.size % 8, dtype=np.float32)]
    return np.concatenate(chunks)


def dump(name, case):
    out = os.path.join(ROOT, 'tests', 'golden', name)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    mx.random.seed(7)
    B, T, S = case['segments'], case['horizon'], case['obs_size']
    act_sizes, c = case['act_sizes'], case['train']

    policy = M.Policy(S, sum(act_sizes), case['hidden'], case['layers'], case['picture'])
    mx.eval(policy.parameters())
    M.save_weights(policy, os.path.join(out, 'weights.bin'))

    obs = rng.normal(size=(B, T, S)).astype(np.float32)
    if case['picture']:
        pixels = math.prod(case['picture'])
        # An env's pixels are 0 to 255. These are 0 to 1, so that a policy nobody has trained
        # is not certain of everything, which would leave the policy gradient nothing to say
        obs[..., S - pixels:] = rng.integers(0, 256, size=(B, T, pixels)).astype(np.float32) / 255
        # Black in places, where a ReLU sits exactly on its corner
        obs[0, 0, S - pixels:S - pixels // 2] = 0
    actions = np.stack([rng.integers(0, a, size=(B, T)) for a in act_sizes], axis=-1).astype(np.int32)
    rewards = np.clip(rng.normal(scale=0.5, size=(B, T)), -1, 1).astype(np.float32)
    terminals = (rng.random(size=(B, T)) < 0.2).astype(np.float32)
    state = rng.normal(scale=0.5, size=(case['layers'], B, case['hidden'])).astype(np.float32)

    # The rollout's log-probabilities and values: near what the policy says now, and far
    # enough that some ratios and some values are past their clips
    logits, values = policy(mx.array(obs), mx.array(state), mx.array(terminals))
    logprob, _ = M.logprob_entropy(logits.reshape(B * T, -1), mx.array(actions).reshape(B * T, -1), act_sizes)
    old_logprobs = (np.array(logprob).reshape(B, T) + rng.normal(scale=0.15, size=(B, T))).astype(np.float32)
    old_values = (np.array(values) + rng.normal(scale=0.3, size=(B, T))).astype(np.float32)

    batch = [mx.array(obs), mx.array(actions), mx.array(old_logprobs), mx.array(old_values), mx.array(rewards),
        mx.array(terminals), mx.array(state), mx.array(case['ent_coef'])]

    def loss_fn(obs, actions, old_logprobs, old_values, rewards, terminals, state, ent_coef):
        logits, values = policy(obs, state, terminals)
        return M.ppo_loss(logits, values, actions, old_logprobs, old_values, rewards, terminals, ent_coef, act_sizes, c)

    loss_and_grad = nn.value_and_grad(policy, loss_fn)
    optimizer = M.Muon(learning_rate=case['learning_rate'], momentum=c['momentum'], weight_decay=c['weight_decay'])
    optimizer.init(policy.trainable_parameters())

    # The convolutions are 76K weights however small the picture, so that case stops at one step
    for step in (1,) if case['picture'] else (1, 2):
        (_, stats), grads = loss_and_grad(*batch)
        mx.eval(stats, grads)
        np.array(stats, dtype=np.float32).tofile(os.path.join(out, f'stats{step}.bin'))
        flat(policy, grads).tofile(os.path.join(out, f'grads{step}.bin'))
        clipped, _ = optim.clip_grad_norm(grads, c['max_grad_norm'])
        optimizer.update(policy, clipped)
        mx.eval(policy.parameters(), optimizer.state)
        M.save_weights(policy, os.path.join(out, f'weights{step}.bin'))

    for key, value in dict(obs=obs, actions=actions.astype(np.uint32), old_logprobs=old_logprobs, old_values=old_values,
            rewards=rewards, terminals=terminals, state=state).items():
        value.tofile(os.path.join(out, f'{key}.bin'))

    with open(os.path.join(out, 'case.ini'), 'w') as f:
        f.write('# Written by tests/parity_dump.py\n[case]\n')
        f.write(f'obs_size = {S}\nact_sizes = {",".join(str(a) for a in act_sizes)},\n')
        f.write(f'hidden = {case["hidden"]}\nlayers = {case["layers"]}\nsegments = {B}\nhorizon = {T}\n')
        height, width, _ = case['picture'] or (0, 0, 0)
        f.write(f'picture_height = {height}\npicture_width = {width}\n')
        f.write(f'learning_rate = {case["learning_rate"]}\nent_coef = {case["ent_coef"]}\n')
        f.write('\n[train]\n' + ''.join(f'{key} = {value}\n' for key, value in c.items()))
    print(f'{name}: {flat(policy, policy.parameters()).size} weights, stats {np.array(stats).round(4).tolist()}')


if __name__ == '__main__':
    for name, case in CASES.items():
        dump(name, case)
