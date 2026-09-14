'''MLX training backend for PufferLib 4.0 on Apple Silicon GPUs.

Ports PufferLib's native CUDA trainer (src/pufferlib.cu, models.cu, muon.cu),
following the structure of its PyTorch reference (pufferlib/torch_pufferl.py)
but matching CUDA where the two differ: bias-free layers, one decoder matrix
for logits and value, sqrt(2) encoder init, the CUDA advantage kernel, and
entropy coefficient annealing. The policy is PufferLib's default (Linear
encoder -> MinGRU -> Linear decoder) optimized by Muon.

It plugs into PufferLib's own CLI, configs, dashboard, checkpointing and eval loop:

    python mlx_pufferl.py train platformer [puffer args]
    python mlx_pufferl.py eval platformer --load-model-path latest
'''

import ctypes
import glob
import math
import os
import time
from collections import defaultdict
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

import pufferlib.pufferl
from pufferlib import _C

LOSS_KEYS = ('policy_loss', 'value_loss', 'entropy', 'old_approx_kl', 'approx_kl', 'clipfrac', 'importance')
PERF_KEYS = ('rollout', 'eval_gpu', 'eval_env', 'train', 'train_misc', 'train_forward')

def _g(x):
    return mx.where(x >= 0, x + 0.5, mx.sigmoid(x))

def _log_g(x):
    return mx.where(x >= 0, mx.log(nn.relu(x) + 0.5), -nn.softplus(-x))

def _highway(x, out, proj):
    gate = mx.sigmoid(proj)
    return gate * out + (1.0 - gate) * x

class MinGRU(nn.Module):
    '''https://arxiv.org/abs/2410.01201v1, matching pufferlib.models.MinGRU'''
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = [nn.Linear(hidden_size, 3 * hidden_size, bias=False) for _ in range(num_layers)]

    def initial_state(self, batch_size):
        return mx.zeros((len(self.layers), batch_size, self.hidden_size))

    def step(self, h, state):
        '''One timestep for rollouts. h: (B, hidden), state: (layers, B, hidden)'''
        next_state = []
        for i, layer in enumerate(self.layers):
            hidden, gate, proj = mx.split(layer(h), 3, axis=-1)
            out = state[i] + mx.sigmoid(gate) * (_g(hidden) - state[i])
            h = _highway(h, out, proj)
            next_state.append(out)
        return h, mx.stack(next_state)

    def __call__(self, h):
        '''Whole segments from zero state via a parallel scan. h: (B, T, hidden)'''
        for layer in self.layers:
            hidden, gate, proj = mx.split(layer(h), 3, axis=-1)
            log_coeffs = -nn.softplus(gate)
            log_values = -nn.softplus(-gate) + _log_g(hidden)
            a_star = mx.cumsum(log_coeffs, axis=1)
            out = mx.exp(a_star + mx.logcumsumexp(log_values - a_star, axis=1))
            h = _highway(h, out, proj)
        return h

class Policy(nn.Module):
    '''PufferLib's default policy as models.cu builds it: bias-free encoder, MinGRU,
    and a single decoder matrix whose last row is the value head'''
    def __init__(self, obs_size, num_logits, hidden_size, num_layers):
        super().__init__()
        # MLX's Linear init is U(+-1/sqrt(fan_in)), same as CUDA at gain 1; the encoder uses gain sqrt(2)
        self.encoder = nn.Linear(obs_size, hidden_size, bias=False)
        self.encoder.weight = self.encoder.weight * math.sqrt(2)
        self.network = MinGRU(hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, num_logits + 1, bias=False)

    def step(self, obs, state):
        h, state = self.network.step(self.encoder(obs.astype(mx.float32)), state)
        out = self.decoder(h)
        return out[..., :-1], out[..., -1], state

    def __call__(self, obs):
        out = self.decoder(self.network(self.encoder(obs.astype(mx.float32))))
        return out[..., :-1], out[..., -1]

def _head_log_probs(logits, act_sizes):
    '''Normalized log-probs for each discrete action head'''
    splits = np.cumsum(act_sizes)[:-1].tolist()
    heads = mx.split(logits, splits, axis=-1) if splits else [logits]
    return [head - mx.logsumexp(head, axis=-1, keepdims=True) for head in heads]

def sample_actions(logits, act_sizes):
    actions, logprob = [], 0.0
    for logp in _head_log_probs(logits, act_sizes):
        action = mx.random.categorical(logp)
        logprob = logprob + mx.take_along_axis(logp, action[:, None], axis=-1).squeeze(-1)
        actions.append(action)
    return mx.stack(actions, axis=-1), logprob

def logprob_entropy(logits, actions, act_sizes):
    logprob, entropy = 0.0, 0.0
    for i, logp in enumerate(_head_log_probs(logits, act_sizes)):
        logprob = logprob + mx.take_along_axis(logp, actions[:, i:i+1], axis=-1).squeeze(-1)
        entropy = entropy - (mx.exp(logp) * logp).sum(axis=-1)
    return logprob, entropy

def puff_advantage(values, rewards, terminals, ratio, gamma, gae_lambda, rho_clip, c_clip):
    '''PufferLib's CUDA advantage kernel: GAE with V-trace clipped importance weights
    scaling the whole TD error. Arrays are (horizon, agents); the last step gets 0.'''
    nonterminal = 1.0 - terminals[1:]
    delta = np.minimum(ratio[:-1], rho_clip) * (rewards[1:] + gamma * values[1:] * nonterminal - values[:-1])
    decay = gamma * gae_lambda * np.minimum(ratio[:-1], c_clip) * nonterminal
    advantages = np.zeros_like(values)
    for t in range(len(delta) - 1, -1, -1):
        advantages[t] = delta[t] + decay[t] * advantages[t + 1]
    return advantages

def cosine_annealing(base, min_ratio, progress):
    minimum = base * min_ratio
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * progress))

NS_COEFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]

def zeropower_via_newtonschulz5(G, eps=1e-7):
    transpose = G.shape[0] > G.shape[1]
    x = G.T if transpose else G
    x = x / mx.maximum(mx.linalg.norm(G), eps)
    eye = mx.eye(x.shape[0])
    for a, b, c in NS_COEFS:
        s = x @ x.T
        x = ((c * s + b * eye) @ s + a * eye) @ x
    return x.T if transpose else x

class Muon(optim.Optimizer):
    '''pufferlib.muon.Muon: Nesterov momentum, orthogonalized updates for matrices.

    An orthogonalized update is the same size however small the gradient was, so a
    reward that says nothing for millions of steps walks the weights in random
    directions until they blow up: the sm64 door-or-nothing run did at 3.4M steps,
    value loss 0 -> 567,829 and entropy to zero. Weight decay holds them to a size.
    PufferLib passes 0, so that is the default.'''
    def __init__(self, learning_rate, momentum=0.9, weight_decay=0.0):
        super().__init__()
        self._maybe_schedule('learning_rate', learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay

    def init_single(self, parameter, state):
        state['momentum_buffer'] = mx.zeros_like(parameter)

    def apply_single(self, gradient, parameter, state):
        buf = self.momentum * state['momentum_buffer'] + gradient
        state['momentum_buffer'] = buf
        update = gradient + self.momentum * buf
        if update.ndim >= 2:
            update = zeropower_via_newtonschulz5(update.reshape(update.shape[0], -1))
            update = update * max(1, update.shape[0] / update.shape[1]) ** 0.5
        lr = self.learning_rate.astype(gradient.dtype)
        return parameter * (1 - lr * self.weight_decay) - lr * update.reshape(parameter.shape)

def _buffer_view(ptr, shape, dtype):
    '''Zero-copy numpy view of a buffer owned by the C vecenv'''
    count = int(np.prod(shape))
    ctype = ctypes.c_uint8 if dtype == np.uint8 else ctypes.c_float
    return np.ctypeslib.as_array((ctype * count).from_address(ptr)).reshape(shape)

def _most_trained_checkpoint(paths):
    '''Puffer names a checkpoint after the agent step it was written at, so the most
    trained one is the highest number and not the newest file. Picking by mtime loads
    whatever ran last, which after a short run is a policy that knows nothing.'''
    def key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        return (int(name) if name.isdigit() else -1, os.path.getctime(path))
    return max(paths, key=key)

class PuffeRL:
    def __init__(self, args, vec, policy):
        config = args['train']
        self.args = args
        self.config = config
        self._vec = vec
        self.policy = policy
        self.act_sizes = list(vec.act_sizes)

        self.total_agents = N = vec.total_agents
        self.horizon = H = config['horizon']
        obs_dtype = np.uint8 if vec.obs_dtype == 'ByteTensor' else np.float32
        self.vec_obs = _buffer_view(vec.obs_ptr, (N, vec.obs_size), obs_dtype)
        self.vec_rewards = _buffer_view(vec.rewards_ptr, (N,), np.float32)
        self.vec_terminals = _buffer_view(vec.terminals_ptr, (N,), np.float32)
        vec.reset()

        self.state = policy.network.initial_state(N)
        self.rewards = np.zeros((H, N), dtype=np.float32)
        self.terminals = np.zeros((H, N), dtype=np.float32)
        self.batch_size = N * H
        self.minibatch_segments = config['minibatch_size'] // H
        self.total_epochs = max(1, config['total_timesteps'] // self.batch_size)
        self.rng = np.random.default_rng(config['seed'])

        self.optimizer = Muon(learning_rate=config['learning_rate'], momentum=config['beta1'],
            weight_decay=config.get('weight_decay', 0.0))
        # With target_entropy set, ent_coef is only where the coefficient starts: see train()
        self.ent_coef = config['ent_coef']
        self.log_ent_coef = math.log(config['ent_coef'])
        self.optimizer.init(policy.trainable_parameters())
        self._act = self._compile_act()
        self._train_step = self._compile_train_step()

        self.model_size = sum(v.size for _, v in tree_flatten(policy.trainable_parameters()))
        self.memory_gb = mx.device_info()['memory_size'] / 1e9
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.losses = {}
        self.env_sums = defaultdict(float)
        self.env_episodes = 0.0
        self.perf = defaultdict(float)

    def _compile_act(self):
        policy, act_sizes = self.policy, self.act_sizes

        @partial(mx.compile, inputs=[policy.state, mx.random.state], outputs=[mx.random.state])
        def act(obs, state):
            logits, value, state = policy.step(obs, state)
            action, logprob = sample_actions(logits, act_sizes)
            return action, logprob, value, state

        return act

    def _compile_train_step(self):
        policy, optimizer, act_sizes, c = self.policy, self.optimizer, self.act_sizes, self.config

        def loss_fn(obs, actions, old_logprobs, values, advantages, prio, ent_coef):
            segments, horizon = values.shape
            logits, newvalue = policy(obs)
            newlogprob, entropy = logprob_entropy(logits.reshape(segments * horizon, -1),
                actions.reshape(segments * horizon, -1), act_sizes)
            logratio = newlogprob.reshape(segments, horizon) - old_logprobs
            ratio = mx.exp(logratio)

            adv = prio * (advantages - advantages.mean()) / (mx.std(advantages, ddof=1) + 1e-8)
            clipped_ratio = mx.clip(ratio, 1 - c['clip_coef'], 1 + c['clip_coef'])
            pg_loss = mx.maximum(-adv * ratio, -adv * clipped_ratio).mean()

            returns = advantages + values
            v_clipped = values + mx.clip(newvalue - values, -c['vf_clip_coef'], c['vf_clip_coef'])
            v_loss = 0.5 * mx.maximum((newvalue - returns) ** 2, (v_clipped - returns) ** 2).mean()

            entropy = entropy.mean()
            loss = pg_loss + c['vf_coef'] * v_loss - ent_coef * entropy
            stats = mx.stack([pg_loss, v_loss, entropy, (-logratio).mean(),
                ((ratio - 1) - logratio).mean(),
                (mx.abs(ratio - 1) > c['clip_coef']).astype(mx.float32).mean(), ratio.mean()])
            return loss, stats, newvalue, ratio

        loss_and_grad = nn.value_and_grad(policy, loss_fn)
        state = [policy.state, optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def train_step(*batch):
            (_, stats, newvalue, ratio), grads = loss_and_grad(*batch)
            grads, _ = optim.clip_grad_norm(grads, c['max_grad_norm'])
            optimizer.update(policy, grads)
            return stats, newvalue, ratio

        return train_step

    def num_params(self):
        return self.model_size

    def rollouts(self):
        N, H = self.total_agents, self.horizon
        start = time.perf_counter()
        if self.args.get('reset_state', True):
            self.state = mx.zeros_like(self.state)

        rewards = np.zeros(N, dtype=np.float32)
        terminals = np.zeros(N, dtype=np.float32)
        observations, actions, logprobs, values = [], [], [], []
        for t in range(H):
            t0 = time.perf_counter()
            obs = mx.array(self.vec_obs)
            action, logprob, value, self.state = self._act(obs, self.state)
            env_actions = action.astype(mx.float32)
            mx.eval(env_actions, logprob, value, self.state)
            t1 = time.perf_counter()

            self.rewards[t] = rewards
            self.terminals[t] = terminals
            observations.append(obs)
            actions.append(action)
            logprobs.append(logprob)
            values.append(value)

            step_actions = np.array(env_actions)  # keep alive while C reads the pointer
            self._vec.cpu_step(step_actions.ctypes.data)
            rewards, terminals = self.vec_rewards, self.vec_terminals
            self.perf['eval_gpu'] += t1 - t0
            self.perf['eval_env'] += time.perf_counter() - t1

        # (agents, horizon, ...) for minibatch indexing; values stay (horizon, agents) for advantages
        self.observations = mx.stack(observations, axis=1)
        self.actions = mx.stack(actions, axis=1)
        self.logprobs = mx.stack(logprobs, axis=1)
        self.values = mx.stack(values, axis=0)
        mx.eval(self.observations, self.actions, self.logprobs, self.values)

        self.global_step += N * H
        self._gather_env_logs()
        self.perf['rollout'] += time.perf_counter() - start

    def _gather_env_logs(self):
        '''The vecenv reports an average over the episodes that ended since it was last read, then
        forgets them, and the CPU binding only exposes that resetting read. So keep the episode-weighted
        sums here: log() reports and clears them like static_vec_log, and eval_log() reports and keeps
        counting like static_vec_eval_log. Without the counting, env/n never passes eval_episodes and
        eval runs every one of its train_epochs // 2 epochs'''
        logs = self._vec.log()
        episodes = logs.get('n', 0.0)
        if not episodes:
            return
        for key, value in logs.items():
            if key != 'n':
                self.env_sums[key] += value * episodes
        self.env_episodes += episodes

    def train(self):
        config = self.config
        N = self.total_agents
        start = time.perf_counter()

        progress = min(self.epoch / self.total_epochs, 1.0)
        prio_alpha, beta0 = config['prio_alpha'], config['prio_beta0']
        anneal_beta = beta0 + (1 - beta0) * prio_alpha * progress
        if config['anneal_lr']:
            self.optimizer.learning_rate = cosine_annealing(config['learning_rate'], config['min_lr_ratio'], progress)
        ent_coef = config['ent_coef']
        target_entropy = config.get('target_entropy', 0)
        if target_entropy:
            ent_coef = self.ent_coef
        elif config['anneal_ent_coef']:
            ent_coef = cosine_annealing(ent_coef, config['min_ent_coef_ratio'], progress)

        # Advantages and prioritized sampling run on CPU. Values and ratios are
        # refreshed from each minibatch, as in the CUDA trainer
        values = np.array(self.values)
        rewards = np.clip(self.rewards, -1, 1)
        ratio = np.ones_like(values)

        losses = np.zeros(len(LOSS_KEYS))
        step_time = 0.0
        num_minibatches = int(config['replay_ratio'] * self.batch_size / config['minibatch_size'])
        for _ in range(num_minibatches):
            advantages = puff_advantage(values, rewards, self.terminals, ratio, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            prio_weights = np.abs(advantages).sum(axis=0).astype(np.float64) ** prio_alpha
            prio_weights = np.nan_to_num(prio_weights, nan=0.0, posinf=0.0, neginf=0.0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = self.rng.choice(N, size=self.minibatch_segments, p=prio_probs / prio_probs.sum())
            mb_prio = ((N * prio_probs[idx, None]) ** -anneal_beta).astype(np.float32)

            t0 = time.perf_counter()
            mb_idx = mx.array(idx)
            stats, newvalue, mb_ratio = self._train_step(
                self.observations[mb_idx], self.actions[mb_idx], self.logprobs[mb_idx],
                mx.array(values.T[idx]), mx.array(advantages.T[idx]), mx.array(mb_prio), mx.array(ent_coef))
            mx.eval(stats, newvalue, mb_ratio)
            step_time += time.perf_counter() - t0

            losses += np.array(stats)
            values.T[idx] = np.array(newvalue)
            ratio.T[idx] = np.array(mb_ratio)

        y_pred = values.flatten()
        y_true = advantages.flatten() + y_pred
        var_y = y_true.var()
        self.losses = dict(zip(LOSS_KEYS, (losses / num_minibatches).tolist()))
        self.losses['explained_variance'] = float('nan') if var_y == 0 else float(1 - (y_true - y_pred).var() / var_y)
        self.losses['ent_coef'] = float(ent_coef)
        if target_entropy:
            # Steer the coefficient toward the entropy asked for: up while the policy is
            # more certain than that, down while it is less, by a factor of e every
            # 1 / (ent_coef_rate * gap) epochs, kept between 1e-4 and 0.5.
            gap = target_entropy - self.losses['entropy']
            self.log_ent_coef = min(max(self.log_ent_coef + config.get('ent_coef_rate', 0.01) * gap,
                math.log(1e-4)), math.log(0.5))
            self.ent_coef = math.exp(self.log_ent_coef)
        self.epoch += 1

        elapsed = time.perf_counter() - start
        self.perf['train'] += elapsed
        self.perf['train_forward'] += step_time
        self.perf['train_misc'] += elapsed - step_time

    def log(self, reset_env=True):
        now = time.time()
        steps = self.global_step - self.last_log_step
        perf, self.perf = self.perf, defaultdict(float)
        env = {}
        if self.env_episodes:
            env = {k: v / self.env_episodes for k, v in self.env_sums.items()}
            env['n'] = self.env_episodes
        if reset_env:
            self.env_sums, self.env_episodes = defaultdict(float), 0.0
        logs = {
            'SPS': steps / (now - self.last_log_time) if steps else 0,
            'agent_steps': self.global_step,
            'uptime': now - self.start_time,
            'epoch': self.epoch,
            'env': env,
            'loss': dict(self.losses),
            'perf': {k: perf[k] for k in PERF_KEYS},
            'util': {'vram_used_gb': mx.get_active_memory() / 1e9, 'vram_total_gb': self.memory_gb},
        }
        self.last_log_time = now
        self.last_log_step = self.global_step
        return logs

    def eval_log(self):
        return self.log(reset_env=False)

    def save_weights(self, path):
        # Write through a file handle so numpy keeps puffer's .bin filename
        with open(path, 'wb') as f:
            np.savez(f, **{k: np.array(v) for k, v in tree_flatten(self.policy.parameters())})

    def load_weights(self, path):
        weights = np.load(path)
        self.policy.load_weights([(k, mx.array(weights[k])) for k in weights.files])

    def render(self, env_id=0):
        self._vec.render(env_id)

    def close(self):
        self._vec.close()

    @classmethod
    def create_pufferl(cls, args):
        '''Matches the _C.create_pufferl(args) interface'''
        for key, default in (('network', 'MinGRU'), ('encoder', 'DefaultEncoder'), ('decoder', 'DefaultDecoder')):
            if args['torch'][key] != default:
                raise ValueError(f'mlx_pufferl only implements the default {key} {default}, not {args["torch"][key]}')

        args['vec']['num_buffers'] = 1
        vec = _C.create_vec(args, 0)
        if sum(vec.act_sizes) == len(vec.act_sizes):
            raise ValueError('mlx_pufferl only implements discrete actions')

        mx.random.seed(args['train']['seed'])
        policy_config = args['policy']
        policy = Policy(vec.obs_size, sum(vec.act_sizes),
            int(policy_config['hidden_size']), int(policy_config['num_layers']))
        mx.eval(policy.parameters())
        pufferl = cls(args, vec, policy)

        load_path = args.get('load_model_path')
        if load_path == 'latest':
            pattern = os.path.join(args['checkpoint_dir'], args['env_name'], '**', '*.bin')
            candidates = glob.glob(pattern, recursive=True)
            if not candidates:
                raise FileNotFoundError(f'No .bin checkpoints found in {args["checkpoint_dir"]}/{args["env_name"]}/')
            load_path = _most_trained_checkpoint(candidates)
            print(f'Loading {load_path}')
        if load_path is not None:
            pufferl.load_weights(load_path)
        return pufferl

def _resolve_backend(args):
    assert _C.env_name == args['env_name'], f'build.sh was run for {_C.env_name}, not {args["env_name"]}'
    return PuffeRL

if __name__ == '__main__':
    # Swap this backend in where PufferLib picks CUDA or --slowly torch, and reuse
    # its CLI, configs, dashboard, checkpointing and eval loop as-is
    pufferlib.pufferl._resolve_backend = _resolve_backend
    pufferlib.pufferl.main()
