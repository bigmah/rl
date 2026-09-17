'''PufferLib 5.0's trainer on MLX, for Apple Silicon GPUs.

PufferLib 5.0 trains with one CUDA program (src/pufferl.cu, src/algo.cu), which
a Mac can't run. This is that trainer in MLX: the same policy (bias-free Linear
encoder -> MinGRU -> one decoder matrix for logits and value), the same PPO
(advantages from each minibatch's own values, no prioritized replay, recurrent
state carried across horizons), Muon, the same config files and CLI overrides,
checkpoints PufferLib's own eval can read, and its dashboard. An env whose
config names a picture_width and picture_height gets three convolutions in front
of the encoder for the picture at the end of its observation.

Envs are PufferLib 5.0 C envs, which build.sh compiles into a vecenv library
(vecenv.c) that this drives through ctypes:

    python mlx_pufferl.py train platformer [--section.key=value ...]
    python mlx_pufferl.py eval platformer [latest | MODEL.bin] [--headless] [--section.key=value ...]
'''

import ctypes
import glob
import math
import os
import resource
import shutil
import signal
import sys
import time
from collections import defaultdict
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

ROOT = os.path.dirname(os.path.abspath(__file__))
PUFFER_CONFIG = os.path.join(ROOT, 'vendor', 'PufferLib', 'config', 'default.ini')

# The names src/pufferl.cu logs them under, in the order the train step returns them
LOSS_KEYS = ('loss/policy', 'loss/value', 'loss/entropy', 'loss/total',
    'loss/old_kl', 'loss/kl', 'loss/clipfrac', 'importance')
PERF_KEYS = ('rollout', 'eval_model', 'eval_env', 'eval_copy', 'train_misc', 'train_model')

# --- Config: src/ini.h's reading of default.ini, the env's ini, and the command line

def _strip_comment(line):
    quote, prev = None, ''
    for i, ch in enumerate(line):
        if ch in '\'"' and prev != '\\':
            quote = None if quote == ch else quote or ch
        elif ch in '#;' and not quote:
            return line[:i]
        prev = ch
    return line

def _load_ini(path, ini):
    section = None
    with open(path) as f:
        for n, line in enumerate(f, 1):
            s = _strip_comment(line.rstrip('\n')).strip()
            if not s:
                continue
            if s[0] == '[' and s[-1] == ']' and len(s) >= 3:
                section = ini.setdefault(s[1:-1].strip(), {})
                continue
            if section is None or '=' not in s:
                sys.exit(f'{path}:{n}: expected ' + ('section before key=value' if section is None else 'key=value'))
            key, value = (part.strip() for part in s.split('=', 1))
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '\'"':
                value = value[1:-1]
            section[key] = value

def _parse(raw):
    '''A value the way ini.h reads one: true and false, numbers with underscores, and lists'''
    if raw.lower() in ('true', 'false'):
        return int(raw.lower() == 'true')
    try:
        if ',' in raw:
            return [float(part) for part in raw.split(',')]
        value = float(''.join(ch for ch in raw if ch != '_' and not ch.isspace()))
    except ValueError:
        return raw
    return int(value) if value.is_integer() and abs(value) < 2**62 else value

def parsed(ini):
    return {section: {key: _parse(raw) for key, raw in keys.items()} for section, keys in ini.items()}

def put(ini, full_key, raw):
    '''Only keys a config file already has can be set, as in ini.h'''
    section, _, key = full_key.rpartition('.')
    if key not in ini.get(section, {}):
        sys.exit(f'missing key [{section}] {key}')
    ini[section][key] = str(raw)

def load_config(env_name, argv):
    '''default.ini, then envs/ENV/ENV.ini, then --section.key=value (or --section.key value) from the
    command line. A key with no section is in [base], and dashes in a key are underscores'''
    env_config = os.path.join(ROOT, 'envs', env_name, f'{env_name}.ini')
    if not os.path.exists(env_config):
        sys.exit(f'no config for {env_name}: {env_config} not found')
    ini = {}
    _load_ini(PUFFER_CONFIG, ini)
    _load_ini(env_config, ini)
    put(ini, 'base.env_name', env_name)
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith('--'):
            sys.exit(f"unexpected argument '{arg}'")
        key, value = arg.lstrip('-'), 'true'
        if '=' in key:
            key, value = key.split('=', 1)
        elif i + 1 < len(argv) and not argv[i + 1].startswith('--'):
            i += 1
            value = argv[i]
        key = key.replace('-', '_')
        put(ini, key if '.' in key else f'base.{key}', value)
        i += 1
    return ini

def write_ini(f, ini):
    for section, keys in ini.items():
        f.write(f'\n[{section}]\n')
        for key, raw in keys.items():
            f.write(f'{key} = {raw}\n')

# --- The env, through vecenv.c

class VecEnv:
    '''build/vecenv_ENV.dylib. The buffers are numpy views of the library's own memory'''
    KEY_BYTES = 128  # PUF_DICT_MAX_KEY

    def __init__(self, env_name, env_config, total_agents, num_threads):
        path = os.path.join(ROOT, 'build', f'vecenv_{env_name}.dylib')
        if not os.path.exists(path):
            sys.exit(f'{path} not found: run ./build.sh {env_name}')
        lib = self.lib = ctypes.CDLL(path)
        c_int, c_void_p = ctypes.c_int, ctypes.c_void_p
        lib.vec_create.restype = c_void_p
        lib.vec_create.argtypes = [c_int, c_int, c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
        for name in ('vec_observations', 'vec_actions', 'vec_rewards', 'vec_terminals'):
            getattr(lib, name).restype = c_void_p
            getattr(lib, name).argtypes = [c_void_p]
        for name in ('vec_reset', 'vec_step', 'vec_close'):
            getattr(lib, name).argtypes = [c_void_p]
        lib.vec_render.argtypes = [c_void_p, c_int]
        lib.vec_log.argtypes = [c_void_p, c_int, c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
        lib.vec_act_sizes.restype = ctypes.POINTER(c_int)

        keys = [key.encode() for key in env_config]
        values = [raw.encode() for raw in env_config.values()]
        self._vec = lib.vec_create(total_agents, num_threads, len(keys),
            (ctypes.c_char_p * len(keys))(*keys), (ctypes.c_char_p * len(values))(*values))
        self.total_agents = N = total_agents
        self.obs_size = lib.vec_obs_size()
        self.num_atns = lib.vec_num_atns()
        self.act_sizes = [lib.vec_act_sizes()[i] for i in range(self.num_atns)]
        obs_type, self.obs_dtype = (ctypes.c_uint8, np.uint8) if lib.vec_obs_bytes() == 1 else (ctypes.c_float, np.float32)
        view = lambda ptr, ctype, shape: np.ctypeslib.as_array((ctype * math.prod(shape)).from_address(ptr)).reshape(shape)
        self.observations = view(lib.vec_observations(self._vec), obs_type, (N, self.obs_size))
        self.actions = view(lib.vec_actions(self._vec), ctypes.c_float, (N, self.num_atns))
        self.rewards = view(lib.vec_rewards(self._vec), ctypes.c_float, (N,))
        self.terminals = view(lib.vec_terminals(self._vec), ctypes.c_float, (N,))

    def reset(self):
        self.lib.vec_reset(self._vec)

    def step(self):
        self.lib.vec_step(self._vec)

    def log(self, clear):
        '''Env stats averaged over the episodes since the last clear, and n, how many'''
        keys = ctypes.create_string_buffer(64 * self.KEY_BYTES)
        values = (ctypes.c_double * 64)()
        count = self.lib.vec_log(self._vec, int(clear), 64, keys, values)
        name = lambda i: keys.raw[i * self.KEY_BYTES:(i + 1) * self.KEY_BYTES].split(b'\0', 1)[0].decode()
        return {name(i): values[i] for i in range(count)}

    def render(self, env_id=0):
        self.lib.vec_render(self._vec, env_id)

    def window_closed(self):
        return bool(self.lib.vec_window_closed())

    def close(self):
        self.lib.vec_close(self._vec)

# --- The policy (src/algo.cu)

def _g(x):
    return mx.where(x >= 0, x + 0.5, mx.sigmoid(x))

def linear_scan(a, b):
    '''h[t] = a[t] * h[t-1] + b[t] along axis 1, as every prefix of it: returns A and B with
    h[t] = A[t] * h[-1] + B[t]. Pairs up neighbours and recurses, so it takes log2(T) rounds.

    Pairs are split by reshaping rather than by step slicing: in MLX 0.32.2 the gradient of
    x[1::2] and x[0::2] is wrong when the slices are one long.'''
    T = a.shape[1]
    if T == 1:
        return a, b
    if T % 2:
        a = mx.concatenate([a, mx.ones_like(a[:, :1])], axis=1)
        b = mx.concatenate([b, mx.zeros_like(b[:, :1])], axis=1)
    pairs = (a.shape[0], a.shape[1] // 2, 2) + a.shape[2:]
    a, b = a.reshape(pairs), b.reshape(pairs)
    a0, a1, b0, b1 = a[:, :, 0], a[:, :, 1], b[:, :, 0], b[:, :, 1]
    A1, B1 = linear_scan(a1 * a0, a1 * b0 + b1)
    A0 = mx.concatenate([a0[:, :1], a0[:, 1:] * A1[:, :-1]], axis=1)
    B0 = mx.concatenate([b0[:, :1], a0[:, 1:] * B1[:, :-1] + b0[:, 1:]], axis=1)
    shape = (pairs[0], 2 * pairs[1]) + pairs[3:]
    return mx.stack([A0, A1], axis=2).reshape(shape)[:, :T], mx.stack([B0, B1], axis=2).reshape(shape)[:, :T]

_SIGMOID = '''
inline float puf_sigmoid(float x) {
    float z = exp(-fabs(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}
'''

# algo.cu's mingru_scan_forward and mingru_scan_backward as Metal kernels: a thread for each
# unit of each segment, stepping through time, and clearing the state where a terminal says
# a new episode starts. scan_h keeps the state each step started from, for the backward pass.
_SCAN_FORWARD = mx.fast.metal_kernel(
    name='mingru_scan_forward',
    input_names=['combined', 'x', 'state', 'terminals'],
    output_names=['out', 'scan_h'],
    header=_SIGMOID,
    source='''
    uint T = combined_shape[1], H = x_shape[2];
    uint b = thread_position_in_grid.x / H, h = thread_position_in_grid.x % H;
    float state_h = state[b * H + h];
    for (uint t = 0; t < T; t++) {
        uint i = (b * T + t) * H + h;
        uint c = (b * T + t) * 3 * H + h;
        if (terminals[b * T + t] != 0.0f) {
            state_h = 0.0f;
        }
        scan_h[i] = state_h;
        float hidden = combined[c], gate = combined[c + H], proj = combined[c + 2 * H];
        float z = puf_sigmoid(gate);
        float h_tilde = hidden >= 0.0f ? hidden + 0.5f : puf_sigmoid(hidden);
        state_h = state_h + z * (h_tilde - state_h);
        float s = puf_sigmoid(proj);
        out[i] = s * state_h + (1.0f - s) * x[i];
    }
''')

_SCAN_BACKWARD = mx.fast.metal_kernel(
    name='mingru_scan_backward',
    input_names=['combined', 'x', 'scan_h', 'terminals', 'grad_out'],
    output_names=['grad_combined', 'grad_x'],
    header=_SIGMOID,
    source='''
    uint T = combined_shape[1], H = x_shape[2];
    uint b = thread_position_in_grid.x / H, h = thread_position_in_grid.x % H;
    float dh = 0.0f;
    for (uint k = T; k > 0; k--) {
        uint t = k - 1;
        uint i = (b * T + t) * H + h;
        uint c = (b * T + t) * 3 * H + h;
        float h_prev = scan_h[i];
        float hidden = combined[c], gate = combined[c + H], proj = combined[c + 2 * H];
        float z = puf_sigmoid(gate);
        float h_tilde = hidden >= 0.0f ? hidden + 0.5f : puf_sigmoid(hidden);
        float h_t = h_prev + z * (h_tilde - h_prev);
        float s = puf_sigmoid(proj);
        float g = grad_out[i];
        grad_x[i] = g * (1.0f - s);
        grad_combined[c + 2 * H] = g * (h_t - x[i]) * s * (1.0f - s);
        float dh_total = dh + g * s;
        float d_h_tilde = dh_total * z;
        grad_combined[c + H] = dh_total * (h_tilde - h_prev) * z * (1.0f - z);
        grad_combined[c] = hidden >= 0.0f ? d_h_tilde : d_h_tilde * h_tilde * (1.0f - h_tilde);
        // The state before a terminal never reached this step
        dh = terminals[b * T + t] != 0.0f ? 0.0f : dh_total * (1.0f - z);
    }
''')

def _scan_threads(x):
    n = x.shape[0] * x.shape[2]
    return dict(grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1))

@mx.custom_function
def mingru_scan(combined, x, state, terminals):
    '''One MinGRU layer over segments. combined: (B, T, 3 * hidden) from the layer's matrix;
    x: (B, T, hidden), its input; state: (B, hidden) before the first step; terminals: (B, T).
    Returns the layer's output and the state each step started from'''
    return _SCAN_FORWARD(inputs=[combined, x, state, terminals], output_shapes=[x.shape, x.shape],
        output_dtypes=[mx.float32, mx.float32], **_scan_threads(x))

@mingru_scan.vjp
def _mingru_scan_vjp(primals, cotangents, outputs):
    combined, x, state, terminals = primals
    grad_combined, grad_x = _SCAN_BACKWARD(inputs=[combined, x, outputs[1], terminals, cotangents[0]],
        output_shapes=[combined.shape, x.shape], output_dtypes=[mx.float32, mx.float32], **_scan_threads(x))
    return grad_combined, grad_x, mx.zeros_like(state), mx.zeros_like(terminals)

class MinGRU(nn.Module):
    '''https://arxiv.org/abs/2410.01201v1 as algo.cu has it: per layer, one bias-free matrix to
    hidden, gate and highway, h = h + z * (g(hidden) - h), and out = s * h + (1 - s) * x'''
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = [nn.Linear(hidden_size, 3 * hidden_size, bias=False) for _ in range(num_layers)]

    def initial_state(self, batch_size):
        return mx.zeros((len(self.layers), batch_size, self.hidden_size))

    def step(self, x, state):
        '''One timestep for rollouts. x: (B, hidden), state: (layers, B, hidden)'''
        next_state = []
        for i, layer in enumerate(self.layers):
            hidden, gate, proj = mx.split(layer(x), 3, axis=-1)
            h = state[i] + mx.sigmoid(gate) * (_g(hidden) - state[i])
            s = mx.sigmoid(proj)
            x = s * h + (1.0 - s) * x
            next_state.append(h)
        return x, mx.stack(next_state)

    def __call__(self, x, state, terminals):
        '''Whole segments. x: (B, T, hidden); state: (layers, B, hidden) before the first step;
        terminals: (B, T), where a 1 clears the state before that step, as the rollout did'''
        for i, layer in enumerate(self.layers):
            x, _ = mingru_scan(layer(x), x, state[i], terminals)
        return x

class PictureEncoder(nn.Module):
    '''For an observation that ends in a picture: the picture through the Nature DQN's
    three convolutions, and what comes before it alongside, into the one Linear encoder.

    The picture is the observation's last height * width * channels numbers, rows from
    the top, channels last, which is the layout the env writes and the one MLX's
    convolutions take. Bias-free with sqrt(2) gain, like the default encoder.'''
    def __init__(self, obs_size, picture_shape, hidden_size):
        super().__init__()
        height, width, channels = picture_shape
        self.picture_shape = picture_shape
        self.features = obs_size - height * width * channels
        self.convs = [
            nn.Conv2d(channels, 32, 8, stride=4, bias=False),
            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.Conv2d(64, 64, 3, stride=1, bias=False),
        ]
        for conv in self.convs:
            conv.weight = conv.weight * math.sqrt(2)
        for kernel, stride in ((8, 4), (4, 2), (3, 1)):
            height, width = (height - kernel) // stride + 1, (width - kernel) // stride + 1
        if height < 1 or width < 1:
            raise ValueError(f'a {picture_shape[1]}x{picture_shape[0]} picture is too small for the convolutions')
        self.linear = nn.Linear(self.features + height * width * 64, hidden_size, bias=False)
        self.linear.weight = self.linear.weight * math.sqrt(2)

    def __call__(self, obs):
        leading = obs.shape[:-1]
        x = obs[..., self.features:].reshape(-1, *self.picture_shape)
        for conv in self.convs:
            x = nn.relu(conv(x))
        x = x.reshape(*leading, -1)
        return self.linear(mx.concatenate([obs[..., :self.features], x], axis=-1))

class Policy(nn.Module):
    '''PufferLib's default policy as algo.cu builds it: bias-free encoder, MinGRU, and a single
    decoder matrix whose last row is the value head. With a picture_shape, the encoder is a
    PictureEncoder instead'''
    def __init__(self, obs_size, num_logits, hidden_size, num_layers, picture_shape=None):
        super().__init__()
        if picture_shape is not None:
            self.encoder = PictureEncoder(obs_size, picture_shape, hidden_size)
        else:
            # MLX's Linear init is U(+-1/sqrt(fan_in)), puf_kaiming_init at gain 1; the encoder uses gain sqrt(2)
            self.encoder = nn.Linear(obs_size, hidden_size, bias=False)
            self.encoder.weight = self.encoder.weight * math.sqrt(2)
        self.network = MinGRU(hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, num_logits + 1, bias=False)

    def step(self, obs, state):
        h, state = self.network.step(self.encoder(obs.astype(mx.float32)), state)
        out = self.decoder(h)
        return out[..., :-1], out[..., -1], state

    def __call__(self, obs, state, terminals):
        out = self.decoder(self.network(self.encoder(obs.astype(mx.float32)), state, terminals))
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

def puff_advantage(values, rewards, terminals, importance, gamma, gae_lambda, rho_clip, c_clip):
    '''algo.cu's puff_advantage over segments (B, T): GAE, with V-trace's clipped importance
    weights on the TD error and on the trace. The last step's advantage is 0.'''
    nonterminal = 1.0 - terminals[:, 1:]
    rho = mx.minimum(importance[:, :-1], rho_clip)
    c = mx.minimum(importance[:, :-1], c_clip)
    delta = rho * (rewards[:, 1:] + gamma * values[:, 1:] * nonterminal - values[:, :-1])
    decay = gamma * gae_lambda * c * nonterminal
    _, advantages = linear_scan(decay[:, ::-1], delta[:, ::-1])
    return mx.concatenate([advantages[:, ::-1], mx.zeros_like(values[:, :1])], axis=1)

def ppo_loss(logits, values, actions, old_logprobs, old_values, rewards, terminals, ent_coef, act_sizes, c):
    '''ppo_loss_compute over segments (B, T): the clipped policy loss, the clipped value loss and
    the entropy bonus, and the stats pufferl.cu logs (LOSS_KEYS)'''
    segments, horizon = old_values.shape
    logprob, entropy = logprob_entropy(logits.reshape(segments * horizon, -1),
        actions.reshape(segments * horizon, -1), act_sizes)
    logratio = logprob.reshape(segments, horizon) - old_logprobs
    ratio = mx.exp(logratio)

    # Advantages come from this minibatch's own values, before the update, and
    # nothing flows back through them
    live = mx.stop_gradient(values)
    importance = mx.stop_gradient(ratio) if c['vtrace'] else mx.ones_like(ratio)
    advantages = puff_advantage(live, rewards, terminals, importance, c['gamma'],
        c['gae_lambda'], c['vtrace_rho_clip'], c['vtrace_c_clip'])
    returns = live + advantages
    if c.get('norm_adv', 0):
        # Not in 5.0, which is why it is off unless a config asks: the minibatch's
        # advantages centered and scaled to unit variance, as 4.0 did. Muon's update
        # is the same size whatever the gradient, so with rewards as small as sm64's
        # the policy term barely steers it otherwise; see the README.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    clipped_ratio = mx.clip(ratio, 1 - c['clip_coef'], 1 + c['clip_coef'])
    pg_loss = mx.maximum(-advantages * ratio, -advantages * clipped_ratio).mean()
    v_clipped = old_values + mx.clip(values - old_values, -c['vf_clip_coef'], c['vf_clip_coef'])
    v_loss = 0.5 * mx.maximum((values - returns) ** 2, (v_clipped - returns) ** 2).mean()
    entropy = entropy.mean()
    loss = pg_loss + c['vf_coef'] * v_loss - ent_coef * entropy
    stats = mx.stack([pg_loss, v_loss, entropy, loss, (-logratio).mean(), ((ratio - 1) - logratio).mean(),
        (mx.abs(ratio - 1) > c['clip_coef']).astype(mx.float32).mean(), ratio.mean()])
    return loss, stats

def cosine_annealing(base, minimum, epoch, total_epochs):
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * epoch / total_epochs))

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
    '''algo.cu's Muon: Nesterov momentum on the clipped gradient, orthogonalized for matrices.

    An orthogonalized update is the same size however small the gradient was, so a
    reward that says nothing for millions of steps walks the weights in random
    directions until they blow up: the sm64 door-or-nothing run did at 3.4M steps,
    value loss 0 -> 567,829 and entropy to zero. Weight decay holds them to a size.
    PufferLib has none, so that is the default.'''
    def __init__(self, learning_rate, momentum=0.95, weight_decay=0.0):
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

# --- Checkpoints

def _checkpoint_order(policy):
    '''The order 5.0's weights_create registers parameters in: encoder, decoder, then the MinGRU'''
    encoder = [(f'encoder.{name}', value) for name, value in tree_flatten(policy.encoder.parameters())]
    network = [(f'network.layers.{i}.weight', layer.weight) for i, layer in enumerate(policy.network.layers)]
    return encoder + [('decoder.weight', policy.decoder.weight)] + network

def save_weights(policy, path):
    '''A 5.0 checkpoint: every parameter as float32 in row order, each padded to a multiple of
    8 numbers as the CUDA trainer's allocator lays them out, so PufferLib's own CPU eval
    (src/puffercpu.c) reads the default policy's'''
    chunks = []
    for _, value in _checkpoint_order(policy):
        flat = np.array(value, dtype=np.float32).ravel()
        chunks += [flat, np.zeros(-flat.size % 8, dtype=np.float32)]
    tmp = f'{path}.tmp.{os.getpid()}'
    np.concatenate(chunks).tofile(tmp)
    os.replace(tmp, path)

def load_weights(policy, path):
    with open(path, 'rb') as f:
        if f.read(4) == b'PK\x03\x04':
            # Numpy's .npz, which the 4.0 port wrote under the same .bin names
            weights = np.load(path)
            policy.load_weights([(name, mx.array(weights[name])) for name in weights.files])
            return
    data = np.fromfile(path, dtype=np.float32)
    weights, offset = [], 0
    for name, value in _checkpoint_order(policy):
        if offset + value.size > data.size:
            raise ValueError(f'{path} has {data.size} numbers, too few for this policy')
        weights.append((name, mx.array(data[offset:offset + value.size].reshape(value.shape))))
        offset += value.size + (-value.size % 8)
    if data.size > offset:
        raise ValueError(f'{path} has {data.size} numbers, more than this policy\'s {offset}')
    policy.load_weights(weights)

def _most_trained_checkpoint(paths):
    '''Puffer names a checkpoint after the agent step it was written at, so the most
    trained one is the highest number and not the newest file. Picking by mtime loads
    whatever ran last, which after a short run is a policy that knows nothing.'''
    def key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        return (int(name) if name.isdigit() else -1, os.path.getctime(path))
    return max(paths, key=key)

def checkpoint_path(args):
    load_path = args['base']['load_model_path']
    if load_path in ('None', ''):
        return None
    if load_path != 'latest':
        return load_path
    root = os.path.join(args['base']['checkpoint_dir'], args['base']['env_name'])
    candidates = glob.glob(os.path.join(root, '**', '*.bin'), recursive=True)
    if not candidates:
        sys.exit(f'no .bin checkpoints found in {root}/')
    return _most_trained_checkpoint(candidates)

# --- The trainer (src/pufferl.cu)

class PuffeRL:
    def __init__(self, args, vec, policy):
        base, train = args['base'], args['train']
        self.args = args
        self.config = train
        self.vec = vec
        self.policy = policy
        self.act_sizes = vec.act_sizes
        self.total_agents = N = vec.total_agents
        self.horizon = H = train['horizon']
        self.batch_size = N * H
        self.segments = train['minibatch_size'] // H
        self.total_epochs = max(1, train['total_timesteps'] // self.batch_size)
        self.reset_every_horizon = bool(base['reset_every_horizon'])

        self.state = policy.network.initial_state(N)
        self.rewards = np.zeros((H, N), dtype=np.float32)
        self.terminals = np.zeros((H, N), dtype=np.float32)
        self.optimizer = Muon(learning_rate=train['learning_rate'], momentum=train['momentum'],
            weight_decay=train.get('weight_decay', 0.0))
        self.optimizer.init(policy.trainable_parameters())
        # With target_entropy set, ent_coef is only where the coefficient starts: see train()
        self.ent_coef = train['ent_coef']
        self.log_ent_coef = math.log(train['ent_coef']) if train['ent_coef'] > 0 else 0.0
        self._act = self._compile_act()
        self._train_step = self._compile_train_step()

        self.num_params = sum(v.size for _, v in tree_flatten(policy.trainable_parameters()))
        self.epoch = 0
        self.global_step = 0
        self.start_time = self.last_log_time = time.time()
        self.last_log_step = 0
        self.loss_sums = np.zeros(len(LOSS_KEYS))
        self.loss_count = 0
        self.perf = defaultdict(float)
        vec.reset()

    def _compile_act(self):
        policy, act_sizes = self.policy, self.act_sizes

        @partial(mx.compile, inputs=[policy.state, mx.random.state], outputs=[mx.random.state])
        def act(obs, terminals, state):
            # An agent whose episode just ended starts the next one from a clear state
            state = state * (1.0 - terminals)[None, :, None]
            logits, value, state = policy.step(obs, state)
            action, logprob = sample_actions(logits, act_sizes)
            return action, logprob, value, state

        return act

    def _compile_train_step(self):
        policy, optimizer, act_sizes, c = self.policy, self.optimizer, self.act_sizes, self.config

        def loss_fn(obs, actions, old_logprobs, old_values, rewards, terminals, state, ent_coef):
            logits, values = policy(obs, state, terminals)
            return ppo_loss(logits, values, actions, old_logprobs, old_values, rewards, terminals,
                ent_coef, act_sizes, c)

        loss_and_grad = nn.value_and_grad(policy, loss_fn)
        state = [policy.state, optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def train_step(*batch):
            (_, stats), grads = loss_and_grad(*batch)
            grads, _ = optim.clip_grad_norm(grads, c['max_grad_norm'])
            optimizer.update(policy, grads)
            return stats

        return train_step

    def rollouts(self, store=True):
        '''horizon steps of every agent. At each: the observation, and the reward and terminal
        the last step left, are stored with the action taken from them'''
        vec, H = self.vec, self.horizon
        start = time.perf_counter()
        if self.reset_every_horizon:
            self.state = mx.zeros_like(self.state)

        observations, actions, logprobs, values = [], [], [], []
        for t in range(H):
            t0 = time.perf_counter()
            obs = mx.array(vec.observations)
            terminals = mx.array(vec.terminals)
            if store:
                self.rewards[t] = vec.rewards
                self.terminals[t] = vec.terminals
                if t == 0:
                    self.initial_state = self.state * (1.0 - terminals)[None, :, None]
            t1 = time.perf_counter()
            action, logprob, value, self.state = self._act(obs, terminals, self.state)
            env_actions = action.astype(mx.float32)
            mx.eval(env_actions, logprob, value, self.state)
            t2 = time.perf_counter()
            if store:
                observations.append(obs)
                actions.append(action)
                logprobs.append(logprob)
                values.append(value)
            vec.actions[:] = np.asarray(env_actions)
            t3 = time.perf_counter()
            vec.step()
            self.perf['eval_copy'] += (t1 - t0) + (t3 - t2)
            self.perf['eval_model'] += t2 - t1
            self.perf['eval_env'] += time.perf_counter() - t3

        if store:
            # (agents, horizon, ...) as train_epoch_gpu transposes them
            self.observations = mx.stack(observations, axis=1)
            self.actions = mx.stack(actions, axis=1)
            self.logprobs = mx.stack(logprobs, axis=1)
            self.values = mx.stack(values, axis=1)
            mx.eval(self.observations, self.actions, self.logprobs, self.values, self.initial_state)
        self.global_step += self.total_agents * H
        self.perf['rollout'] += time.perf_counter() - start

    def train(self):
        c = self.config
        start = time.perf_counter()
        if c['anneal_lr']:
            self.optimizer.learning_rate = cosine_annealing(c['learning_rate'],
                c['min_lr_ratio'] * c['learning_rate'], self.epoch, self.total_epochs)
        ent_coef = c['ent_coef']
        target_entropy = c.get('target_entropy', 0)
        if target_entropy:
            ent_coef = self.ent_coef
        elif c['anneal_ent_coef']:
            ent_coef = cosine_annealing(ent_coef, c['min_ent_coef_ratio'] * ent_coef, self.epoch, self.total_epochs)
        self.current_ent_coef = ent_coef

        rewards = mx.array(np.clip(self.rewards, -1, 1).T)
        terminals = mx.array(self.terminals.T)
        state = self.initial_state
        mx.eval(rewards, terminals)
        model_start = time.perf_counter()

        # Minibatches are runs of whole rows in agent order, as train_epoch_gpu slices them
        rows, segs = self.total_agents, self.segments
        num_minibatches = int(c['replay_ratio'] * self.batch_size / c['minibatch_size'])
        entropy_sum = 0.0
        for mb in range(num_minibatches):
            lo = (mb * segs) % rows
            hi = lo + segs
            stats = self._train_step(self.observations[lo:hi], self.actions[lo:hi], self.logprobs[lo:hi],
                self.values[lo:hi], rewards[lo:hi], terminals[lo:hi], state[:, lo:hi], mx.array(ent_coef))
            mx.eval(stats, self.policy.state, self.optimizer.state)
            stats = np.array(stats)
            self.loss_sums += stats
            self.loss_count += 1
            entropy_sum += stats[2]

        if target_entropy and num_minibatches:
            # Steer the coefficient toward the entropy asked for: up while the policy is
            # more certain than that, down while it is less, by a factor of e every
            # 1 / (ent_coef_rate * gap) epochs, kept between 1e-4 and 0.5.
            gap = target_entropy - entropy_sum / num_minibatches
            self.log_ent_coef = min(max(self.log_ent_coef + c.get('ent_coef_rate', 0.01) * gap,
                math.log(1e-4)), math.log(0.5))
            # That sum alone overshoots where entropy answers the coefficient late: it keeps
            # climbing until entropy has crossed the target, and by then is far past what holds
            # it there. ent_coef_damping adds the gap itself, a factor of e for every
            # 1 / ent_coef_damping of entropy off target, which is gone the moment the gap is.
            damped = self.log_ent_coef + c.get('ent_coef_damping', 0.0) * gap
            self.ent_coef = math.exp(min(max(damped, math.log(1e-4)), math.log(0.5)))
        self.epoch += 1

        end = time.perf_counter()
        self.perf['train_misc'] += model_start - start
        self.perf['train_model'] += end - model_start

    def log(self, clear=True):
        '''The keys run_train logs, under the names pufferl.cu gives them'''
        now = time.time()
        dt = now - self.last_log_time
        logs = {
            'SPS': (self.global_step - self.last_log_step) / dt if dt > 0 else 0,
            'agent_steps': self.global_step,
            'uptime': now - self.start_time,
            'epoch': self.epoch,
        }
        self.last_log_time, self.last_log_step = now, self.global_step
        logs.update({f'env/{k}': v for k, v in self.vec.log(clear).items()})
        if self.loss_count:
            logs.update(zip(LOSS_KEYS, (self.loss_sums / self.loss_count).tolist()))
            logs['train/ent_coef'] = self.current_ent_coef
        self.loss_sums[:] = 0
        self.loss_count = 0
        logs['util/vram_used_gb'] = mx.get_active_memory() / 1e9
        logs['util/vram_total_gb'] = mx.device_info()['memory_size'] / 1e9
        logs['util/cpu_mem_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        for key in PERF_KEYS:
            logs[f'perf/{key}'] = self.perf[key]
        logs['perf/train'] = self.perf['train_misc'] + self.perf['train_model']
        self.perf = defaultdict(float)
        return logs

    def restart(self):
        '''Fresh episodes and a clear recurrent state, as env_restart gives eval'''
        self.vec.reset()
        self.state = mx.zeros_like(self.state)
        self.vec.log(clear=True)

    def close(self):
        self.vec.close()

    @classmethod
    def create(cls, ini):
        args = parsed(ini)
        base, vec_config, train = args['base'], args['vec'], args['train']
        env_name = base['env_name']
        vec = VecEnv(env_name, ini['env'], int(vec_config['total_agents']), int(vec_config['num_threads']))
        if sum(vec.act_sizes) == len(vec.act_sizes):
            sys.exit('mlx_pufferl only implements discrete actions')

        mx.random.seed(base['seed'])
        # An env that shows its policy a picture puts it last in the observation, RGB,
        # and says how big it is in its own config
        picture_shape = None
        width, height = int(args['env'].get('picture_width', 0)), int(args['env'].get('picture_height', 0))
        if width > 0 and height > 0:
            picture_shape = (height, width, 3)
            if vec.obs_size < height * width * 3:
                sys.exit(f'an observation of {vec.obs_size} has no room for a {width}x{height} picture')
        policy = Policy(vec.obs_size, sum(vec.act_sizes), int(args['policy']['hidden_size']),
            int(args['policy']['num_layers']), picture_shape)
        mx.eval(policy.parameters())
        pufferl = cls(args, vec, policy)

        load_path = checkpoint_path(args)
        if load_path is not None:
            print(f'Loading {load_path}')
            load_weights(policy, load_path)
        return pufferl

# --- Dashboard (puf_dashboard_print)

WIDTH = 80
BASE_ROWS = 12
MAX_USER_ROWS = 15

class Dashboard:
    def __init__(self):
        self.tty = sys.stdout.isatty()
        self.frame = 0
        self.last_size = None

    def color(self, code):
        return code if self.tty else ''

    def cell(self, text, width):
        '''Right-aligned in width columns, with unit letters of a number in gray'''
        text = text[:width]
        numeric = any(ch.isdigit() for ch in text)
        out = [self.color('\033[97m'), ' ' * (width - len(text))]
        for ch in text:
            unit = numeric and ch in '%KMBTGdhms'
            out.append((self.color('\033[90m') if unit else self.color('\033[97m')) + ch)
        return ''.join(out) + self.color('\033[0m')

    @staticmethod
    def abbrev(value):
        for suffix in ('', 'K', 'M', 'B'):
            if value < 1000:
                return f'{value:.1f}{suffix}'
            value /= 1000
        return f'{value:.1f}T'

    @staticmethod
    def duration(seconds):
        ms = int(max(seconds, 0) * 1000 + 0.5)
        if ms < 1000:
            return f'{ms}ms'
        if ms < 60000:
            return f'{ms // 1000}s {ms % 1000:03d}ms'
        if ms < 3600000:
            return f'{ms // 60000}m {(ms // 1000) % 60:02d}s {ms % 1000:03d}ms'
        s = ms // 1000
        return f'{s // 86400}d {(s // 3600) % 24}h {(s // 60) % 60}m {s % 60}s'

    def print(self, args, pufferl, log, epoch):
        A, W, G, R = (self.color(code) for code in ('\033[96m', '\033[97m', '\033[90m', '\033[0m'))
        size = shutil.get_terminal_size((WIDTH, 1000))
        rows, cols = (size.lines, size.columns) if self.tty else (1000, WIDTH)
        g = lambda key, default=0: log.get(key, default)
        env_name = args['base']['env_name']
        steps, sps = g('agent_steps', pufferl.global_step), g('SPS')
        target = pufferl.total_epochs * pufferl.batch_size
        remaining = (target - steps) / sps if sps > 0 and target > steps else 0
        rollout, train = g('perf/rollout'), g('perf/train')
        perf_total = rollout + train

        lines = []
        if self.tty:
            lines.append('\033[?2026h' + ('\033[H\033[J' if (rows, cols) != self.last_size else '\033[H'))
            self.last_size = (rows, cols)
        if self.tty and (cols < WIDTH or rows <= BASE_ROWS):
            compact = (f'PufferLib 5.0 (MLX)  env={env_name}  steps={self.abbrev(steps)}  SPS={self.abbrev(sps)}  '
                f'score={g("env/score"):.3f}  epoch={epoch}  to_go={self.duration(remaining)}')
            sys.stdout.write(''.join(lines) + compact[:cols - 1] + '\033[K\033[J\033[?2026l')
            sys.stdout.flush()
            return

        eol = '\033[K\n' if self.tty else '\n'
        rule = lambda left, right: f'{W}{left}{"─" * (WIDTH - 2)}{right}{R}'
        fish_span = 18
        fish = (fish_span - 3) - (self.frame % (fish_span - 2))
        self.frame += 1
        memory = f'{g("util/vram_used_gb"):.1f}/{g("util/vram_total_gb"):.0f}G'
        ram = f'{g("util/cpu_mem_gb"):.1f}G'
        lines.append(rule('╭', '╮'))
        # 🐡 is two columns wide
        lines.append(f'{W}│{R} {A}PufferLib {W}5.0{R}{" " * fish}{A}🐡{R}{" " * (fish_span - 2 - fish)}'
            f'   {A}MLX{G}:{R}{self.cell(memory, 10)}    {A}Peak RAM{G}:{R}{self.cell(ram, 6)}{" " * 10}{W}│{R}')
        lines.append(f'{W}│{" " * (WIDTH - 2)}│{R}')
        table = [
            ('Env', env_name, 'Evaluate', None, rollout, 'Losses', 'loss/total'),
            ('Params', self.abbrev(pufferl.num_params), '  Model', 'perf/eval_model', 0, 'policy', 'loss/policy'),
            ('Steps', self.abbrev(steps), '  Env', 'perf/eval_env', 0, 'value', 'loss/value'),
            ('SPS', self.abbrev(sps), '  Copy', 'perf/eval_copy', 0, 'entropy', 'loss/entropy'),
            ('Epoch', str(epoch), 'Train', None, train, 'old_kl', 'loss/old_kl'),
            ('Uptime', self.duration(g('uptime')), '  Model', 'perf/train_model', 0, 'kl', 'loss/kl'),
            ('To go', self.duration(remaining), '  Misc', 'perf/train_misc', 0, 'clipfrac', 'loss/clipfrac'),
        ]
        if args['train'].get('target_entropy', 0):
            table.append(('', '', '', None, None, 'ent_coef', 'train/ent_coef'))
        for a, av, b, perf_key, perf_sec, c, loss_key in table:
            sec = g(perf_key) if perf_key else perf_sec
            timing = (self.cell(self.duration(sec), 6) + ' '
                + self.cell(f'{int(100 * sec / perf_total) if perf_total > 0 else 0}%', 4)) if sec is not None else ' ' * 11
            value = f'{log[loss_key]:.3f}' if loss_key in log else ''
            lines.append(f'{W}│{R} {A}{a:<9.9}{R} {self.cell(av, 13)}    {A}{b:<12.12}{R} {timing}'
                f'    {A}{c:<10.10}{R} {self.cell(value, 7)}    {W}│{R}')
        lines.append(f'{W}│{" " * (WIDTH - 2)}│{R}')

        user_rows = min(max(rows - BASE_ROWS - 1, 0), MAX_USER_ROWS) if self.tty else MAX_USER_ROWS
        stats = [(k[4:], v) for k, v in log.items() if k.startswith('env/') and k != 'env/n'][:2 * user_rows]
        for i in range(0, len(stats), 2):
            left = f'{A}{stats[i][0]:<25.25}{R} {W}{stats[i][1]:>9.3f}{R}'
            right = f'{A}{stats[i + 1][0]:<25.25}{R} {W}{stats[i + 1][1]:>9.3f}{R}' if i + 1 < len(stats) else ' ' * 35
            lines.append(f'{W}│{R} {left}   {right}    {W}│{R}')
        lines.append(rule('╰', '╯'))
        out = lines[0] + eol.join(lines[1:]) + eol if self.tty else '\n'.join(lines) + '\n'
        sys.stdout.write(out + ('\033[J\033[?2026l' if self.tty else ''))
        sys.stdout.flush()

# --- Train and eval (run_train, run_eval)

def _bin_mean(history, key, points):
    '''log_history_bin_mean: history[key] averaged over points bins of agent_steps, the last
    bin being the final value'''
    if points == 1:
        return [history[-1][key]]
    final_steps = history[-1]['agent_steps']
    out, bin_sum, bin_n = [], 0.0, 0
    fallback = history[0][key]
    next_bin = final_steps / (points - 1)
    for log in history:
        bin_sum += log[key]
        bin_n += 1
        if log['agent_steps'] < next_bin or len(out) >= points - 1:
            continue
        fallback = bin_sum / bin_n
        out.append(fallback)
        bin_sum, bin_n = 0.0, 0
        next_bin += final_steps / (points - 1)
    out += [fallback] * (points - 1 - len(out))
    return out + [history[-1][key]]

def eval_loop(ini, pufferl, dashboard, render, episodes, board=None):
    '''Rollouts with no training until episodes have ended, or until the window closes when
    rendering. With a board, env stats are merged into it for the dashboard'''
    args = parsed(ini)
    last_dash = 0.0
    while True:
        if render:
            pufferl.vec.render(0)
            if pufferl.vec.window_closed():
                return {}
        pufferl.rollouts(store=False)
        if render:
            continue
        log = pufferl.log(clear=False)
        if board is not None:
            board.update({k: v for k, v in log.items() if k.startswith('env/')})
        show = board if board is not None else log
        n = log.get('env/n', 0)
        if n >= episodes or time.time() - last_dash >= 0.6:
            dashboard.print(args, pufferl, show, pufferl.epoch)
            last_dash = time.time()
        if n >= episodes:
            return log

def run_train(ini):
    args = parsed(ini)
    base, train = args['base'], args['train']
    minibatch, horizon, agents = train['minibatch_size'], train['horizon'], args['vec']['total_agents']
    if horizon <= 0 or minibatch % horizon:
        sys.exit('train.minibatch_size must be divisible by train.horizon')
    if minibatch > horizon * agents:
        sys.exit('train.minibatch_size must be <= train.horizon * vec.total_agents')
    if agents % (minibatch // horizon):
        sys.exit('vec.total_agents must be divisible by minibatch rows')
    if base['run_id'] in ('None', ''):
        put(ini, 'base.run_id', int(1000 * time.time()))
    args = parsed(ini)
    base = args['base']
    env_name, run_id = base['env_name'], str(base['run_id'])
    checkpoint_dir = os.path.join(base['checkpoint_dir'], env_name, run_id)
    log_dir = os.path.join(base['log_dir'], env_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    pufferl = PuffeRL.create(ini)
    dashboard = Dashboard()
    train_epochs = train['total_timesteps'] // pufferl.batch_size
    interval = base['checkpoint_interval']
    target_key = f'env/{args["sweep"]["metric"]}'
    last_log, history = {}, []
    for epoch in range(train_epochs):
        pufferl.rollouts()
        pufferl.train()
        if epoch == train_epochs - 1 or (interval > 0 and (epoch + 1) % interval == 0):
            save_weights(pufferl.policy, os.path.join(checkpoint_dir, f'{pufferl.global_step:016d}.bin'))
        if last_log and time.time() < pufferl.last_log_time + 0.6 and epoch < train_epochs - 1:
            continue
        log = pufferl.log()
        dashboard.print(args, pufferl, log, pufferl.epoch)
        # A log with no episodes in it has no env stats, so keep the last one that did
        episodes = log.get('env/n', 0) > 0
        if episodes or not last_log:
            last_log = dict(log)
        else:
            last_log.update(uptime=log['uptime'], agent_steps=log['agent_steps'])
        if episodes and target_key in log:
            history.append(dict(last_log))

    eval_episodes = base['eval_episodes']
    if eval_episodes > 0 and train_epochs > 0:
        if base['eval_agents'] != -1:
            pufferl.close()
            put(ini, 'vec.total_agents', base['eval_agents'])
            policy = pufferl.policy
            pufferl = PuffeRL.create(ini)
            pufferl.policy.update(policy.parameters())
        pufferl.restart()
        eval_loop(ini, pufferl, dashboard, render=False, episodes=eval_episodes, board=last_log)
    pufferl.close()

    history.append(dict(last_log))
    points = args['sweep']['downsample']
    with open(os.path.join(log_dir, f'{run_id}.ini'), 'w') as f:
        f.write('# PufferLib log v1\n')
        write_ini(f, ini)
        f.write('\n[metrics]\n')
        if target_key in last_log:
            for key in last_log:
                if not key.startswith('loss/'):
                    values = _bin_mean([h for h in history if key in h], key, points)
                    f.write(f'{key} = {",".join(f"{v:.17g}" for v in values)}\n')

def run_eval(ini, render):
    args = parsed(ini)
    if render:
        # One env on screen, a step at a time
        put(ini, 'vec.total_agents', args['env']['num_agents'])
        put(ini, 'vec.num_threads', 1)
        put(ini, 'train.horizon', 1)
    put(ini, 'base.reset_every_horizon', 0)
    episodes = args['base']['eval_episodes']
    pufferl = PuffeRL.create(ini)
    log = eval_loop(ini, pufferl, Dashboard(), render, episodes)
    if not render:
        print(f'EVAL env={args["base"]["env_name"]} score={log.get("env/score", 0):.6f} '
            f'perf={log.get("env/perf", 0):.6f} games={int(log.get("env/n", 0))} params={pufferl.num_params}')
    pufferl.close()

def main():
    usage = 'usage: mlx_pufferl.py train|eval ENV [latest|MODEL.bin] [--headless] [--section.key=value ...]'
    if len(sys.argv) < 3 or sys.argv[1] not in ('train', 'eval'):
        sys.exit(usage)
    mode, env_name, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    model = None
    if mode == 'eval' and rest and not rest[0].startswith('-') and '=' not in rest[0]:
        model = rest.pop(0)
    headless = '--headless' in rest
    rest = [arg for arg in rest if arg != '--headless']
    ini = load_config(env_name, rest)
    if model:
        put(ini, 'base.load_model_path', model)
    if mode == 'train':
        run_train(ini)
    else:
        run_eval(ini, render=not headless)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))  # Aggressively exit on ctrl+c, as puffer does
    main()
