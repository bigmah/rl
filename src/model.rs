//! The policy (src/algo.cu): PufferLib's default, a bias-free Linear encoder, a MinGRU, and a
//! single decoder matrix whose last row is the value head. An env that shows its policy a
//! picture gets the Nature DQN's three convolutions in front of the encoder.
//!
//! Every weight lives in one buffer, in the order and with the padding of a 5.0 checkpoint,
//! so a checkpoint is that buffer written out. Gradients, momentum and Muon's updates are
//! buffers laid out the same way.

use std::rc::Rc;

use crate::gpu::{Buf, Gpu, Op, Params};
use crate::ops::{matmul, Out, View};

const IM2COL: &str = include_str!("kernels/im2col.wgsl");
const COL2IM: &str = include_str!("kernels/col2im.wgsl");
const RELU: &str = include_str!("kernels/relu.wgsl");
const RELU_MASK: &str = include_str!("kernels/relu_mask.wgsl");

/// The convolutions of a PictureEncoder: channels out, kernel, stride.
const CONVS: [(usize, usize, usize); 3] = [(32, 8, 4), (64, 4, 2), (64, 3, 1)];

#[derive(Clone, Debug)]
pub struct Shapes {
    pub obs_size: usize,
    pub act_sizes: Vec<usize>,
    pub hidden: usize,
    pub layers: usize,
    /// Height, width and channels of the picture an observation ends in, if it does.
    pub picture: Option<(usize, usize, usize)>,
}

impl Shapes {
    pub fn num_logits(&self) -> usize {
        self.act_sizes.iter().sum()
    }

    pub fn num_heads(&self) -> usize {
        self.act_sizes.len()
    }

    /// The decoder's columns: a logit for every action of every head, and the value.
    pub fn dec_cols(&self) -> usize {
        self.num_logits() + 1
    }

    /// WGSL for the action heads, which the kernels that walk logits are compiled with.
    pub fn heads_wgsl(&self) -> String {
        let sizes: Vec<String> = self.act_sizes.iter().map(|a| format!("{a}u")).collect();
        format!("const NUM_HEADS: u32 = {}u;\nconst NUM_LOGITS: u32 = {}u;\nvar<private> ACT_SIZES: array<u32, {}> = array<u32, {}>({});",
            self.num_heads(), self.num_logits(), self.num_heads(), self.num_heads(), sizes.join(", "))
    }
}

/// One weight matrix in the flat buffers. A convolution's (out, ky, kx, in) is a matrix of
/// `out` rows to Muon, as it is to PufferLib.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub shape: Vec<usize>,
    pub off: usize,
    pub rows: usize,
    pub cols: usize,
    /// Kaiming-uniform gain: weights start in U(-gain / sqrt(fan_in), gain / sqrt(fan_in))
    gain: f32,
}

impl Param {
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }
}

/// One convolution's geometry.
#[derive(Clone, Copy, Debug)]
pub struct Conv {
    pub in_h: usize, pub in_w: usize, pub in_c: usize,
    pub out_h: usize, pub out_w: usize, pub out_c: usize,
    pub kernel: usize, pub stride: usize,
}

impl Conv {
    pub fn in_size(&self) -> usize {
        self.in_h * self.in_w * self.in_c
    }

    pub fn out_size(&self) -> usize {
        self.out_h * self.out_w * self.out_c
    }

    /// Places the kernel is laid on one picture
    pub fn places(&self) -> usize {
        self.out_h * self.out_w
    }

    /// Numbers under the kernel at one place, which is the columns of a weight
    pub fn patch(&self) -> usize {
        self.kernel * self.kernel * self.in_c
    }
}

pub struct Model {
    pub gpu: Rc<Gpu>,
    pub shapes: Shapes,
    /// In checkpoint order: the encoder's, the decoder, then the MinGRU's layers
    pub params: Vec<Param>,
    pub convs: Vec<Conv>,
    /// Numbers of an observation before its picture, which go straight to the encoder's Linear
    pub features: usize,
    pub weights: Buf,
    /// The flat buffers' length: every matrix padded to a multiple of 8 numbers
    pub flat_len: usize,
    pub num_params: usize,
}

/// What a forward pass of the encoder leaves behind for its backward pass, and room for
/// both to work in.
pub struct EncoderActs {
    conv_out: Vec<Buf>,
    conv_grad: Vec<Buf>,
    /// Every place a convolution's kernel is laid, as rows: im2col.wgsl. One convolution's
    /// at a time, so the backward pass makes them again rather than keep all three.
    cols: Buf,
}

impl Model {
    pub fn new(gpu: Rc<Gpu>, shapes: Shapes, seed: u64) -> Self {
        let mut params: Vec<Param> = Vec::new();
        let mut convs = Vec::new();
        let mut off = 0;
        let mut add = |name: String, shape: Vec<usize>, gain: f32, params: &mut Vec<Param>| {
            let rows = shape[0];
            let cols: usize = shape[1..].iter().product();
            params.push(Param { name, shape, off, rows, cols, gain });
            off += (rows * cols).next_multiple_of(8);
        };

        let sqrt2 = std::f32::consts::SQRT_2;
        let mut features = shapes.obs_size;
        if let Some((height, width, channels)) = shapes.picture {
            features = shapes.obs_size - height * width * channels;
            let (mut h, mut w, mut c) = (height, width, channels);
            for (i, &(out_c, kernel, stride)) in CONVS.iter().enumerate() {
                if h < kernel || w < kernel {
                    crate::fail(&format!("a {width}x{height} picture is too small for the convolutions"));
                }
                let conv = Conv { in_h: h, in_w: w, in_c: c, out_h: (h - kernel) / stride + 1,
                    out_w: (w - kernel) / stride + 1, out_c, kernel, stride };
                add(format!("encoder.convs.{i}.weight"), vec![out_c, kernel, kernel, c], sqrt2, &mut params);
                convs.push(conv);
                (h, w, c) = (conv.out_h, conv.out_w, conv.out_c);
            }
            add("encoder.linear.weight".into(), vec![shapes.hidden, features + h * w * c], sqrt2, &mut params);
        } else {
            add("encoder.weight".into(), vec![shapes.hidden, shapes.obs_size], sqrt2, &mut params);
        }
        add("decoder.weight".into(), vec![shapes.dec_cols(), shapes.hidden], 1.0, &mut params);
        for i in 0..shapes.layers {
            add(format!("network.layers.{i}.weight"), vec![3 * shapes.hidden, shapes.hidden], 1.0, &mut params);
        }
        let flat_len = off;

        // puf_kaiming_init: uniform within gain / sqrt(fan_in) of 0
        let mut rng = crate::rng::Rng::new(seed);
        let mut init = vec![0f32; flat_len];
        for param in &params {
            let bound = param.gain / (param.cols as f32).sqrt();
            for value in &mut init[param.off..param.off + param.size()] {
                *value = (2.0 * rng.uniform() - 1.0) * bound;
            }
        }
        let weights = gpu.storage_from("weights", &init);
        let num_params = params.iter().map(Param::size).sum();
        Self { gpu, shapes, params, convs, features, weights, flat_len, num_params }
    }

    pub fn param(&self, name: &str) -> &Param {
        self.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no weight called {name}"))
    }

    pub fn decoder(&self) -> &Param {
        self.param("decoder.weight")
    }

    pub fn layer(&self, i: usize) -> &Param {
        self.param(&format!("network.layers.{i}.weight"))
    }

    fn encoder_linear(&self) -> &Param {
        self.param(if self.convs.is_empty() { "encoder.weight" } else { "encoder.linear.weight" })
    }

    /// A buffer laid out like the weights: for gradients, momentum and updates.
    pub fn flat(&self, label: &str) -> Buf {
        self.gpu.storage(label, self.flat_len)
    }

    /// y = x * w^T for `rows` rows of x.
    pub fn linear(&self, rows: usize, x: &Buf, w: &Param, y: &Buf) -> Vec<Op> {
        matmul(&self.gpu, rows, w.rows, w.cols, View::rows(x, 0, w.cols),
            View::transposed(&self.weights, w.off, w.cols), Out::new(y, w.rows))
    }

    /// grad_x = grad_y * w, added to what grad_x holds if `accumulate`.
    pub fn linear_grad_input(&self, rows: usize, grad_y: &Buf, w: &Param, grad_x: &Buf, accumulate: bool) -> Vec<Op> {
        let out = Out::new(grad_x, w.cols);
        matmul(&self.gpu, rows, w.cols, w.rows, View::rows(grad_y, 0, w.rows),
            View::rows(&self.weights, w.off, w.cols), if accumulate { out.accumulate() } else { out })
    }

    /// grad_w = grad_y^T * x, into `grads` where w is in the weights.
    pub fn linear_grad_weight(&self, rows: usize, grad_y: &Buf, x: &Buf, w: &Param, grads: &Buf) -> Vec<Op> {
        matmul(&self.gpu, w.rows, w.cols, rows, View::transposed(grad_y, 0, w.rows),
            View::rows(x, 0, w.cols), Out::at(grads, w.off, w.cols))
    }

    pub fn encoder_acts(&self, rows: usize, backward: bool) -> EncoderActs {
        let conv_out = self.convs.iter().enumerate()
            .map(|(i, conv)| self.gpu.storage(&format!("conv_out.{i}"), rows * conv.out_size())).collect();
        let conv_grad = if backward {
            self.convs.iter().enumerate()
                .map(|(i, conv)| self.gpu.storage(&format!("conv_grad.{i}"), rows * conv.out_size())).collect()
        } else {
            Vec::new()
        };
        let cols = self.convs.iter().map(|conv| rows * conv.places() * conv.patch()).max().unwrap_or(0);
        EncoderActs { conv_out, conv_grad, cols: self.gpu.storage("conv_cols", cols) }
    }

    /// One convolution's input, cut into the rows of `acts.cols`.
    fn im2col(&self, rows: usize, i: usize, obs: &Buf, acts: &EncoderActs) -> Op {
        let conv = &self.convs[i];
        let (input, in_stride, in_off) = match i {
            0 => (obs, self.shapes.obs_size, self.features),
            _ => (&acts.conv_out[i - 1], conv.in_size(), 0),
        };
        let rest = Params::new().u(in_stride).u(in_off).u(conv.in_w).u(conv.in_c)
            .u(conv.out_h).u(conv.out_w).u(conv.kernel).u(conv.stride);
        self.gpu.op_flat("im2col", IM2COL, rows * conv.places() * conv.kernel, rest, &[&input.buffer, &acts.cols.buffer])
    }

    /// `rows` observations to `rows` hidden vectors.
    pub fn encoder_forward(&self, rows: usize, obs: &Buf, acts: &EncoderActs, out: &Buf) -> Vec<Op> {
        let obs_size = self.shapes.obs_size;
        let linear = self.encoder_linear();
        if self.convs.is_empty() {
            return self.linear(rows, obs, linear, out);
        }

        // A convolution is its places times its weights' transpose, and a ReLU
        let mut ops = Vec::new();
        for (i, conv) in self.convs.iter().enumerate() {
            let w = self.param(&format!("encoder.convs.{i}.weight"));
            let places = rows * conv.places();
            ops.push(self.im2col(rows, i, obs, acts));
            ops.extend(matmul(&self.gpu, places, conv.out_c, conv.patch(), View::rows(&acts.cols, 0, conv.patch()),
                View::transposed(&self.weights, w.off, conv.patch()), Out::new(&acts.conv_out[i], conv.out_c)));
            ops.push(self.gpu.op_flat("relu", RELU, places * conv.out_c, Params::new(), &[&acts.conv_out[i].buffer]));
        }

        // The Linear sees the features and the last convolution side by side. They are in
        // two buffers, so it is two products into the one output: each with its own block
        // of the matrix's columns.
        let flat = &acts.conv_out[self.convs.len() - 1];
        let flat_size = self.convs[self.convs.len() - 1].out_size();
        let w = View::transposed(&self.weights, linear.off, linear.cols);
        let mut picture = Out::new(out, linear.rows);
        if self.features > 0 {
            ops.extend(matmul(&self.gpu, rows, linear.rows, self.features,
                View::rows(obs, 0, self.features).with_row_stride(obs_size), w, Out::new(out, linear.rows)));
            picture = picture.accumulate();
        }
        ops.extend(matmul(&self.gpu, rows, linear.rows, flat_size, View::rows(flat, 0, flat_size),
            View { off: linear.off + self.features, ..w }, picture));
        ops
    }

    /// The encoder's weight gradients, from the gradient of what it put out.
    pub fn encoder_backward(&self, rows: usize, obs: &Buf, acts: &EncoderActs, grad_out: &Buf, grads: &Buf) -> Vec<Op> {
        let obs_size = self.shapes.obs_size;
        let linear = self.encoder_linear();
        if self.convs.is_empty() {
            return self.linear_grad_weight(rows, grad_out, obs, linear, grads);
        }

        let last = self.convs.len() - 1;
        let flat_size = self.convs[last].out_size();
        let grad_y = View::transposed(grad_out, 0, linear.rows);
        let mut ops = Vec::new();
        if self.features > 0 {
            ops.extend(matmul(&self.gpu, linear.rows, self.features, rows, grad_y,
                View::rows(obs, 0, self.features).with_row_stride(obs_size), Out::at(grads, linear.off, linear.cols)));
        }
        ops.extend(matmul(&self.gpu, linear.rows, flat_size, rows, grad_y,
            View::rows(&acts.conv_out[last], 0, flat_size), Out::at(grads, linear.off + self.features, linear.cols)));
        ops.extend(matmul(&self.gpu, rows, flat_size, linear.rows, View::rows(grad_out, 0, linear.rows),
            View::rows(&self.weights, linear.off + self.features, 1).with_row_stride(linear.cols),
            Out::new(&acts.conv_grad[last], flat_size)));

        for (i, conv) in self.convs.iter().enumerate().rev() {
            let w = self.param(&format!("encoder.convs.{i}.weight"));
            let grad = &acts.conv_grad[i];
            let places = rows * conv.places();
            ops.push(self.gpu.op_flat("relu_mask", RELU_MASK, places * conv.out_c, Params::new(),
                &[&acts.conv_out[i].buffer, &grad.buffer]));
            // The weights' gradient is the output's, transposed, times the places
            ops.push(self.im2col(rows, i, obs, acts));
            ops.extend(matmul(&self.gpu, conv.out_c, conv.patch(), places, View::transposed(grad, 0, conv.out_c),
                View::rows(&acts.cols, 0, conv.patch()), Out::at(grads, w.off, conv.patch())));
            if i > 0 {
                // And the places' gradient is the output's times the weights, gathered
                // back into the picture the places were cut from
                ops.extend(matmul(&self.gpu, places, conv.patch(), conv.out_c, View::rows(grad, 0, conv.out_c),
                    View::rows(&self.weights, w.off, conv.patch()), Out::new(&acts.cols, conv.patch())));
                let rest = Params::new().u(conv.in_h).u(conv.in_w).u(conv.in_c)
                    .u(conv.out_h).u(conv.out_w).u(conv.kernel).u(conv.stride);
                ops.push(self.gpu.op_flat("col2im", COL2IM, rows * conv.in_size(), rest,
                    &[&acts.cols.buffer, &acts.conv_grad[i - 1].buffer]));
            }
        }
        ops
    }

    pub fn read_weights(&self) -> Vec<f32> {
        self.gpu.read(&self.weights, 0, self.flat_len)
    }

    pub fn write_weights(&self, flat: &[f32]) {
        assert_eq!(flat.len(), self.flat_len);
        self.gpu.write(&self.weights, 0, flat);
    }
}
