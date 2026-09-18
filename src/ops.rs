//! The kernels in src/kernels as ops: what each one is given, and how many threads it runs on.

use crate::gpu::{Buf, Gpu, Op, Params};

const MATMUL: &str = include_str!("kernels/matmul.wgsl");
const SUM_LAYERS: &str = include_str!("kernels/sum_layers.wgsl");
const REDUCE: &str = include_str!("kernels/reduce.wgsl");
const REDUCE_DEV: &str = include_str!("kernels/reduce_dev.wgsl");

/// A matmul thread's block of the result is this many rows and columns, and a workgroup
/// is this many threads each way.
const BLOCK: usize = 4;
const MATMUL_GROUP: usize = 8;
/// A k longer than twice this is cut into chunks of it.
const LONG_CHUNK: usize = 512;
/// A product of fewer threads than this leaves the GPU idle, and its k is cut finer.
const FEW_THREADS: usize = 4096;
const SHORT_CHUNK: usize = 32;

/// A matrix somewhere in a buffer: where it starts, and how far apart its rows and its
/// columns are. A transpose swaps the two strides.
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub buf: &'a Buf,
    pub off: usize,
    pub row_stride: usize,
    pub col_stride: usize,
    /// In a batch, how far one matrix is from the next
    pub batch_stride: usize,
}

impl<'a> View<'a> {
    /// A matrix of `cols` columns stored row after row from `off`.
    pub fn rows(buf: &'a Buf, off: usize, cols: usize) -> Self {
        Self { buf, off, row_stride: cols, col_stride: 1, batch_stride: 0 }
    }

    /// The transpose of a matrix of `cols` columns stored row after row from `off`.
    pub fn transposed(buf: &'a Buf, off: usize, cols: usize) -> Self {
        Self { buf, off, row_stride: 1, col_stride: cols, batch_stride: 0 }
    }

    /// The same matrix with each row `stride` numbers long, of which it is the first few.
    pub fn with_row_stride(mut self, stride: usize) -> Self {
        self.row_stride = stride;
        self
    }

    pub fn batched(mut self, stride: usize) -> Self {
        self.batch_stride = stride;
        self
    }
}

/// Where a matmul's result goes: row after row from `off`, rows `row_stride` apart.
#[derive(Clone, Copy)]
pub struct Out<'a> {
    pub buf: &'a Buf,
    pub off: usize,
    pub row_stride: usize,
    pub batch_stride: usize,
    pub accumulate: bool,
}

impl<'a> Out<'a> {
    pub fn new(buf: &'a Buf, cols: usize) -> Self {
        Self { buf, off: 0, row_stride: cols, batch_stride: 0, accumulate: false }
    }

    pub fn at(buf: &'a Buf, off: usize, row_stride: usize) -> Self {
        Self { buf, off, row_stride, batch_stride: 0, accumulate: false }
    }

    pub fn batched(mut self, stride: usize) -> Self {
        self.batch_stride = stride;
        self
    }

    pub fn accumulate(mut self) -> Self {
        self.accumulate = true;
        self
    }
}

/// c = a * b, where a is m by k and b is k by n.
pub fn matmul(gpu: &Gpu, m: usize, n: usize, k: usize, a: View, b: View, c: Out) -> Vec<Op> {
    matmul_batched(gpu, 1, m, n, k, a, b, c)
}

/// `batches` products of one shape. One op, or where k is cut into layers two: the products
/// in layers, into a buffer made here for them, and their sums.
#[allow(clippy::too_many_arguments)]
pub fn matmul_batched(gpu: &Gpu, batches: usize, m: usize, n: usize, k: usize, a: View, b: View, c: Out) -> Vec<Op> {
    let threads = batches * m.div_ceil(BLOCK) * n.div_ceil(BLOCK);
    let chunk = if k > 2 * LONG_CHUNK {
        LONG_CHUNK
    } else if threads < FEW_THREADS && k >= 2 * SHORT_CHUNK {
        SHORT_CHUNK
    } else {
        k
    };
    let layers = k.div_ceil(chunk);
    let partials = (layers > 1).then(|| gpu.storage("matmul.partials", batches * layers * m * n));
    let (target, c_off, c_rs, c_zs, accumulate) = match &partials {
        Some(partials) => (partials, 0, n, m * n, false),
        None => (c.buf, c.off, c.row_stride, c.batch_stride, c.accumulate),
    };
    let params = Params::new()
        .u(m).u(n).u(k)
        .u(a.off).u(a.row_stride).u(a.col_stride)
        .u(b.off).u(b.row_stride).u(b.col_stride)
        .u(c_off).u(c_rs)
        .u(accumulate as usize)
        .u(chunk).u(layers)
        .u(a.batch_stride).u(b.batch_stride)
        .u(c_zs);
    let across = BLOCK * MATMUL_GROUP;
    let groups = [n.div_ceil(across) as u32, m.div_ceil(across) as u32, (batches * layers) as u32];
    let mut ops = vec![gpu.op("matmul", MATMUL, &params, &[&a.buf.buffer, &b.buf.buffer, &target.buffer], groups)];
    if let Some(partials) = &partials {
        let rest = Params::new().u(m * n).u(n).u(layers).u(c.off).u(c.row_stride).u(c.batch_stride).u(c.accumulate as usize);
        ops.push(gpu.op_flat("sum_layers", SUM_LAYERS, batches * m * n, rest, &[&partials.buffer, &c.buf.buffer]));
    }
    ops
}

#[derive(Clone, Copy, PartialEq)]
pub enum Sum {
    Values,
    Squares,
}

/// The numbers one reduction works through.
pub struct Reduce<'a> {
    pub input: &'a Buf,
    pub in_off: usize,
    pub n_in: usize,
    /// Numbers that are `columns` apart are summed together, into `columns` sums.
    pub columns: usize,
    pub sum: Sum,
    pub output: &'a Buf,
    pub out_off: usize,
    pub accumulate: bool,
    /// Sums in the middle of a reduction in two passes, at least `columns * 256` numbers.
    pub scratch: &'a Buf,
}

/// How many sums the first of two passes leaves for each column.
const FAN: usize = 256;

pub fn scratch_len(columns: usize) -> usize {
    columns * FAN
}

#[allow(clippy::too_many_arguments)]
fn reduce_op(gpu: &Gpu, input: &Buf, in_off: usize, n_in: usize, n_out: usize, sum: Sum,
        output: &Buf, out_off: usize, accumulate: bool) -> Op {
    let rest = Params::new().u(n_in).u(in_off).u(out_off)
        .u((sum == Sum::Squares) as usize).u(accumulate as usize).f(1.0);
    gpu.op_flat("reduce", REDUCE, n_out, rest, &[&input.buffer, &output.buffer])
}

/// One pass where that is short work for a thread, and two where it is not.
pub fn reduce(gpu: &Gpu, r: Reduce) -> Vec<Op> {
    if r.n_in <= 4 * FAN * r.columns {
        return vec![reduce_op(gpu, r.input, r.in_off, r.n_in, r.columns, r.sum, r.output, r.out_off, r.accumulate)];
    }
    let middle = r.columns * FAN;
    assert!(r.scratch.len >= middle, "reduce: scratch too small");
    vec![
        reduce_op(gpu, r.input, r.in_off, r.n_in, middle, r.sum, r.scratch, 0, false),
        reduce_op(gpu, r.scratch, 0, middle, r.columns, Sum::Values, r.output, r.out_off, r.accumulate),
    ]
}

/// sums[0] = the sum of `input`, and sums[1] = the sum of its squared distances from its
/// mean: what a mean and a standard deviation are made of.
pub fn mean_and_deviation(gpu: &Gpu, input: &Buf, n: usize, sums: &Buf, scratch: &Buf) -> Vec<Op> {
    let mut ops = reduce(gpu, Reduce { input, in_off: 0, n_in: n, columns: 1, sum: Sum::Values,
        output: sums, out_off: 0, accumulate: false, scratch });
    let middle = n.min(FAN);
    let rest = Params::new().u(n).f(1.0 / n as f32);
    ops.push(gpu.op_flat("reduce_dev", REDUCE_DEV, middle, rest, &[&input.buffer, &sums.buffer, &scratch.buffer]));
    ops.push(reduce_op(gpu, scratch, 0, middle, 1, Sum::Values, sums, 1, false));
    ops
}
