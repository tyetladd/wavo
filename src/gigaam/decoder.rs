// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Greedy decoding on the CPU. RNN-T: embedding + one LSTM layer as the predictor, joint network
//! `out(relu(enc + pred(h)))`; the encoder side of the joint is projected on the GPU. CTC: argmax
//! per frame over logits computed on the GPU, then collapse.

use crate::cpu::{matmul, rows4};
use crate::error::Result;
use crate::gguf::Gguf;

/// At most this many tokens per encoder frame.
const MAX_SYMBOLS: usize = 10;
/// Frames scored at once while the predictor output stays fixed.
const SPAN: usize = 2;

pub struct Decoder {
  hidden: usize,
  joint: usize,
  blank: usize,
  /// `[vocab][hidden]`
  embed: Vec<f32>,
  /// `[4·hidden][hidden]`, gates i, f, g, o.
  wx: Vec<f32>,
  wh: Vec<f32>,
  /// `bias_ih + bias_hh`
  bias: Vec<f32>,
  /// `[joint][hidden]`, rows padded to a multiple of 4.
  pred_w: Vec<f32>,
  pred_b: Vec<f32>,
  /// `[vocab][joint]`, rows padded to a multiple of 4.
  out_w: Vec<f32>,
  out_b: Vec<f32>,
}

struct State {
  h: Vec<f32>,
  c: Vec<f32>,
}

impl Decoder {
  pub fn new(g: &Gguf, hidden: usize, joint: usize, vocab: usize, blank: usize) -> Result<Self> {
    let t = |name: &str, dims: &[usize]| g.tensor(name, dims).and_then(|t| t.to_f32());
    Ok(Self {
      hidden,
      joint,
      blank,
      embed: t("pred.embed.weight", &[hidden, vocab])?,
      wx: t("pred.lstm.0.Wx", &[hidden, 4 * hidden])?,
      wh: t("pred.lstm.0.Wh", &[hidden, 4 * hidden])?,
      bias: t("pred.lstm.0.bias", &[4 * hidden])?,
      pred_w: rows4(t("joint.pred.weight", &[hidden, joint])?, hidden),
      pred_b: t("joint.pred.bias", &[joint])?,
      out_w: rows4(t("joint.out.weight", &[joint, vocab])?, joint),
      out_b: t("joint.out.bias", &[vocab])?,
    })
  }

  /// Greedy search over `enc` `[frames][joint]` (encoder projection with bias). Returns
  /// `(token, frame)` pairs. The first predictor input is zero; the predictor state advances
  /// only when a token is emitted, so frames are scored `SPAN` at a time against one predictor
  /// output, and the frames after a token are scored again.
  pub fn decode(&self, enc: &[f32]) -> Vec<(u32, u32)> {
    let (n, j) = (self.hidden, self.joint);
    let zero = State { h: vec![0.0; n], c: vec![0.0; n] };
    let mut next = self.lstm(&vec![0.0; n], &zero);
    let mut pred = self.predict(&next.h);
    let mut out = Vec::new();
    let frames = enc.len() / j;
    let mut t = 0;
    while t < frames {
      let span = self.joint(&enc[t * j..(t + SPAN).min(frames) * j], &pred);
      let Some(skip) = span.iter().position(|&token| token != self.blank) else {
        t += span.len();
        continue;
      };
      t += skip;
      let mut token = span[skip];
      for _ in 0..MAX_SYMBOLS {
        out.push((token as u32, t as u32));
        next = self.lstm(&self.embed[token * n..][..n], &next);
        pred = self.predict(&next.h);
        token = self.joint(&enc[t * j..][..j], &pred)[0];
        if token == self.blank {
          break;
        }
      }
      t += 1;
    }
    out
  }

  /// [`Decoder::decode`]'s search with the bonus of `trie` ([`crate::hotwords::transducer`]).
  pub fn beam(
    &self,
    enc: &[f32],
    texts: &[Vec<char>],
    trie: &crate::hotwords::Trie,
    boost: f32,
    beam: usize,
  ) -> Vec<(u32, u32)> {
    let steps =
      crate::hotwords::Steps { blank: self.blank, durations: &[], max_symbols: MAX_SYMBOLS };
    let model = Beam { decoder: self, enc };
    let frames = enc.len() / self.joint;
    crate::hotwords::transducer(&model, frames, &steps, texts, trie, boost, beam)
  }

  fn lstm(&self, x: &[f32], s: &State) -> State {
    let n = self.hidden;
    let (gx, gh) = (matmul(&self.wx, x, n), matmul(&self.wh, &s.h, n));
    let gate = |i: usize| self.bias[i] + gx[i] + gh[i];
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let mut next = State { h: vec![0.0; n], c: vec![0.0; n] };
    for j in 0..n {
      let (i, f, g, o) = (gate(j), gate(n + j), gate(2 * n + j), gate(3 * n + j));
      next.c[j] = sigmoid(f) * s.c[j] + sigmoid(i) * g.tanh();
      next.h[j] = sigmoid(o) * next.c[j].tanh();
    }
    next
  }

  fn predict(&self, h: &[f32]) -> Vec<f32> {
    let d = matmul(&self.pred_w, h, self.hidden);
    d.iter().zip(&self.pred_b).map(|(d, b)| b + d).collect()
  }

  /// Argmax of the joint logits for each frame in `enc`; ties go to the lowest id.
  fn joint(&self, enc: &[f32], pred: &[f32]) -> Vec<usize> {
    let z: Vec<f32> = (enc.chunks_exact(self.joint))
      .flat_map(|e| e.iter().zip(pred).map(|(e, p)| (e + p).max(0.0)))
      .collect();
    let logits = matmul(&self.out_w, &z, self.joint);
    (logits.chunks_exact(self.out_w.len() / self.joint))
      .map(|l| {
        let mut best = (0, f32::NEG_INFINITY);
        for (v, (d, b)) in l.iter().zip(&self.out_b).enumerate() {
          let logit = d + b;
          if logit > best.1 {
            best = (v, logit);
          }
        }
        best.0
      })
      .collect()
  }
}

/// The predictor and joint over one clip's `enc`, for the beam search.
struct Beam<'a> {
  decoder: &'a Decoder,
  enc: &'a [f32],
}

impl crate::hotwords::Transducer for Beam<'_> {
  type State = State;

  fn start(&self) -> (State, Vec<f32>) {
    let d = self.decoder;
    let zero = State { h: vec![0.0; d.hidden], c: vec![0.0; d.hidden] };
    let next = d.lstm(&vec![0.0; d.hidden], &zero);
    let pred = d.predict(&next.h);
    (next, pred)
  }

  fn step(&self, state: &State, token: usize) -> (State, Vec<f32>) {
    let d = self.decoder;
    let next = d.lstm(&d.embed[token * d.hidden..][..d.hidden], state);
    let pred = d.predict(&next.h);
    (next, pred)
  }

  /// The LSTM and the joint's projection for all `steps` in one product per matrix.
  fn steps(&self, steps: &[(&State, usize)]) -> Vec<(State, Vec<f32>)> {
    let (d, n) = (self.decoder, self.decoder.hidden);
    let x: Vec<f32> =
      steps.iter().flat_map(|&(_, token)| &d.embed[token * n..][..n]).copied().collect();
    let h: Vec<f32> = steps.iter().flat_map(|(s, _)| &s.h).copied().collect();
    let (gx, gh) = (matmul(&d.wx, &x, n), matmul(&d.wh, &h, n));
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let states: Vec<State> = (steps.iter().enumerate())
      .map(|(k, (s, _))| {
        let gate = |i: usize| d.bias[i] + gx[k * 4 * n + i] + gh[k * 4 * n + i];
        let mut next = State { h: vec![0.0; n], c: vec![0.0; n] };
        for j in 0..n {
          let (i, f, g, o) = (gate(j), gate(n + j), gate(2 * n + j), gate(3 * n + j));
          next.c[j] = sigmoid(f) * s.c[j] + sigmoid(i) * g.tanh();
          next.h[j] = sigmoid(o) * next.c[j].tanh();
        }
        next
      })
      .collect();
    let hs: Vec<f32> = states.iter().flat_map(|s| &s.h).copied().collect();
    let pred = matmul(&d.pred_w, &hs, n);
    let rows = d.pred_w.len() / n;
    (states.into_iter().zip(pred.chunks_exact(rows)))
      .map(|(s, p)| (s, p.iter().zip(&d.pred_b).map(|(p, b)| p + b).collect()))
      .collect()
  }

  /// The joint's log probabilities of the tokens for frame `t`, all predictions in one product.
  fn joint(&self, t: usize, preds: &[&[f32]]) -> Vec<(Vec<f32>, Vec<f32>)> {
    let (d, j) = (self.decoder, self.decoder.joint);
    let enc = &self.enc[t * j..][..j];
    let z: Vec<f32> =
      preds.iter().flat_map(|p| enc.iter().zip(*p).map(|(e, p)| (e + p).max(0.0))).collect();
    let logits = matmul(&d.out_w, &z, j);
    (logits.chunks_exact(d.out_w.len() / j))
      .map(|l| {
        let mut l: Vec<f32> = l.iter().zip(&d.out_b).map(|(l, b)| l + b).collect();
        crate::hotwords::log_softmax(&mut l);
        (l, Vec::new())
      })
      .collect()
  }
}

/// Greedy CTC over `logits` `[frames][classes]`: per-frame argmax (ties go to the lowest id; the
/// reference takes it after a log-softmax, which only shifts each row), repeats of the previous
/// frame's label dropped, then blanks. Returns `(token, frame)` pairs; a token's frame is the
/// first of its run.
pub fn ctc(logits: &[f32], classes: usize, blank: usize) -> Vec<(u32, u32)> {
  let mut out = Vec::new();
  let mut prev = None;
  for (t, row) in logits.chunks_exact(classes).enumerate() {
    let mut best = (0, row[0]);
    for (v, &x) in row.iter().enumerate().skip(1) {
      if x > best.1 {
        best = (v, x);
      }
    }
    if prev != Some(best.0) && best.0 != blank {
      out.push((best.0 as u32, t as u32));
    }
    prev = Some(best.0);
  }
  out
}
