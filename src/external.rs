// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! An encoder that runs outside wavo, for [`Model::load_external`](crate::Model::load_external):
//! the caller's own runtime (MNN, LiteRT or another) runs GigaAM's encoder, and wavo keeps the
//! frontend, the decoder, the splitting at pauses and the token times.

use crate::gpu::Device;

/// GigaAM's encoder in another runtime: the two stride-2 conv1d + ReLU subsampling layers, the
/// Conformer blocks and the head's linear layer, exactly as wavo's CPU encoder computes them
/// (`src/gigaam/cpu.rs`). wavo calls it from the thread that calls
/// [`Model::transcribe`](crate::Model::transcribe), one call at a time.
pub trait Encoder: Send + Sync {
  /// Encodes `mel`, time-major `[frames][mels]` (wavo's log-mel features), and returns the head's
  /// outputs time-major `[t][width]`, where `t` is `frames` halved twice by the stride-2 convs
  /// (`(n − 1) / 2 + 1` each time) and `width` is the joint's hidden size for an RNN-T head
  /// (`stt.gigaam.joint.hidden`) or the classes for a CTC head. A failure's message reaches the
  /// caller as [`Error::Encoder`](crate::Error::Encoder).
  fn encode(&self, mel: &[f32], mels: usize) -> std::result::Result<Vec<f32>, String>;

  /// What [`Model::device`](crate::Model::device) reports while this encoder is loaded, such as
  /// `Device::external("CPU", "MNN int8, 4 threads", false)`.
  fn device(&self) -> &Device;
}

impl Device {
  /// A device for an [`Encoder`]: shown as `{name} ({runtime})`, with
  /// [`Api::External`](crate::Api::External) and [`DeviceKind::Cpu`](crate::DeviceKind::Cpu), or
  /// [`DeviceKind::IntegratedGpu`](crate::DeviceKind::IntegratedGpu) when `gpu`.
  pub fn external(name: &str, runtime: &str, gpu: bool) -> Self {
    use crate::gpu::{Api, DeviceKind, Kernels};
    Self {
      name: name.into(),
      api: Api::External,
      kind: if gpu { DeviceKind::IntegratedGpu } else { DeviceKind::Cpu },
      driver: runtime.into(),
      kernels: Kernels::External,
    }
  }
}
