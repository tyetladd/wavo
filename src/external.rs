// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! An encoder that runs outside wavo, for [`Model::load_external`](crate::Model::load_external):
//! the caller's own runtime (MNN, LiteRT or another) runs GigaAM's, Parakeet's or Whisper's
//! encoder, and wavo keeps the frontend, the decoder, the splitting at pauses and the token times.

use crate::gpu::Device;

/// A family's encoder in another runtime, exactly as wavo's CPU encoder computes it. GigaAM's
/// (`src/gigaam/cpu.rs`): the two stride-2 conv1d + ReLU subsampling layers, the Conformer blocks
/// and the head's linear layer. Parakeet's (`src/parakeet/cpu.rs`): the 8× conv2d subsampling,
/// the Conformer blocks with relative positions and the joint's encoder projection. Whisper's
/// (`src/whisper/cpu.rs`): a 30 s window of mel `[3000][mels]` through the two conv1d + GELU, the
/// learned positions, the blocks and the final LayerNorm, returning `[1500][d]`, which wavo's CPU
/// decoder attends to. wavo calls it from the thread that calls
/// [`Model::transcribe`](crate::Model::transcribe), one call at a time.
pub trait Encoder: Send + Sync {
  /// Encodes `mel`, time-major `[frames][mels]` (wavo's log-mel features), and returns the head's
  /// outputs time-major `[t][width]`, where `t` is `frames` halved twice by the stride-2 convs
  /// (`(n − 1) / 2 + 1` each time) and `width` is the joint's hidden size for an RNN-T head
  /// (`stt.gigaam.joint.hidden`) or the classes for a CTC head. A failure's message reaches the
  /// caller as [`Error::Encoder`](crate::Error::Encoder).
  fn encode(&self, mel: &[f32], mels: usize) -> std::result::Result<Vec<f32>, String>;

  /// [`encode`](Self::encode) with the mel frames from `valid` on as padding: Parakeet calls this,
  /// and a model with `length_masking` `kestrel` passes one frame fewer than it has. Its
  /// subsampling zeroes conv0's output rows from ⌈`valid` / 2⌉ on and conv2's from ⌈`valid` / 4⌉
  /// on, as `src/parakeet/cpu.rs` does; the outputs are the joint's encoder projections
  /// `[t][joint]`, `t` the frames halved three times. The default ignores `valid`, which is exact
  /// only when it is every frame.
  fn encode_masked(
    &self,
    mel: &[f32],
    mels: usize,
    valid: usize,
  ) -> std::result::Result<Vec<f32>, String> {
    let _ = valid;
    self.encode(mel, mels)
  }

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
