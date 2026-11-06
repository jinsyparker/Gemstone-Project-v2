# Efficiency measurement protocol

One protocol for every model (`code/efficiency.py`; results in `results/efficiency.csv`, machine record in
`results/efficiency_machine.json`).

- **Input:** one 3 x 224 x 224 float32 image (batch 1) or a batch of 32; random values, since the architectures have no
  data-dependent control flow. Weights do not matter for timing, so models are built without pretrained weights.
- **Parameters:** all parameters; the table splits them into pretrained (ImageNet backbone) and new (colour/texture branch,
  projections, gates, head).
- **FLOPs:** `torch.utils.flop_counter.FlopCounterMode`, counting 2 x multiply-accumulates of convolutions and matrix
  products for one image. Element-wise operations, normalisation, pooling and the colour/texture map arithmetic (HSV, Sobel,
  Laplacian) are not counted, so the proposed model's branch B is slightly understated.
- **Latency:** per-image milliseconds, median and inter-quartile range over 200 timed forward passes after 20 warm-up passes,
  `torch.inference_mode()`, eval mode, fp32. GPU timings synchronise after every pass.
  - Apple M1 GPU (MPS), batch 1 and batch 32.
  - CPU, batch 1, 4 threads.
- **Conditions:** nothing else running (the script refuses to run while training or evaluation processes exist), the Mac
  plugged in, run once at the end of the project, never during training. Python, PyTorch and the OS version are recorded.
- Numbers from this table are comparable with each other only; they are not comparable with timings from other hardware or
  with the earlier single-number "inference time" column of `results/benchmark.csv`, which was taken while other work ran.
