# Stochastic Parrots (Llama 2 in C)

C inference engine for quantized Llama 2 checkpoints based on Andrej Karpathy's `llama2.c`, extended with custom implementations of the core math routines (`dynamics.c`), unit tests (`tests/public_tests.c`), and a token-level sequence log-probability analyzer (`analyze.c`).

## Implemented Components

- **Tensor and normalization primitives (`dynamics.c`)**: `sqnorm`, `rms`, `rmsnorm`, numerically stable `max_val` and `softmax`, and int8 group-wise quantized `matmul`.
- **Generation and chat CLI (`parrot.c`, `runq.c`, `sampler.c`)**: Supports `generate` and `chat` modes with temperature (`-t`), seed (`-s`), step count (`-n`), prompt (`-i`), and system prompt (`-y`).
- **Log-probability analyzer (`analyze.c`)**: Computes and compares $\sum_i \log P(t_i \mid t_{<i}, \text{prompt})$ across candidate continuations token by token.
- **Logit plotting (`plots/plot_logits.plt`)**: Gnuplot script to visualize token probability distributions from `results/logits.txt`.

## Structure

```text
├── parrot.c                # CLI entry point (generate / chat)
├── analyze.c               # Candidate sequence log-probability comparison
├── dynamics.c              # RMSNorm, softmax, and quantized matmul
├── transformer.c           # Transformer forward pass and weight loading
├── runq.c                  # Generation and chat loops
├── sampler.c               # Top-p and temperature sampling
├── tokenizer.c             # BPE tokenizer loader/decoder
├── utils.c / head.h        # Shared structs and helpers
├── win.c / win.h           # Windows mmap and clock compatibility
├── models/tokenizer.bin    # Llama 2 tokenizer vocabulary
├── results/                # Sample logit outputs
├── plots/plot_logits.plt   # Gnuplot script
├── tests/public_tests.c    # Unit tests for math and sampling
└── Makefile
```

## Build & Run

```bash
make all
make test
```
*(Use `mingw32-make` on Windows with MinGW.)*

Place a compatible quantized Llama 2 `.bin` checkpoint (ignored by Git due to size) in `models/`, e.g. `stories110M.bin` or `tinyLlama.bin`, then run:

```bash
# Text generation
./parrot models/stories110M.bin -m generate -i "Math is boring, physics is much" -t 0 -n 47

# Chat mode
./parrot models/tinyLlama.bin -m chat -i "Hello!" -y "You are a helpful assistant."

# Log-probability comparison
./analyze
```

## Credits

Built on top of `llama2.c` (Andrej Karpathy) for Computational Physics coursework (`computacional-unizar`), with custom implementations of the math primitives (`dynamics.c`) and sequence log-probability evaluation (`analyze.c`).
