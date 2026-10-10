# LLM Performance Benchmarking CLI Tool

A production-grade, modular Python CLI application built to systematically benchmark and compare the inference latency, throughput, memory consumption, and generation quality of open-source Large Language Models (LLMs) from Hugging Face.

---

## 🎯 Architecture & Workflow

The tool follows a decoupled three-tier architecture:

```
[ User Interface ]
       │  --config config.yaml
       ▼
[ Configuration & Dataset Engine ]
       │  Validated models & standardized prompts
       ▼
[ Core Benchmarking Engine ]
       ├── Hugging Face Model Runner (Dynamic loading, CPU/CUDA device management)
       ├── Hardware Profiler (psutil RSS RAM sampling, PyTorch CUDA peak VRAM)
       └── Quality Engine (Lexical Diversity / Type-Token Ratio)
       │
       ▼
[ Reporting & Persistence Layer ]
       ├── outputs/results.csv (Full per-prompt telemetry audit log)
       ├── outputs/*.png (Comparative visualizations via Matplotlib)
       └── Terminal Output (Formatted Markdown/ASCII summary table)
```

---

## 🚀 Features

- **Dynamic Model Loading**: Evaluates any Causal LM from Hugging Face (`gpt2`, `facebook/opt-125m`, `TinyLlama`, etc.) via configuration—no hardcoded model names.
- **Hardware-Aware Placement**: Automatically uses NVIDIA CUDA if a compatible GPU is detected; falls back gracefully to CPU.
- **Accurate Telemetry**:
  - **End-to-End Latency**: High-precision wall-clock timing using `time.perf_counter()`.
  - **Throughput**: Measured in Tokens-Per-Second (TPS).
  - **RAM Peak Monitoring**: Dedicated background thread sampling Resident Set Size (RSS) via `psutil`.
  - **VRAM Peak Tracking**: Resetting and recording peak CUDA allocations via `torch.cuda.max_memory_allocated()`.
- **Quality Verification**: Computes the Type-Token Ratio (TTR) and flags empty/degenerate generation loops to differentiate fast degenerate models from coherent ones.
- **Memory Isolation**: Sequentially frees models, triggers garbage collection (`gc.collect()`), and purges CUDA cache (`torch.cuda.empty_cache()`) between runs to prevent Out-Of-Memory crashes.
- **Reproducible Containerization**: Pre-configured `Dockerfile` and `docker-compose.yml` for isolated cross-platform execution.

---

## 📁 Repository Structure

```
llm-benchmark/
├── src/
│   ├── __init__.py           # Package marker
│   ├── cli.py                # CLI entry point & orchestrator (argparse + tqdm)
│   ├── config_parser.py      # Schema validation for YAML/JSON configurations
│   ├── dataset_loader.py     # Prompt ingestion (.jsonl, .csv, .json)
│   ├── model_runner.py       # Hugging Face model & tokenizer abstraction
│   ├── profiler.py           # Latency, memory sampler, and quality metric engine
│   └── reporter.py          # CSV persistence, terminal summary & Matplotlib charts
├── datasets/
│   └── sample_prompts.jsonl  # Diverse test prompts across reasoning, coding, and general tasks
├── outputs/                  # Generated benchmark artifacts
│   ├── results.csv           # Detailed per-prompt telemetry
│   ├── latency_comparison.png# Bar chart: Average Latency
│   ├── memory_comparison.png # Bar chart: Peak RAM / VRAM
│   └── throughput_comparison.png # Bar chart: Generation Throughput
├── tests/
│   ├── test_config.py        # Config parsing & dataset loader tests
│   └── test_profiler.py      # Profiler, metrics, and reporter tests
├── config.yaml               # Default benchmark configuration
├── config.json               # JSON equivalent configuration
├── .env.example              # Environment variables template
├── Dockerfile                # Container definition
├── docker-compose.yml        # Multi-container / volume orchestration
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # Project documentation & results analysis
```

---

## 🛠️ Quickstart & Setup

### 1. Local Setup (Virtual Environment)

```powershell
# Clone the repository
git clone https://github.com/vikram0678/llm-benchmark.git
cd llm-benchmark

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the template:
```bash
cp .env.example .env
```
Key variables:
- `HF_HOME`: Directory to cache downloaded model weights (defaults to local `.cache`).
- `HF_TOKEN`: Hugging Face user token (required only for gated models).

---

## 💻 Running the Benchmark

### Validate Configuration (Dry Run)
Verify models, parameters, and dataset accessibility without downloading weights:
```bash
python src/cli.py --config config.yaml --dry-run
```

### Execute Full Benchmark
```bash
python src/cli.py --config config.yaml
```

You can also run with JSON configurations:
```bash
python src/cli.py --config config.json
```

---

## 🐳 Docker Execution

Run the entire pipeline in an isolated container with cached model volumes:

```bash
# Build and run via docker-compose
docker-compose up --build
```

Or run standalone using Docker:
```bash
docker build -t llm-benchmark .
docker run --rm -v ${PWD}/outputs:/app/outputs llm-benchmark
```

---

## 📊 Benchmark Findings & Analysis

During our standardized benchmark run on 3 diverse prompts, the following results were recorded:

### Summary Table

| Model | Avg Latency | Throughput | Peak RAM | Peak VRAM | Lexical TTR | Degenerate % | Coherence Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`distilbert/distilgpt2`** | **2.007 s** | **15.31 TPS** | 1073.9 MB | 0.0 MB | 0.000 | 100.0% | ❌ Degenerate / Empty |
| **`facebook/opt-125m`** | 4.910 s | 10.19 TPS | 915.0 MB | 0.0 MB | **0.589** | **0.0%** | ✅ Coherent & Varied |
| **`gpt2`** | 5.219 s | 9.63 TPS | **870.0 MB** | 0.0 MB | 0.554 | **0.0%** | ✅ Coherent & Accurate |

### Key Takeaways:
1. **Speed vs. Coherence Trade-off**: `distilgpt2` exhibited the highest raw generation throughput (15.31 TPS) and lowest latency (2.00s). However, its Type-Token Ratio was 0.0 and it was flagged as 100% degenerate. This proves why quality metrics are essential in LLM benchmarking.
2. **Opt-125m vs. GPT-2**: `facebook/opt-125m` offered a ~6% higher throughput (10.19 TPS vs 9.63 TPS) and higher lexical diversity (0.589 vs 0.554) compared to `gpt2`, while requiring only ~45 MB additional peak RAM.
3. **Resource Isolation**: Explicit memory purges (`gc.collect()` and `cuda.empty_cache()`) prevented memory leaks across consecutive model evaluations.

---

## 🧪 Running Unit Tests

Run the test suite using Python's built-in test runner:

```bash
python -m unittest discover tests -v
```

All 6 unit tests validate schema constraints, prompt extraction, profiler calculations, and report generation.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
