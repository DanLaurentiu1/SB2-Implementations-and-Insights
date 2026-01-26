<a id="readme-top"></a>

<p align="center">
  <b><font size="7">SB2: Implementations & Insights</font></b><br/>
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python Version">
  <img src="https://img.shields.io/github/issues/DanLaurentiu1/SB2-Implementations-and-Insights" alt="Issues">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/docker-ready-lightblue" alt="Docker">
  <img src="https://img.shields.io/badge/status-active-success" alt="Docker">
</p>

<div align="center">
<p>
    <a href="#about">About</a> •
    <a href="#getting-started-local-setup">Getting Started</a> •
    <a href="#usage">Usage</a> •
    <a href="#project-structure">Project Structure</a> •
    <a href="https://github.com/users/DanLaurentiu1/projects/1">Roadmap</a>
</p>
</div>

<p align="center">
  <img src="assets/grass.png" width="100%">
</p>

## About
In this repository, you will find clean, modular implementations of algorithms from the [Sutton & Barto: Reinforcement Learning (2nd Edition)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) textbook. Reproductions of exercises, plot and tables are provided, along with simple ablation reports, small extensions to some algorithm's modules and, for the most popular concepts, theoretical proofs.

The code in this project is centered around modularization and ease-of-use. Everything runs (slowly) on the CPU, no GPU modifications and optimizations. All of the modules are fully tested, with integrations between modules (agent-environment) mostly tested through fully deterministic mocked runs.

The repository is structured around a four-stage synthesis of my learning process:
* **Implementation** – translating the ideas and concepts into functional code
* **Reproduction** – verifying those ideas by re-creating established benchmarks
* **Analysis** – experimenting through ablations and proofs
* **Extension** – developing (small and simple) enhancements

<p align="center">
  <img src="assets/grass.png" width="100%">
</p>

## Getting Started (Local Setup)
If you prefer to run the project natively. This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Prerequisites
* **Python 3.11+**
* **Poetry** (Install via `pip install poetry` or follow the [official guide](https://python-poetry.org/docs/#installation))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DanLaurentiu1/SB2-Implementations-and-Insights.git
   cd SB2-Implementations-and-Insights
   ```

2. **Install dependencies**
    ```bash
    poetry install
    ```

3. **Verify the setup by running tests**
    ```bash
    poetry run pytest tests/
    ```

<p align="center">
  <img src="assets/grass.png" width="100%">
</p>

## Usage
This repository allows you to run algorithms in two modes: Standard (training/testing) and Reproduction (benchmarks). 

### Execution Modes & Storage
| Mode | Purpose | Logs Location | Artifact Location |
| :--- | :--- | :--- | :--- |
| **Standard** | Training & testing | `[algorithm]/experiments/[run_name]` | N/A |
| **Reproduction** | Textbook figures/tables | `[algorithm]/experiments/[run_name]` | `[algorithm]/reproducibility/[repro_folder_name]` |

> **Note:** The `run.name` (experiment folder) and `repro_folder_name` do not need to match.

<br/>

### 1. Standard Execution
Run the default multi-armed-banits algorithm using the default configuration.

```bash
poetry run python -m algorithms.bandits.run
```

<br/>

### 2. Reproduction Execution
To save reproduction artifacts (plots, tables), you must append the `repro_folder_name`.

```bash
poetry run python -m algorithms.bandits.run_reproducibility +repro_folder_name=new_folder
```

<br/>

### 3. Overriding Configuration Columns
By default, Hydra loads `[algorithm]/configs/default.yaml`. You can override specific values (like the run name) directly from the CLI:

```bash
poetry run python -m algorithms.bandits.run run.name=new-run-name
```

> **Recommended:** The `run.name` option defines the experiment directory where logs will be saved. Always change `run.name`.

<br/>

### 4. Overriding Whole Configs
To use a completely different configuration file:

```bash
poetry run python -m algorithms.bandits.run --config-name=different_config
```

<br/>

### 5. Multi-runs (Parameter Sweeps)
Hydra supports multi-runs using the cartesian product of all overridden parameters.

```bash
# Example: Running 2 algorithms for 2000 seeds each (4000 total runs)
python -m algorithms.bandits.run -m run.env_seed=range(1,2001) algorithm=algo_A,algo_B run.name=multirun
```

> **Important**: The `-m` flag must be placed exactly before the parameters.


<br/>

### 6. Docker Execution
The Dockerfile utilizes an `entrypoint.sh` script to parse arguments. <br/>

The syntax is: `docker compose run --rm <service_name> <algorithm_name> [mode] [overrides]`

* **Standard Execution Mode**: `n` or `normal`
* **Reproduction Execution Mode**: `r` or `repro`

```bash
docker compose run --rm sim bandits normal run.env_seed=range(1,2001) algorithm=algo_A,algo_B
```

> **Important**: `<algorithm_name>` must match the directory structure.

<br/>

_For more detailed examples, visit the `[algorithm]/examples/` directories._

<p align="center">
  <img src="assets/grass.png" width="100%">
</p>

## Project Structure

```
.
├── algorithms/
│   ├── make_agent.py                # factory for instantiating algorithm from hydra config
│   └── [algorithm_name]/            # algorithm module
│       ├── run.py                   # main entry point for the algorithm
│       ├── run_reproducibility.py   # wrapper that creates reproduction artifacts
│       ├── configs/                 # hydra configuration files
│       ├── examples/                # usage examples
│       ├── implementations/
│       ├── proofs/
│       ├── reports/                 # ablation documents
│       ├── reproducibility/         # artifacts for each exercise/plots
│       └── utils/                   # some helpers
├── environments/
│   ├── make_env.py                  # factory for instantiating environment from hydra config
│   └── custom_envs/
│       └── [environment_name]/      # environment module
├── tests/
├── utils/                           # project-wide utilities
└── Dockerfile                       # container environment definition
```

<p align="right">
    <a href="#readme-top">
        <img width="40rem" src="assets/top-up.png" alt="Back to top">
    </a>
</p>