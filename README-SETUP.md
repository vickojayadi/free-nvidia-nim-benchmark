# NVIDIA NIM Benchmark Dashboard - Setup

## 1. Create the repository
Create a new **public** GitHub repository, for example:

`nvidia-nim-benchmark`

Upload these files while keeping the folder structure intact:

- `nim_benchmark.py`
- `test_nim_benchmark.py`
- `index.html`
- `.github/workflows/benchmark.yml`

## 2. Add the NVIDIA API key
In the repository open:

`Settings -> Secrets and variables -> Actions -> New repository secret`

Create:

- Name: `NVIDIA_API_KEY`
- Secret: your `nvapi-...` key

Never put the API key directly in Python, HTML, YAML, commits, or README files.

## 3. Enable GitHub Pages
Open:

`Settings -> Pages`

Under **Build and deployment**, set:

`Source -> GitHub Actions`

## 4. Run it manually the first time
Open:

`Actions -> NVIDIA NIM Benchmark -> Run workflow`

The workflow will:

1. test the Python benchmark;
2. benchmark the NVIDIA NIM endpoints;
3. create `latest.json`;
4. publish `index.html` + `latest.json` to GitHub Pages.

## 5. Open the dashboard
After the first successful deployment, GitHub shows the Pages URL in:

`Settings -> Pages`

A project repository normally has a URL shaped like:

`https://YOUR-USERNAME.github.io/nvidia-nim-benchmark/`

## 6. Hourly schedule
The workflow uses:

`17 * * * *`

This means once per hour at minute 17 (UTC scheduling semantics do not matter for an every-hour schedule).

The benchmark starts with **2 runs per model**, `192` max tokens, and a `1.5` second pause. This is intentionally conservative for shared free endpoints.

To increase to 3 runs later, edit:

`--runs 2`

to:

`--runs 3`

## What the dashboard means

- **Score**: relative local performance score for that exact benchmark run.
- **Uptime**: successful benchmark requests / attempted requests. It is not NVIDIA's official SLA.
- **Intelligence**: external reference score available in the benchmark script; it is not measured by GitHub Actions.
- **Avg time**: average total streamed response time.
- **TTFT**: time until the first generated token appears.
- **Best time**: fastest completed request in that benchmark run.
- **Throughput**: output tokens generated per second when the endpoint reports usage.

## Important limitation
The measurement is from the GitHub-hosted runner used by that workflow run to NVIDIA NIM. GitHub-hosted runners are ephemeral, so use the dashboard for comparative trends rather than as a production latency SLA.
