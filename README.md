# Code-DCR

This is the supplementary script that goes with the paper. The layout is minimal: `code_dcr.py`, `LICENSE`, `requirements.txt`, plus this file.

## What you need

Use Python 3.10+. Install `torch` with wheels that match your machine (and CUDA build if you need the GPU). The selector at [pytorch.org](https://pytorch.org) is the least painful way. After `torch` looks good, `pip install -r requirements.txt` pulls the rest.

## Run it

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python code_dcr.py
```

On Linux or macOS swap the activate line for `source .venv/bin/activate`.

First run hits the network: it pulls dolfinx demo files into `data/dolfinx_demos/` (gitignored).

Training writes checkpoint files `codecino_ep*.pt` in the current working directory.

## Logging

By default the script stays quiet. To print progress and metrics:

```text
set CODECINO_VERBOSE=1
python code_dcr.py
```

(PowerShell: `$env:CODECINO_VERBOSE = "1"`)

Download or OOD quirks still emit `warnings` on stderr unless you filter them.

## License

AGPL-3.0-or-later. Full text in `LICENSE`. Header in `code_dcr.py` has SPDX lines for reuse tooling.

**Research Paper:** [Download Code-DCR Preprint (PDF)](./Code_DCR.pdf)  
*Note: This architecture is currently undergoing arXiv endorsement/submission for the cs.AI category.*

**© 2026 Sridhar Mani. All Rights Reserved.** *(The AGPL-3.0 license below applies to the code. The PDF manuscript is protected prior art).*
