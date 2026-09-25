# Clean & Analyze Social Media

Two Jupyter notebooks from a data-cleaning and analysis course: set up the Python
data stack, generate a social-media-style dataset, clean it, and chart it.

Owner / author: Juan J Serrano P (`022UGDW213`).

All facts, counts and versions in this README were measured on 2026-09-26 with the
commands shown — see [Verification log](#verification-log-2026-09-26).

---

## Contents

| File | What it is |
|---|---|
| `JupyterLab Clean & Analyze Social Media with Python.ipynb` | The analysis notebook: build a DataFrame, clean it, plot Likes, print per-category means. |
| `JupyterLab Prerequisites.ipynb` | Recorded `pip install` transcripts for the libraries the analysis notebook imports. **Not runnable as-is** — see below. |
| `How to Install Jupyter "UBUNTU".md` | Upstream install instructions for JupyterLab and Jupyter Notebook (pip, conda/mamba, Homebrew). |
| `README.md` | This file. |

Nothing else is in the repository — no data files, no `requirements.txt`, no scripts.
Both notebooks are kernel metadata `python3` / Python **3.10.12**, `nbformat` **4.5**.

---

## What the analysis notebook actually does

`JupyterLab Clean & Analyze Social Media with Python.ipynb` — **6 cells, all of type
`code`** (verified count below). Cell-by-cell:

| Cell | Code | What it does |
|---|---|---|
| 0 | `import pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `seaborn as sns`, `random` | Imports |
| 1 | `n = 500`; `pd.date_range('2021-01-01', periods=n)`; `random.choice(categories)`; `np.random.randint(0, 10000, size=n)` | **Generates** the dataset |
| 2 | `df.head()`, `df.info()`, `df.describe()`, `df['Category'].value_counts()` | Inspect the frame |
| 3 | `df.dropna(inplace=True)`, `df.drop_duplicates(inplace=True)`, `pd.to_datetime`, `.astype(int)` | "Cleaning" |
| 4 | `sns.histplot(...)`, `sns.boxplot(...)`, then mean of Likes overall and per category | Two charts (2 saved PNGs) + summary stats |
| 5 | *(empty)* | — |

The concrete shape of the generated frame:

- **8 categories**: `Food`, `Travel`, `Fashion`, `Fitness`, `Music`, `Culture`, `Family`, `Health`
- **500 rows**, dates `2021-01-01` → `2022-05-15`
- `Likes`: integers in the range **0–9999** (`np.random.randint(0, 10000, size=500)`)

The notebook's committed outputs contain a `describe()` over those 500 rows, a
`value_counts()` per category, and the overall `Likes` mean.

### The data is generated in the notebook — there is no dataset to download

This is the most important thing to know before reading the charts.

The notebook does **not** read any dataset. There is no URL and no `.csv`/`.json`
filename anywhere in either notebook, and no `read_csv` / `read_json` / `read_excel`
call (checked with `grep -rnE 'read_csv|read_json|https?://' *.ipynb`). The
"social media" rows are fabricated in-session by `random.choice` and
`np.random.randint`. So:

- **You do not need to supply any file** — the notebook is self-contained.
- **No real social-media metric is being analysed.** The Likes values are random
  integers; the per-category differences in cell 4 are noise, not a finding.

There is also **no random seed**, so every run produces a different frame. Two
actual runs of the identical committed code:

| Run | `Likes` mean | `Likes` min / max |
|---|---|---|
| Outputs saved in the committed notebook | **5031.07** | 15 / 9999 |
| Re-run on this workstation, 2026-09-26 | **4819.98** | 15 / 9962 |

Same code, different numbers — that is the missing seed, not a bug. If you want
reproducible figures, add `random.seed(...)` and `np.random.seed(...)` before cell 1
and re-run.

### The "cleaning" cell is a no-op on this generated data

Cell 3 runs `dropna()`, `drop_duplicates()`, `pd.to_datetime()` and `astype(int)`.
Measured against the frame cell 1 produces: **0 null values, 0 duplicate rows**, and
`Date` is already `datetime64[ns]` while `Likes` is already `int64`. The cell is
correct code, but on this synthetic frame it changes nothing. Its value would show on
a real, messy export — which this repository does not contain.

---

## What the prerequisites notebook actually does

`JupyterLab Prerequisites.ipynb` — **3 cells, all of type `code`**, with **0 saved
outputs** and `execution_count: null` for all three.

Each cell is a pasted transcript: the install command on the first line, then the real
pip log beneath it, all inside the cell *source*.

| Cell | First line | Packages recorded as installed in that transcript |
|---|---|---|
| 0 | `pip install pandas` | `numpy-1.26.4`, `pandas-2.2.1`, `tzdata-2024.1` |
| 1 | `pip install matplotlib` | `contourpy-1.2.1`, `cycler-0.12.1`, `fonttools-4.50.0`, `kiwisolver-1.4.5`, `matplotlib-3.8.3` |
| 2 | `pip install seaborn` | `seaborn-0.13.2` |

### This notebook does not execute

The pasted pip log includes the box-drawing progress bar character `━` (U+2501),
which is not valid Python, so each cell fails to compile:

```
$ jupyter nbconvert --to notebook --execute --output pre-exec.ipynb "JupyterLab Prerequisites.ipynb"
SyntaxError: invalid character '━' (U+2501)     # Cell In[1], line 6
$ echo $?
1
```

To make it runnable, keep the install commands and move the pip logs out of the cell
(or comment them out). Use Jupyter's shell escape for the command itself:

```python
%pip install pandas
```

A bare `pip` does resolve to the pip module inside IPython here (`pip --version` →
`pip 26.0.1`), but `pip install pandas` on its own line is still a Python syntax
error, so the escape prefix is required.

---

## Prerequisites

Derived from the analysis notebook's actual imports (cell 0): **pandas, numpy,
matplotlib, seaborn**. Python's stdlib `random` needs no install.

Verified present on this workstation, 2026-09-26:

| Tool | Real version | Command |
|---|---|---|
| Python | **3.10.12** (`/usr/bin/python3`) | `python3 -V` |
| pandas | 2.3.3 | `python3 -c "import importlib.metadata as m;print(m.version('pandas'))"` |
| numpy | 2.2.6 | same, `'numpy'` |
| matplotlib | 3.10.8 | same, `'matplotlib'` |
| seaborn | 0.13.2 | same, `'seaborn'` |
| JupyterLab | 4.5.6 (`/usr/local/bin/jupyter-lab`) | `jupyter-lab --version` |
| notebook | 7.5.5 | `jupyter --version` |
| nbconvert | 7.17.0 | `jupyter --version` |
| pip | 26.0.1 | `python3 -m pip --version` |

Note the split between *recorded* and *present* versions: the committed prerequisites
transcripts record pandas 2.2.1 / numpy 1.26.4 / matplotlib 3.8.3, while this machine
has newer ones. The analysis notebook's code runs unchanged on the newer versions
(exit code 0, see below). pip installs into
`/usr/local/lib/python3.10/dist-packages` here, which is writable, so no `--user`
flag or sudo is needed.

Run it:

```bash
jupyter lab          # then open "JupyterLab Clean & Analyze Social Media with Python.ipynb"
```

---

## Filename note

The committed filename `How to Install Jupyter "UBUNTU".md` contains ASCII double
quotes (U+0022). exFAT/FAT filesystems forbid `"` in a filename, so git cannot
recreate that name on such a checkout:

```
$ git checkout HEAD -- 'How to Install Jupyter "UBUNTU".md'
error: unable to create file How to Install Jupyter "UBUNTU".md: Invalid argument
```

The file's *content* is intact in git; only the on-disk name may be substituted
(e.g. a bullet-like character) on those filesystems. On ext4/APFS/NTFS the real name
checks out normally.

---

## Verification log (2026-09-26)

Every value above comes from these commands, run in this working tree on the machine
described. Reproduce them and you get the same answers.

```bash
# 1. Both notebooks are valid nbformat JSON — exit 0 for each
python3 -c "import json;json.load(open('JupyterLab Prerequisites.ipynb'))"
python3 -c "import json;json.load(open('JupyterLab Clean & Analyze Social Media with Python.ipynb'))"

# 2. Cell counts: 3 (all code) and 6 (all code)
python3 -c "
import json
from collections import Counter
for f in ['JupyterLab Prerequisites.ipynb',
          'JupyterLab Clean & Analyze Social Media with Python.ipynb']:
    c = json.load(open(f))['cells']
    print(f, len(c), dict(Counter(x['cell_type'] for x in c)))
"

# 3. The analysis notebook runs end to end — exit 0
jupyter nbconvert --to notebook --execute --output main-exec.ipynb \
  "JupyterLab Clean & Analyze Social Media with Python.ipynb"

# 4. The prerequisites notebook does not run — SyntaxError invalid character U+2501, exit 1
jupyter nbconvert --to notebook --execute --output pre-exec.ipynb \
  "JupyterLab Prerequisites.ipynb"

# 5. No external dataset is referenced anywhere
grep -rnE 'read_csv|read_json|read_excel|https?://' *.ipynb

# 6. The generated frame's real shape
python3 -c "
import pandas as pd
d = pd.date_range('2021-01-01', periods=500)
print(len(d), d[0].date(), d[-1].date())
"
# -> 500 2021-01-01 2022-05-15
```

Known limitations, stated plainly: this repository contains **no real social-media
dataset**, so no real analysis result can be reported — the numbers in the notebook are
random by construction. The prerequisites notebook is a recorded transcript, not a
runnable installer. Nothing here has been presented as more than that.
