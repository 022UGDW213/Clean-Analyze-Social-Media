# Clean & Analyze Social Media

Two Jupyter notebooks, both about a synthetic social-media-style dataset in Python:
transcribe the `pip install` steps for the data stack, generate the dataset,
"clean" it, and chart it.

Owner: GitHub account `022UGDW213` (profile display name "Time Loops", id 85779081).
Remote: `git@github.com:022UGDW213/Clean-Analyze-Social-Media.git` (`git remote -v`).
The 2026-09-27 capture of `curl -s https://api.github.com/users/022UGDW213` reports
`"login": "022UGDW213"`, `"name": "Time Loops"`, `"id": 85779081`. The personal name
"Juan J Serrano P" used in this project's `AGENTS.md` is **not** verifiable from this
repository or from that payload.

Repository metadata from the same 2026-09-27 capture
(`curl -s https://api.github.com/repos/022UGDW213/Clean-Analyze-Social-Media`):
`"private": false`, `"fork": false`, `"default_branch": "main"`,
`"language": "Jupyter Notebook"`, `"stargazers_count": 1`, `"watchers_count": 1`,
`"forks_count": 0`, `"open_issues_count": 0`, `"size": 177`, `"license": null`,
created `2024-04-03T06:12:01Z`, last push `2026-09-25T21:00:37Z`. At that capture the
account itself had **57 public repos** (14 own + 43 forks), **26 stars** across all repos
(**20** on its own), **19 followers**, **168 following** and **3 public gists** —
counted directly from `https://api.github.com/users/022UGDW213/repos?per_page=100`.

All facts, counts and versions in this README were measured on 2026-09-27 with the
commands shown — see [Verification log](#verification-log-2026-09-27).

---

## Contents

| File | What it is |
|---|---|
| `JupyterLab Clean & Analyze Social Media with Python.ipynb` | The analysis notebook: build a DataFrame, clean it, plot Likes, print per-category means. |
| `JupyterLab Prerequisites.ipynb` | Recorded `pip install` transcripts for the libraries the analysis notebook imports. **Not runnable as-is** — see below. |
| `How to Install Jupyter (UBUNTU).md` | Third-party install instructions: an excerpt of the **JupyterLab** and **Jupyter Notebook** sections of [jupyter.org/install](https://jupyter.org/install) (the page's Voilà and Homebrew sections are not included). Not the author's own text. |
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
| Re-run on this workstation, 2026-09-27 | **4738.42** | 2 / 9914 |

Same code, different numbers — that is the missing seed, not a bug. `grep -n seed *.ipynb`
returns nothing, so no seed is set anywhere. If you want reproducible figures, add
`random.seed(...)` and `np.random.seed(...)` before cell 1 and re-run.

The committed-output row is read straight out of the notebook, so it is stable; the
re-run row changes every time and is only a sample. Both are reproducible with the
commands in the verification log.

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

A bare `pip` line does execute in IPython here — `ipython -c "pip --version"` prints
`pip 26.0.1 from /usr/local/lib/python3.10/dist-packages/pip (python 3.10)` and exits
0 (IPython's line-magic automagic), and the committed transcript's own output shows
`pip install pandas` running in that kernel. With automagic off, `pip install pandas`
on its own line is a Python syntax error, so `%pip` is the portable form.

---

## Prerequisites

Derived from the analysis notebook's actual imports (cell 0): **pandas, numpy,
matplotlib, seaborn**. Python's stdlib `random` needs no install.

Verified present on this workstation, 2026-09-27:

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
(exit code 0, see below).

pip installs into `/usr/local/lib/python3.10/dist-packages` here. It is writable
**because the commands above were run as root (uid 0)** — `test -w` and a `touch` both
succeed — so no `--user` flag is needed. A normal user gets the
`Defaulting to user installation because normal site-packages is not writeable` line
that the committed transcript recorded.

Run it:

```bash
jupyter lab          # then open "JupyterLab Clean & Analyze Social Media with Python.ipynb"
```

---

## Filename and third-party content note

`How to Install Jupyter (UBUNTU).md` is **third-party text, not the author's own**: it
is an excerpt of the *JupyterLab* and *Jupyter Notebook* install sections of
<https://jupyter.org/install>. The page's Voilà and Homebrew sections are not included.
The URL `https://jupyter.org/install` is on the file's last line
(`grep -n 'jupyter.org' 'How to Install Jupyter (UBUNTU).md'` → line 36), which is what
identifies the source.

Until 2026-09-27 the file was committed as `How to Install Jupyter "UBUNTU".md`. ASCII
double quotes (U+0022) are illegal in a FAT/exFAT filename, so git cannot recreate that
name on an exFAT checkout:

```
$ git checkout HEAD -- 'How to Install Jupyter "UBUNTU".md'
error: unable to create file How to Install Jupyter "UBUNTU".md: Invalid argument
$ echo $?
255
```

The exFAT driver silently substituted U+F022 (a private-use character) for each `"`.
That left the tree permanently dirty — `git status` reported the tracked name as deleted
plus an untracked file named `How to Install Jupyter <U+F022>UBUNTU<U+F022>.md` whose
content was byte-identical to HEAD. To end that, the file was renamed to the portable
ASCII name `How to Install Jupyter (UBUNTU).md` on 2026-09-27. The content was not
touched (`git diff --cached --stat -M` reports 0 insertions / 0 deletions for the
rename); the old name remains in the history (`git log --follow`).

---

## Verification log (2026-09-27)

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
jupyter nbconvert --to notebook --execute --output /tmp/main-exec.ipynb \
  "JupyterLab Clean & Analyze Social Media with Python.ipynb"

# 4. The prerequisites notebook does not run — SyntaxError invalid character U+2501, exit 1
jupyter nbconvert --to notebook --execute --output /tmp/pre-exec.ipynb \
  "JupyterLab Prerequisites.ipynb"

# 5. No external dataset is referenced anywhere — grep exits 1 (no match)
grep -rnE 'read_csv|read_json|read_excel|https?://' *.ipynb

# 6. The generated frame's real shape
python3 -c "
import pandas as pd
d = pd.date_range('2021-01-01', periods=500)
print(len(d), d[0].date(), d[-1].date())
"
# -> 500 2021-01-01 2022-05-15

# 7. "Cleaning" is a no-op on the generated frame: 0 nulls, 0 duplicates, dtypes already right
python3 -c "
import pandas as pd, numpy as np, random
cats = ['Food','Travel','Fashion','Fitness','Music','Culture','Family','Health']
df = pd.DataFrame({'Date': pd.date_range('2021-01-01', periods=500),
                   'Category': [random.choice(cats) for _ in range(500)],
                   'Likes': np.random.randint(0, 10000, size=500)})
print('nulls', int(df.isnull().sum().sum()), 'dups', int(df.duplicated().sum()),
      dict(df.dtypes.astype(str)))
"
# -> nulls 0 dups 0 {'Date': 'datetime64[ns]', 'Category': 'object', 'Likes': 'int64'}

# 8. No random seed anywhere in either notebook — grep exits 1
grep -n seed *.ipynb

# 9. Likes mean / min / max as saved in the committed outputs
python3 -c "
import json
nb = json.load(open('JupyterLab Clean & Analyze Social Media with Python.ipynb'))
txt = ''.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]))
print([l.strip() for l in txt.splitlines()
       if l.strip().startswith(('mean','min','max')) or \"Mean of 'Likes'\" in l])
"
# -> ['mean   2021-09-07 12:00:00  5031.068000', 'min    2021-01-01 00:00:00    15.000000',
#     'max    2022-05-15 00:00:00  9999.000000', "Mean of 'Likes': 5031.07"]

# 10. The versions in the table above
python3 -V
python3 -m pip --version
jupyter-lab --version
python3 -c "
import importlib.metadata as m
for p in ['pandas','numpy','matplotlib','seaborn','notebook','nbconvert']:
    print(p, m.version(p))
"

# 11. exFAT cannot recreate the old quoted filename — exit 255
git checkout HEAD -- 'How to Install Jupyter "UBUNTU".md'
# error: unable to create file How to Install Jupyter "UBUNTU".md: Invalid argument
```

Known limitations, stated plainly: this repository contains **no real social-media
dataset**, so no real analysis result can be reported — the numbers in the notebook are
random by construction. The prerequisites notebook is a recorded transcript, not a
runnable installer. Nothing here has been presented as more than that.

## Not implemented / not present

There is nothing else claimed and nothing pretending to work:

- **No real dataset and no download step** — the frame is generated in-session by
  `random.choice` / `np.random.randint`, so the per-category differences carry no
  meaning.
- **No random seed** — the analysis notebook is not reproducible run-to-run. To fix,
  add `random.seed(...)` and `np.random.seed(...)` before cell 1.
- **No `requirements.txt`, `environment.yml`, packaging, CI or tests** — nothing to run
  but the two notebooks. The four files listed under Contents are the whole repository.
- **`JupyterLab Prerequisites.ipynb` does not execute** — it is a pasted transcript, and
  its recorded versions (pandas 2.2.1 / numpy 1.26.4 / matplotlib 3.8.3) are older than
  what is installed on this machine today (pandas 2.3.3 / numpy 2.2.6 / matplotlib 3.10.8).
- **No original documentation** — `How to Install Jupyter (UBUNTU).md` is third-party
  text from jupyter.org; only `README.md` is the author's own writing.
