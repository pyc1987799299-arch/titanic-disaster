# Titanic survival prediction

Homework 3 — Introduction to Data Engineering, Northwestern University MLDS.

This project loads the Titanic data, prints a few descriptive findings, fills
missing values, trains a logistic regression model, evaluates it, and exports
survival predictions. Python and R run in separate Docker containers. All
processing steps and accuracy results are printed in the terminal.

**Data are not included.** Download the competition files as described below.
Do not commit the dataset, credentials, environments, or generated results.

## 1. Prerequisites and repository

Install [Git](https://git-scm.com/downloads) and
[Docker Desktop](https://docs.docker.com/desktop/). Start Docker Desktop and use
Linux containers. On Linux, Docker Engine is sufficient. Python and R do not
need to be installed on the host when using Docker. Internet access is needed
for downloading the dataset and building the images for the first time.

Clone the public repository:

```text
git clone https://github.com/pyc1987799299-arch/titanic-disaster.git
cd titanic-disaster
docker version
```

`docker version` must show both Client and Server information. Run all remaining
commands from the `titanic-disaster` repository root. If you already have this
local repository, open a terminal in its directory and skip cloning.

## 2. Download the data

1. Open the [Kaggle Titanic data page](https://www.kaggle.com/competitions/titanic/data).
2. Sign in to Kaggle and join the competition if prompted. Download the data.
3. Extract `train.csv` and `test.csv` from the downloaded archive.
4. Create `src/data` and copy both files directly into it, with these exact names.

Create the folders in **Windows PowerShell**:

```powershell
New-Item -ItemType Directory -Force -Path src/data,results/python,results/r | Out-Null
```

Or in **macOS/Linux Bash**:

```bash
mkdir -p src/data results/python results/r
```

The expected inputs are `src/data/train.csv` (891 passengers, with `Survived`)
and `src/data/test.csv` (418 passengers, without `Survived`).
`gender_submission.csv` is not needed: it contains example predictions, not
true test labels. Git ignores the entire `src/data` folder, so the folder must
be recreated after cloning. Docker also excludes data from its build context.

## 3. Build and run both containers

Build the images; the final `.` is required and means the repository root:

```text
docker build -f src/python/Dockerfile -t titanic-python .
docker build -f src/r/Dockerfile -t titanic-r .
```

**Windows PowerShell** — run each command on one line:

```powershell
docker run --rm --mount "type=bind,source=$((Get-Location).Path)/src/data,target=/app/data,readonly" --mount "type=bind,source=$((Get-Location).Path)/results/python,target=/app/results" titanic-python
docker run --rm --mount "type=bind,source=$((Get-Location).Path)/src/data,target=/app/data,readonly" --mount "type=bind,source=$((Get-Location).Path)/results/r,target=/app/results" titanic-r
```

**macOS/Linux Bash**:

```bash
docker run --rm --mount "type=bind,source=$(pwd)/src/data,target=/app/data,readonly" --mount "type=bind,source=$(pwd)/results/python,target=/app/results" titanic-python
docker run --rm --mount "type=bind,source=$(pwd)/src/data,target=/app/data,readonly" --mount "type=bind,source=$(pwd)/results/r,target=/app/results" titanic-r
```

The inputs are mounted read-only. Results are saved on the host even after
`--rm` removes the finished container. Running again overwrites that language's
previous results. No data are downloaded automatically by either container.

Each language creates two files:

| File | Contents |
| --- | --- |
| `results/python/submission.csv` | Python's 418 test predictions |
| `results/python/metrics.csv` | Python's row counts and accuracy measurements |
| `results/r/submission.csv` | R's 418 test predictions |
| `results/r/metrics.csv` | R's row counts and accuracy measurements |

Both submissions contain only `PassengerId,Survived`, retain the original test
row order, and use integer predictions of 0 (did not survive) or 1 (survived).
The metrics distinguish fitting-subset accuracy, holdout validation accuracy,
a majority-class baseline, and full-training accuracy after the final refit.

## 4. Data processing and evaluation

The scripts print input sizes, missing-value counts, overall survival rate, and
survival rates grouped by sex and passenger class. In the training data, the
overall survival rate is approximately 38.38%; the descriptive rates are 74.20%
for female passengers and 18.89% for male passengers. First-, second-, and
third-class rates are approximately 62.96%, 47.28%, and 24.24%. These describe
this historical sample, not causal effects.

Both implementations use `Age`, `Fare`, `FamilySize`, `Pclass`, `Sex`, and
`Embarked`. `FamilySize = SibSp + Parch + 1`; if a component is missing, family
size is treated as missing. Numeric missing values use the fitting data's
median; categorical missing values use its most frequent value. `Name`,
`Ticket`, and `Cabin` are excluded. `PassengerId` is used only for splitting
and output, never as a model feature.

Evaluation uses the same deterministic, stratified split in both languages:
within each `Survived` class, order passengers by
`(PassengerId * 1103515245 + 42) mod 2147483648` (then by ID), and reserve the
first `floor(20% of that class)` rows for validation, with a minimum of one.
This produces **714 fitting rows and 177 holdout rows** for the competition
data. It is reproducible across languages and independent of CSV row order;
it is a fixed holdout, not cross-validation.

Preprocessing is learned on the 714 fitting rows before being applied to the
holdout. Python uses a scikit-learn pipeline, standardized numeric features,
one-hot categorical encoding, and L2-regularized logistic regression. R uses
`stats::glm(..., family = binomial())`, unscaled numeric features, and factor
contrasts, without regularization. Their predictions need not be identical.
Both classify at a probability threshold of 0.5. No threshold tuning or
hyperparameter search is performed on the holdout.

After evaluation, each script refits preprocessing and its model on all 891
labeled passengers and predicts the 418 test passengers. Reported full-training
accuracy is in-sample and should not be interpreted as generalization accuracy.

### Official Kaggle test scores

Assignment steps 18 and 24 request test-set accuracy. Kaggle does not supply
the true `Survived` values in `test.csv`, so **local official-test accuracy
cannot be computed**. The holdout accuracy is a separate local estimate,
explicitly labeled as validation accuracy.

Both generated prediction files were manually submitted to Kaggle. The
completed submissions page reported these scores (verified from the submitted
screenshots on November 8, 2026):

| Submitted file | Model | Kaggle score | Percentage |
| --- | --- | ---: | ---: |
| `submission_python.csv` | Python logistic regression | 0.76794 | 76.794% |
| `submission_r.csv` | R logistic regression | 0.77033 | 77.033% |

These are Kaggle-reported scores against hidden labels, not the 78.53% local
holdout accuracy. The scripts print the recorded external score only when the
generated CSV's file checksum matches the corresponding submitted file.
This is a lookup of a completed evaluation, not a new local accuracy calculation
or an automatic Kaggle request. If predictions or file formatting change, the
scripts instead ask for a new submission; they do not reuse the old score.
File checksums identify the submitted artifacts, not model inputs or labels.

To obtain a fresh score, open the competition's **Submit Prediction** page
and upload `results/python/submission.csv` or `results/r/submission.csv`
separately. Use the model name in the submission description to distinguish
them. No Kaggle submission is made automatically.

### Verified Docker results

| Measurement | Python | R |
| --- | ---: | ---: |
| Fitting-subset accuracy | 80.95% | 80.53% |
| Holdout validation accuracy | 78.53% | 78.53% |
| Majority-class holdout baseline | 61.58% | 61.58% |
| Full-training accuracy (in-sample) | 80.81% | 80.36% |
| Exported test predictions | 418 | 418 |

Both images were successfully built and run on Docker Desktop 4.98.0 with
Docker Engine 29.9.2, WSL 2, and Linux/amd64 containers. The containers use
Python 3.12.8 with the exact versions in `requirements.txt`, and R 4.4.2 with
its bundled `stats`, `utils`, and `tools` packages. Both exited successfully and wrote
the expected CSVs using the PowerShell commands in section 3.

The predictions match the earlier native Python and WebR checks exactly;
metrics agree within floating-point tolerance. Output columns, 418 passenger
IDs in original test order, binary labels, and unchanged input files were
verified. Running either image without a data mount exits with code 1 and a
clear missing-file message, confirming the dataset is not baked into it.

Initial verification data came from the
[Hands-On Machine Learning Titanic mirror](https://github.com/ageron/handson-ml2/tree/master/datasets/titanic)
and are excluded from version control. Both containers were then rerun using
the supplied `titanic.zip` competition files. Despite minor name and age
precision differences from the mirror, all predictions and reported metrics
were unchanged. Those predictions were used for the completed Kaggle
submissions above. The reproduction instructions use Kaggle. Both official
base images are pinned by manifest digest.

## 5. Optional native execution

Docker is the main reproduction method. For a native Python run, use Python
3.12 and an isolated environment. No activation or PowerShell policy change
is required.

**Windows PowerShell**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r src/python/requirements.txt
.\.venv\Scripts\python.exe src/python/train.py --data-dir src/data --output-dir results/python
```

**macOS/Linux Bash**:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r src/python/requirements.txt
.venv/bin/python src/python/train.py --data-dir src/data --output-dir results/python
```

With R installed and `Rscript` on your PATH, either shell can run:

```text
Rscript src/r/install_packages.R
Rscript src/r/train.R src/data results/r
```

R only needs the `stats`, `utils`, and `tools` packages included in R itself.
`install_packages.R` checks them and prints their versions; it does not download
unnecessary CRAN packages. Python and R both print an actionable error and
exit unsuccessfully if inputs are missing or invalid.

## 6. Repository layout and Git workflow

```text
titanic-disaster/
├── .dockerignore
├── .gitattributes
├── .gitignore
├── README.md
└── src/
    ├── data/                 # local only; ignored by Git
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── train.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── train.R
```

The public repository uses `main`, `develop`, and feature branches created
from `develop`: `feature/setup`, `feature/python`, `feature/r`, and
`feature/docs`. Changes pass through feature → `develop` PRs, followed by
`develop` → `main` PRs for initial setup and the completed project.
See the [pull request history](https://github.com/pyc1987799299-arch/titanic-disaster/pulls?q=is%3Apr+is%3Aclosed).
Commit messages and PR summaries are intentionally short.

To clone with GitHub Desktop, choose **File → Clone repository → URL**, paste
`https://github.com/pyc1987799299-arch/titanic-disaster.git`, and select a new
local folder. Then choose **Repository → Open in your external editor**.
Create the ignored data folder and follow sections 2–3 to reproduce the run.

Submit the public repository link, not a ZIP file. `CODEOWNERS` is optional
and omitted; a CODEOWNERS file by itself would not restrict who can push.

## 7. Troubleshooting

| Symptom | Action |
| --- | --- |
| `docker` is not recognized | Install Docker, reopen the terminal, and start Docker Desktop. |
| Cannot connect to the Docker daemon | Wait for Docker Desktop to finish starting; use Linux containers. |
| Bind mount source path does not exist | Create all three folders in section 2 and run from the repository root. |
| Missing `train.csv` or `test.csv` | Extract both CSVs directly into `src/data`, not a nested `titanic` folder. |
| `COPY` cannot find a file during build | Run the build command from the root with the final `.` included. |
| No results after running | Check the exit status and mounts; results should be in the language's host output folder. |
| Native Python dependency install fails | Use Python 3.12 or the Docker workflow; the pinned versions target this environment. |
| Image has no matching CPU architecture | On an ARM host, add `--platform linux/amd64` to both build and run commands. Emulation may be slower. |

The mounted folder path should not contain a comma, because Docker's `--mount`
syntax uses commas as separators. Quoted paths support spaces.

## References

- [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic)
- [scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn logistic regression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [R generalized linear models](https://stat.ethz.ch/R-manual/R-patched/library/stats/html/glm.html)
- [Docker container execution and mounts](https://docs.docker.com/engine/containers/run/)
