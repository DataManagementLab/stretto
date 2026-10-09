"""One definition of "these two answers are the same answer".

Extracted values are compared as *strings*, in two places that must agree:

- :func:`reasondb.evaluation.evaluate` scores a run's predictions against its labels.
- :meth:`reasondb.optimizer.profiler.ProfileLevelOutput.consoldidate` decides, during
  profiling, which candidate operators agree with the step's label source -- it builds the
  merged tuple set by row-content equality, so a candidate whose extracted value differs
  from the label source's is scored a false positive.

Both apply it unconditionally so that the profiler and the scorer use the same notion
of equality; otherwise ``Berlin`` vs ``berlin`` vs ``"Berlin"`` would count as a miss
for reasons unrelated to operator quality.

What it folds: case, quotes, a ``STRING:`` prefix, surrounding whitespace, and trailing
sentence punctuation. What it does *not*: anything inside the value. ``Berlin, Germany``
and ``Berlin`` remain different answers, as do ``1889`` and ``1889 AD``.

Lives here rather than in ``reasondb.evaluation.evaluation`` so the profiler can use it:
that module imports the executor, which imports the optimizer, which imports the profiler.
Standard library and pandas only.
"""

import pandas as pd


#: Sentence-terminating punctuation an instruction-tuned model tacks onto a one-word
#: answer. Deliberately not brackets, hyphens or slashes: those carry meaning *inside* a
#: value ("n/a", "1939-1945", "10/10") rather than decorating the end of one.
TRAILING_PUNCTUATION = ".,;:!?"


def postprocess_string(s: str) -> str:
    """
    Post-process a single string value:
    1. Convert to lowercase
    2. Remove " and ' symbols
    3. Remove "STRING:" prefix if present
    4. Strip leading/trailing whitespace
    5. Strip trailing sentence punctuation
    """
    if not isinstance(s, str):
        return s

    # 1. Convert to lowercase
    s = s.lower()

    # 2. Remove " and ' symbols
    s = s.replace('"', "").replace("'", "")

    # 3. Remove "STRING:" prefix if present
    if s.startswith("string:"):
        s = s[7:]  # Remove "string:" (7 characters)

    # 4. Strip leading/trailing whitespace
    s = s.strip()

    # 5. Strip trailing sentence punctuation, including any whitespace it hides behind
    # ("berlin ." and "berlin..." both land on "berlin"). Kept only if something remains:
    # an answer that is *entirely* punctuation stays itself rather than collapsing to "",
    # which would merge it with every genuinely empty answer.
    without_punctuation = s.rstrip(TRAILING_PUNCTUATION + " \t\n\r")
    if without_punctuation:
        s = without_punctuation

    return s


def postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process a dataframe by applying the following transformations to string columns:
    1. Convert to lowercase
    2. Remove " and ' symbols
    3. Remove "STRING:" prefix if present
    4. Strip leading/trailing whitespace (keep it between words)
    5. Strip trailing sentence punctuation
    """
    df = df.copy()

    for col in df.columns:
        # Only process columns that contain strings
        if df[col].dtype == "object" or df[col].dtype.name == "string":
            df[col] = df[col].apply(
                lambda x: postprocess_string(x) if isinstance(x, str) else x
            )

    return df
