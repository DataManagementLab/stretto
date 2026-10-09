"""Setup script for the package."""

from setuptools import setup, find_packages

setup(
    name="stretto",
    version="0.0.1",
    description="Stretto execution engine for Semantic Data Systems",
    url="https://github.com/DataManagementLab/stretto",
    author="Gabriele Sanmartino, Matthias Urban, Paolo Papotti, and Carsten Binnig",
    author_email="matthias.urban@cs.tu-darmstadt.de",
    license="MIT",
    packages=find_packages(),
    # The pinned dataset statistics (`evaluation/dataset_token_stats.json`) are read
    # through `Path(__file__).parent`, so a non-editable install has to carry them or the
    # plotting layer silently loses a dimension.
    package_data={"reasondb.evaluation": ["*.json"]},
)
