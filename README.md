# Setup

Use Python v3.13 or later.

```sh
# Export your OpenAI API key for query planning
export OPENAI_API_KEY=....

# Update the submodules
git submodule update --init --recursive

# Install dependencies and the package
pip install -r requirements.txt
pip install -e .

# On another terminal: Start the backend servers
bash scripts/start_servers.sh

# Run the demos
python demos/artwork.py
python demos/emails.py
python demos/real_estate.py
python demos/rotowire.py

# Run benchmarks
# Basic usage (requires a GPU device specification to run Stretto):
python scripts/run_benchmark.py --device cuda:0

# You can specify which approaches to run with --select-executors (optim_global = Stretto, abacus, lotus, optim_local, ...) 
# You can specify the datasets with --benchmarks (e.g. artwork_random_medium, movie_random, email_random, rotowire_random, ecommerce_random_large are used in the paper)
# Set precision and recall guarantees: --precision-guarantees / --recall-guarantees
# For example:
python scripts/run_benchmark.py --device cuda:0 --benchmarks artwork rotowire --select-executors optim_global lotus --precision-guarantees 0.7 0.9 --recall-guarantees 0.7 0.9

```

## Benchmarks

### Datasets

Here is how to obtain the data sets:

- Artwork: Already included, running an artwork query will automatically download the required images from Wikidata.
- Rotowire: Already included, see `reasondb/benchmarks/evaluation/files`
- Email: Running the benchmark will automatically run the Palimpzest Script to download the data set. See also the instructions of Palimpzest.
- Movies: Already included, see `reasondb/benchmarks/evaluation/files/movies_1000.csv`
- ecommerce: Download the dataset [here](https://sembench.ngrok.io/). We expect the following file structure <project-root>/SemBench/ecomm/1/fashion-dataset/...

### Query Generation

Random queries are generated from predefined query shapes and operator options.
For each benchmark (i.e. artwork, rotowire, email, movie, ecommerce), you can find:

- **Query definitions and operator options**: `reasondb/evaluation/benchmarks/<benchmark_name>.py`
  - Example: `reasondb/evaluation/benchmarks/artwork.py`
  - Contains:
    - `ARTWORK_OPERATOR_OPTIONS`: List of possible operators (filters and extracts) that can be used in random query generation
    - `ARTWORK_QUERY_SHAPES`: Templates defining the structure of randomly generated queries (e.g., 2 operators, 3 operators)

- **Query generation**: The `RandomBenchmark` class (base class) generates queries by:
  1. Selecting a query shape (number and order of operators)
  2. Randomly sampling operators from the operator options
  3. Creating valid operator sequences following the shape template
