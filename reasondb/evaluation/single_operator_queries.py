"""Query sets that exercise exactly one operator type.

The single-operator studies (``scripts/run_benchmark_single_operator*.py``) need query
sets built from a ``RandomBenchmark``'s ``OPERATOR_OPTIONS``/``QUERY_SHAPES`` that
isolate one operator - every extract option on its own, or every join option on its
own - so a measurement attributes cost and accuracy to that operator rather than to a
whole pipeline.
"""

import logging

from reasondb.evaluation.benchmark import RandomBenchmark

logger = logging.getLogger(__name__)


def get_extract_single_operator_queries(benchmark: RandomBenchmark):
    """
    Generate queries for only extract operators from the benchmark's operator options.
    For benchmarks that need joins (like rotowire), include the join operations.
    """
    from reasondb.query_plan.query import Queries, QueryShape
    from reasondb.query_plan.logical_plan import LogicalExtract
    from reasondb.database.indentifier import VirtualTableIdentifier
    from reasondb.query_plan.query import OperatorPlaceholder
    import random
    
    random.seed(42)
    queries = []
    
    # Get the base table name from the benchmark's single filter shape
    single_filter_shapes = benchmark.single_filter_shape()
    
    # Get operator options for the benchmark
    operator_options = benchmark.get_operator_options()
    
    # Process each category of operators
    for key, ops_by_type in operator_options.items():
        # Get the corresponding shape for this key
        if key in single_filter_shapes:
            base_shape = single_filter_shapes[key]
            
            # Handle LogicalExtract operators only
            if LogicalExtract in ops_by_type:
                # Clone the shape structure but replace Filter placeholder with Extract placeholder
                # The shape includes any necessary joins before the operator
                shape_steps = list(base_shape.shape)
                
                # Replace the last step (which is a Filter placeholder) with an Extract placeholder
                if len(shape_steps) > 0 and isinstance(shape_steps[-1], OperatorPlaceholder):
                    # Get the input and output from the original placeholder
                    original_placeholder = shape_steps[-1]
                    
                    # Create new shape with Extract instead of Filter
                    new_shape_steps = shape_steps[:-1] + [
                        OperatorPlaceholder(
                            LogicalExtract,
                            inputs=original_placeholder.inputs,
                            output=original_placeholder.output,
                        )
                    ]
                    
                    extract_shape = QueryShape(*new_shape_steps)
                    
                    # Generate a query for each extract option
                    for option in ops_by_type[LogicalExtract]:
                        query = extract_shape.instantiate({LogicalExtract: [option]})
                        queries.append(query)
                else:
                    # Fallback: simple extract without joins (for benchmarks that don't need joins)
                    first_shape_step = shape_steps[0] if shape_steps else None
                    if first_shape_step and hasattr(first_shape_step, 'inputs'):
                        base_table_name = first_shape_step.inputs[0].name
                    else:
                        base_table_name = "data"  # Default fallback
                    
                    extract_shape = QueryShape(
                        OperatorPlaceholder(
                            LogicalExtract,
                            inputs=[VirtualTableIdentifier(base_table_name)],
                            output=VirtualTableIdentifier("output"),
                        ),
                    )
                    for option in ops_by_type[LogicalExtract]:
                        query = extract_shape.instantiate({LogicalExtract: [option]})
                        queries.append(query)
    
    return Queries(*queries)


def get_join_single_operator_queries(benchmark: RandomBenchmark):
    """
    Generate queries for only join operators from the benchmark's operator options.
    Prefers a `get_join_queries()` method on the benchmark (which creates proper
    Query objects directly) over the QueryShape.instantiate() path, which does not
    support self-joins with column renaming.
    """
    from reasondb.query_plan.query import Queries
    from reasondb.query_plan.logical_plan import LogicalJoin

    # Prefer direct Query construction if the benchmark supports it
    if hasattr(benchmark, "get_join_queries"):
        return benchmark.get_join_queries()

    operator_options = benchmark.get_operator_options()

    # Fall back to QueryShape.instantiate() for benchmarks without get_join_queries()
    join_shape = None
    for key, shapes in benchmark.get_query_shapes().items():
        for shape in shapes:
            if LogicalJoin in shape.get_required_operators_per_type():
                join_shape = (key, shape)
                break
        if join_shape is not None:
            break

    if join_shape is None:
        logger.warning("No join shape found in benchmark query shapes")
        return Queries()

    key, shape = join_shape
    ops_by_type = operator_options.get(key, {})

    if LogicalJoin not in ops_by_type:
        logger.warning(f"No join operator options found for key '{key}'")
        return Queries()

    queries = []
    for option in ops_by_type[LogicalJoin]:
        query = shape.instantiate({LogicalJoin: [option]})
        queries.append(query)

    return Queries(*queries)
