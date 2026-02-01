from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.query_plan.logical_plan import (
    LogicalFilter,
    LogicalLimit,
    LogicalPlan,
    LogicalProject,
    LogicalSorting,
    LogicalExtract,
)
from reasondb.query_plan.query import Queries, Query


class FewShotDatabase:
    def __init__(self, queries=()):
        self._samples: Queries = Queries(*queries)

    def add_sample(self, query: Query):
        assert query.has_ground_truth()
        self._samples.append(query)

    def retrieve(self, query: Query) -> Queries:
        return self._samples


DUMMY_FEW_SHOT_DATABASE = FewShotDatabase(
    [
        Query(
            "What is the capital of the most populous country in Europe?",
            _ground_truth_logical_plan=LogicalPlan(
                [
                    LogicalFilter(
                        explanation="First, we need to get only countries in Europe.",
                        inputs=[VirtualTableIdentifier("countries")],
                        output=VirtualTableIdentifier("countries_in_europe"),
                        expression="{countries.description} describes {countries.name} to be in 'Europe'",
                    ),
                    LogicalExtract(
                        explanation="Next, we need to extract the population from the description.",
                        inputs=[VirtualTableIdentifier("countries_in_europe")],
                        output=VirtualTableIdentifier("countries_in_europe"),
                        expression="Get the population [population] of {countries_in_europe.name} from {countries_in_europe.description}",
                    ),
                    LogicalSorting(
                        explanation="Then, we need to sort the countries by population.",
                        inputs=[VirtualTableIdentifier("countries_in_europe")],
                        output=VirtualTableIdentifier("sorted_countries"),
                        expression="sort in descending order by {countries_in_europe.population}",
                    ),
                    LogicalLimit(
                        explanation="We need to get the first country.",
                        inputs=[VirtualTableIdentifier("sorted_countries")],
                        output=VirtualTableIdentifier("most_populous_country"),
                        expression="Keep the first row",
                    ),
                    LogicalExtract(
                        explanation="Finally, we need to get the capital of that country.",
                        inputs=[VirtualTableIdentifier("most_populous_country")],
                        output=VirtualTableIdentifier("most_populous_country"),
                        expression="Get the capital [capital] of {most_populous_country.name} from {most_populous_country.description}",
                    ),
                    LogicalProject(
                        explanation="Finally, we need to get the capital of the country.",
                        inputs=[VirtualTableIdentifier("most_populous_country")],
                        output=VirtualTableIdentifier("capital"),
                        expression="Keep {most_populous_country.capital}",
                    ),
                ]
            ),
        ),
    ]
)
