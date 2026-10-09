import logging
from pathlib import Path
import warnings
from reasondb.database.indentifier import RemoteColumn
from reasondb.evaluation.benchmarks.email import EnronEmail
from reasondb.interface.connect import RaccoonDB
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee


warnings.filterwarnings("error", message=".*not callable.*")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
path = Path("palimpzest/testdata/enron-eval")
EnronEmail.load("train")
csv_path = EnronEmail.load_email_table(path)

with RaccoonDB("email") as rc:
    emailEnron = rc.add_table(
        path=csv_path,
        table_name="emails",
        text_columns=[RemoteColumn("emails.text_path", "emails.text")],
    )
    df_query = enronEmail = emailEnron.extract(
        "Extract the [sender] from {text}"
    ).filter("{text} refers to a fraudulent Enron Entity (e.g. mentions Raptor, ...)")
    result = df_query.execute(
        "fraudulent_mail_senders", PrecisionGuarantee(0.6), RecallGuarantee(0.6)
    )
    print()
    print(" Results:")
    result.pprint()
    print()

    # Alternatively, the same query can be posed in natural language:
    # nl_query = rc.nl_query(
    #     'What are the senders of E-Mails that refer to a fraudulent scheme (i.e., "Raptor", ...)?',
    # )
    # result = nl_query.execute("fraudulent_mail_senders")
    # print()
    # print(" Results:")
    # result.pprint()

    ###########################################################
    #### Example queries for individual operators
    ###########################################################

    # -- filter query --
    # df_query = emailEnron.filter("{text} mentions suspicious activity")

    # -- extract query --
    # df_query = emailEnron.extract("Extract the [sender] from {text}")

    # -- project query --
    # df_query = emailEnron.extract("Extract the [sender] from {text}").project(
    #     "keep distinct {sender}"
    # )

    # -- transform query --
    # df_query = emailEnron.transform("Convert {text} to lowercase [text_lower]")

    # -- limit query --
    # df_query = emailEnron.limit("Limit to 10")

    ###########################################################
    #### Example multi-operator queries
    ###########################################################

    # df_query = (
    #     emailEnron.filter("{text} mentions suspicious activity")
    #     .extract("Extract the [sender] from {text}")
    #     .transform("Convert {sender} to lowercase [sender_lower]")
    #     .project("Keep distinct {sender_lower}")
    # )
    # df_query = (
    #     emailEnron.filter(
    #         "{text} refers to a fraudulent Enron Entity (e.g. mentions Raptor, ...)"
    #     )
    #     .extract("Extract the [sender] from {text}")
    #     .groupby("Group by {sender}")
    #     .aggregate("Count messages [count] by sender")
    #     .orderby("Order by {count} descending")
    # )

    # Join of two filtered/extracted views of the same table:
    # df_query = (
    #     emailEnron.filter("{text} mentions suspicious activity")
    #     .extract("Extract the [sender] from {text}")
    #     .join(
    #         emailEnron.filter("{text} mentions confidential information").extract(
    #             "Extract the [sender] from {text}"
    #         ),
    #         "join on {sender}",
    #     )
    #     .project("{sender}")
    # )

    ###########################################################
    #### Execution and Results
    ###########################################################

    # guarantees = [PrecisionGuarantee(0.8), RecallGuarantee(0.8)]
    # result_alt = df_query.execute("ordered_emails", *guarantees)
    # print()
    # print(" Results:")
    # result_alt.pprint()
