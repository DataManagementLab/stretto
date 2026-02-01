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
### Link to User Study page ==>> https://form.typeform.com/to/xSpYQv5t
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

    # nl_query = rc.nl_query(
    #     'What are the senders of E-Mails that refer to a fraudulent scheme (i.e., "Raptor", ...)?',
    # )
    # result = nl_query.execute("fraudulent_mail_senders")
    # print()
    # print(" Results:")
    # result.pprint()

    # print("*" * 100)
    # print("(--labgroup implementation--)")
    # print()

    ###########################################################
    #### Simple Queries
    ###########################################################

    # -- filter query --
    # df_query = emailEnron.filter("{text} mentions suspicious activity")

    # -- extract query --
    # df_query = emailEnron.extract("Extract the [sender] from {text}")

    # -- project query --     #NOTE needs col name from previous operator?
    # df_query = emailEnron.project(
    #     "Keep only {text}"
    # )  # gives -> IndexError: list index out of range -> if {id} or {text}
    # df_query = emailEnron.extract("Extract the [sender] from {text}").project(
    #     "keep distinct {sender}"
    # )

    # -- transform query --
    # df_query = emailEnron.transform("Convert {text} to lowercase [text_lower]")

    # -- limit query --      
    # df_query = emailEnron.limit("Limit to 10")

    # -- offset query --
    # not implemented yet in reasondb

    # -- groupby query --     

    # -- aggregate query --  

    # -- orderby query --
    # df_query = (emailEnron.orderby("Order by {id} ascending"))  #index out of range error -> similiar to project query

    # -- join query --       

    ###########################################################
    #### Complex Queries
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

    # partial query for join test -- about 30 more rows
    # df_query = emailEnron.filter("{text} mentions suspicious activity").extract(
    #     "Extract the [sender] from {text}"
    # )

    # partial query for join test -- about 129 more rows
    # df_query = emailEnron.filter("{text} mentions confidential information").extract(
    #     "Extract the [sender] from {text}"
    # )

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

    # evaluator = MetricsManager(
    #     predictions_df=result_alt.to_df(),
    #     ground_truth_json=" ",  # provide path to ground truth json file
    #     attribute_eval="sender",
    # )
    # results = evaluator.evaluate_emails()
    # print(results)

    ###########################################################
    #### Old Queries
    ###########################################################

    # test Filtering  DataFrameInterface#############################

    # df_query = emailEnron.filter(
    #     "{emails.text} refers to a fraudulent Enron Entity (e.g. mentions Raptor, ...)"
    # ).extract("Extract the [sender] from {text}")

    # df_query = df_query.project(f"Keep distinct {{{df_query.output_table}.sender}}")

    # result = df_query.execute("fraudulent_mail_senders")
    # print()
    # print("Results:")
    # result.pprint()

    # test Groupby and aggregate  DataFrameInterface#############################

    # df_query = emailEnron.filter(
    #     "{emails.text} refers to a fraudulent Enron Entity (e.g. mentions Raptor, ...)"
    # ).extract("Extract the [sender] from {text}")
    # # .extract("Extract the [sender] from {FilteredEmails.text}")
    # # df_query.groupby("Group by {FilteredEmails_extracted.sender}")
    # df_query = df_query.groupby("Group by {sender}").aggregate(
    #     "Count messages [count] by sender"
    # )

    # result_alt = df_query.execute("sender_Groups_Aggregated")
    # print()
    # print(" Results:")
    # result_alt.pprint()

    # test orderby/sorting DataFrameInterface##################################################
    # df_query = emailEnron.filter(
    #     "{emails.text} refers to a fraudulent Enron Entity (e.g. mentions Raptor, ...)"
    # ).orderby("Order by {emails.id} ascending")

    # df_query = (
    #     emailEnron.filter(
    #         "{emails.text} refers to a fraudulent Enron Entity (e.g. mentions Raptor, ...)"
    #     )
    #     .extract("Extract the [sender] from {text}")
    #     .groupby("Group by {sender}")
    #     .aggregate("Count messages [count] by sender")
    #     .orderby("Order by {count} descending")
    # )

    # test join DataFrameInterface##################################################
    # maybe extract sender 2times on emails and join

    # df_query = (
    #     emailEnron.filter("{text} refers to a fraudulent Enron Entity (e.g. mentions Raptor, ...)")
    #     .join(emailEnron.filter("{text} refers to a original Entity"), "join on {names}")
    #     .filter("{text} refers to emails in general")
    # )

    # df_query = emailEnron.filter("{text} refers to security")
    # df_query = emailEnron.join(emailEnron, "join on {sender}")
    # df_query = emailEnron.extract("Extract the [sender] from {text}")

    # first query
    # df_query = (
    #     emailEnron.filter("{text} mentions suspicious activity")
    #     .extract("Extract the [sender] from {text}")
    #     .transform("Convert {sender} to lowercase [sender_lower]")
    #     .project("Keep only {sender_lower}")
    # )

    # 2nd query
    # df_query = (
    #     emailEnron.extract("Extract the [sender] from {text}")  # First extract
    #     .join(
    #         emailEnron.extract("Extract the [sender] from {text}"),  # Second extract
    #         "join on {sender}",  # Joining on the extracted sender field
    #     )
    # )

    # 3rd query
    # df_query = (
    #     emailEnron.filter("{text} mentions Raptor")  # Filter for fraudulent emails
    #     .extract("Extract the [sender] from {text}")  # Extract sender from those emails
    #     .join(
    #         emailEnron.filter("{text} mentions security") # filter for S mails
    #         .extract("Extract the [sender] from {text}"),
    #         "join on {sender}"  # Join on extracted senders
    #     )
    #     .project("{sender}")  # Project only sender
    # )

    # 4th query
    # df_query = (
    #     emailEnron.extract("Extract the [sender] from {text}")  # First extract
    #     .extract("Extract the [topic] from {text}")   # Second extract on same table
    #     .join(
    #         emailEnron.extract("Extract the [sender] from {text}"),
    #         "join on {sender}"
    #     )
    #     .filter("{topic} contains 'fraud'")  # Filter based on extracted topic
    #     .project("{sender}, {topic}")
    # )

    #  5th query
    # df_query = (
    #     emailEnron.filter("{text} refers to security")
    #     .extract("Extract the [sender] from {text}")  # First extract
    #     .join(
    #         emailEnron.filter("{text} mentions Raptor").extract("Extract the [sender] from {text}"),
    #         "join on {sender}"
    #     )
    #     .project("{sender}")
    # )

    # df_query = (
    #     emailEnron.filter("{text} refers to a fraudulent Enron Entity (e.g. mentions Raptor, ...)")
    #     .extract("Extract the [sender] from {text}")
    #     .project("Keep distinct {sender}")
    # )

    # df_query = emailEnron.filter("{text} refers to security").extract("Extract the [sender] from {text}").project("{sender}")
    # df_query = emailEnron.join(emailEnron, "join on {sender}")
    # df_query = emailEnron.extract("Extract the [sender] from {text}")

    # df_query = emailEnron.filter("{text} refers to security").extract("Extract the [sender] from {text}").extract("Extract the [topic] from {text}")
    # df_query1 = emailEnron.filter("{text} mentions Raptor").extract("Extract the [sender] from {text}").join(df_query,"{df_query1.sender} = {df_query.sender}")
    # df_query = (emailEnron.filter("{text} mentions suspicious activity")
    #             .extract("Extract the [sender] from {text}")
    #             .transform("Convert {sender} to lowercase [sender_lower]")
    #             .project("Keep only {sender_lower}")
    # )

    # guarantees = Guarantee(0)

    # result_alt = df_query.execute("ordered_emails", guarantees)
    # print()
    # print(" Results:")
    # result_alt.pprint()
    # df = result_alt.to_pandas()
    # print(df.head())

    # result_table = emailEnron.connection.database.register_query_result(df_query.execute("ordered_emails"))
    # print(result_table.to_pandas())
