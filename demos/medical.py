import logging
import os
import gdown
import zipfile
from pathlib import Path
from reasondb.database.indentifier import InPlaceColumn, RemoteColumn
from reasondb.interface.connect import RaccoonDB
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_path = Path("SemBench/files/medical")
if not data_path.exists():
    data_path.mkdir(parents=True, exist_ok=True)
    gdown.download(
        "https://drive.google.com/uc?id=1v4--C7PE_SQDNZj6hZ8wNn4OvswIyu2s",
        output="SemBench/files/medical/medical_data.zip",
    )
    with zipfile.ZipFile("SemBench/files/medical/medical_data.zip", "r") as zip_ref:
        zip_ref.extractall("SemBench/files/medical/")


csv_text = orig_csv = (
    Path(__file__).parents[1]
    / "reasondb"
    / "evaluation"
    / "benchmarks"
    / "files"
    / "text_symptoms_data_1000.csv"
)

csv_skin = orig_csv = (
    Path(__file__).parents[1]
    / "reasondb"
    / "evaluation"
    / "benchmarks"
    / "files"
    / "image_skin_data_1000.csv"
)

csv_x_ray = orig_csv = (
    Path(__file__).parents[1]
    / "reasondb"
    / "evaluation"
    / "benchmarks"
    / "files"
    / "image_x_ray_data_1000.csv"
)

csv_patient = orig_csv = (
    Path(__file__).parents[1]
    / "reasondb"
    / "evaluation"
    / "benchmarks"
    / "files"
    / "patient_data_with_labels_1000.csv"
)

with RaccoonDB("medical_sampled") as rc:
    symptoms = rc.add_table(
        path=csv_text,
        table_name="patients_symptoms",
        text_columns=[
            InPlaceColumn("patients_symptoms.symptoms"),
        ],
    )

    skin = rc.add_table(
        path=csv_skin,
        table_name="skin",
        image_columns=[
            RemoteColumn("skin.image_path", "skin.skin_picture"),
        ],
    )

    x_ray = rc.add_table(
        path=csv_x_ray,
        table_name="x_ray",
        image_columns=[
            RemoteColumn("x_ray.image_path", "x_ray.xray_picture"),
        ],
    )

    patient = rc.add_table(
        path=csv_patient,
        table_name="patient",
    )

    # Q1

    #    nl_query = rc.nl_query(
    #       "Which patients have symptoms that indicate allergies?",
    #    )

    # df_query = symptoms.filter("{symptoms} indicates allergies").project(
    #     "Keep {patient_id}"
    # )
    #
    # result = df_query.execute(
    #     "allergy_patients",
    #     PrecisionGuarantee(0.6),
    #     RecallGuarantee(0.6),
    # )
    # result.pprint()

    # Q3

    #    nl_query = rc.nl_query(
    #        "Find patients who have cancer in their family history and have lung problems according to their X-ray images for the patients considered sick"
    #    )
    #
    # df_query = (
    #    patient.filter("{is_sick} is True")
    #    .filter("{did_family_have_cancer} is 1")
    #    .join(
    #        x_ray,
    #        "Join on {patient_id}"
    #    )
    #    .filter("{xray_picture} shows lung problems")
    #    .project("Keep {patient_id}")
    # )
    #
    # result = df_query.execute(
    #    "cancer_family_lung_problems_patients",
    #    PrecisionGuarantee(0.6),
    #    RecallGuarantee(0.6),
    # )
    # result.pprint()
    #
    #    # Q4
    #
    #    nl_query = rc.nl_query(
    #        "What is the average age of patients with acne?"
    #    )
    #
    df_query = (
        symptoms.filter("{symptoms} indicates acne")
        .join(patient, "Join on {patient_id}")
        .project("Keep {age}")
        .aggregate("Average of {age}[average_age_acne_patients]")
    )

    result = df_query.execute(
        "average_age_acne_patients",
        PrecisionGuarantee(0.6),
        RecallGuarantee(0.6),
    )
    result.pprint()
#
#    # Q9
#
#    nl_query = rc.nl_query(
#        "Find patients who are sick according to both skin moles image and lung X-Ray image"
#    )
#
# df_query = (
#    skin.filter("{skin_picture} shows sick moles")
#    .join(x_ray, "Join on {patient_id}")
#    .filter("{xray_picture} shows lung problems")
#    .project("Keep {patient_id}")
# )
#
# result = df_query.execute(
#    "sick_patients_skin_xray",
#    PrecisionGuarantee(0.6),
#    RecallGuarantee(0.6),
# )
# result.pprint()
