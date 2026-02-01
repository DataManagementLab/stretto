import pandas as pd
import json


def load_description(product_id: str) -> str:
    with open(
        f"../../../../SemBench/ecomm/1/fashion-dataset/styles/{product_id}.json",
    ) as file:
        data = json.load(file)

        try:
            return data["data"]["productDescriptors"]["description"]["value"]
        except KeyError:
            print(f"Description not found for product ID: {product_id}")
            return "No description available."


# df = pd.read_csv("ecommerce_products.csv")
# df["product_image"] = df["id"].map(
#     lambda x: f"SemBench/ecomm/1/fashion-dataset/images/{x}.jpg"
# )
# df["description"] = df["id"].map(load_description)
# df.to_csv("ecommerce_products.csv", index=False)

df = pd.read_csv("ecommerce_products_large.csv")
df["product_image"] = df["id"].map(
    lambda x: f"SemBench/ecomm/1/fashion-dataset/images/{x}.jpg"
)
df["description"] = df["id"].map(load_description)
df.to_csv("ecommerce_products_large.csv", index=False)
