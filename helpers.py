import pandas as pd
import numpy as np
import scipy
from datetime import datetime


def parse_data(query_data):
    client = query_data.get("client_id", None)
    history = query_data.get("transaction_history", None)

    if not client:
        return None, None

    products_history = []
    if history:
        for session in history:
            transaction_time = datetime.strptime(session.get("datetime", None), "%Y-%m-%dT%H:%M:%S")

            session_products = session.get("products", None)
            if session_products:
                bucket = []
                for product in session_products:
                    bucket.append(product["product_id"])

                products_history.append((bucket, transaction_time))

        products_history = sorted(products_history, key=lambda x: x[1])  # sort by time
        products_history = [tup[0] for tup in products_history]
        return products_history
    else:
        return None


def sort_by_dict(lst, dct):
    counts = []

    for element in lst:
        counts.append((element, dct.get(element, 0)))

    return [count[0] for count in sorted(counts, key=lambda x: x[1], reverse=True)]


def predict_user(model, products, product_dict, reverse_product_dict, matrix_shape):
    enum_clients = np.zeros(
        len([product for product in products if product in product_dict])
    )
    enum_products = np.array(
        [product_dict[product]
            for product in products if product in product_dict]
    )

    sparse_matrix = scipy.sparse.csr_matrix(
        (np.ones(shape=(len(enum_clients))), (enum_clients, enum_products)),
        shape=matrix_shape,
    )

    rec = model.recommend(
        0, sparse_matrix, N=30, recalculate_user=True, filter_already_liked_items=False
    )

    return [reverse_product_dict[r[0]] for r in rec]
