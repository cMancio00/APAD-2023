import networkx as nx
import pandas as pd
import numpy as np

import os
os.environ["MODIN_ENGINE"] = "ray"
import modin.pandas as pd
import ray
ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})
import pandas as pd

def main():
    
    df = pd.read_csv(
        "Data/out-dblp_article.csv",
        delimiter=";",
        usecols=[
            "id",
            "author",
            "title"
        ]
        )

    df.dropna()

    print(df.head())


if __name__ == "__main__":
    main()