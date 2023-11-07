import networkx as nx
import pandas as pd
import numpy as np
import itertools
import os
os.environ["MODIN_ENGINE"] = "ray"
import modin.pandas as pd
import ray
ray.shutdown()
ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})


def read_csv(name:str, path:str = "Data")->pd.DataFrame:
    cols = ["id","author","title"]
    if name == "out-dblp_proceedings.csv":
        cols = ["id","editor","title"]
        
    df = pd.read_csv(
    path + "/" + name,
    delimiter=";",
    usecols=cols,
    nrows=500
    )
    df.name = name.split(".")[0]
    df.rename(columns={"editor":"author"}, inplace=True)
    return df

def create_DataFrame_list(csv_list:list[str])->list[pd.DataFrame]:
    df_list = list()
    for csv in csv_list:
        df_list.append(
            read_csv(csv)
        )
        df_list[-1].dropna(inplace = True)
    return df_list

def create_csv_list(path:str = "Data")->list[str]:
    return os.listdir(path)

def create_graph(df:pd.DataFrame)->nx.Graph:
    G = nx.Graph()
    for publication_id, row in df.iterrows():
        authors = row["author"].split("|")
        title = row["title"]
        G.add_node(publication_id, bipartite = 0, title=title, authors_counter = len(authors))
        for author in authors:
            G.add_node(author, bipartite = 1)
            G.add_edge(publication_id,author)
    return G

def create_graph_list(df_list:list[pd.DataFrame])->list[nx.Graph]:
    graph_list = list()
    for df in df_list:
        graph_list.append(
            create_graph(df)
        )
    return graph_list

def get_publication_with_max_authors(G:nx.Graph)->tuple[str, list[str], int]:
    publication_ids = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    publication_ids = list(publication_ids)

    authors_counter_array = np.array(
        list(map(lambda id: G.nodes[id]["authors_counter"], publication_ids))
        )

    max_authors_pubication_id = publication_ids[authors_counter_array.argmax()]

    authors = G.neighbors(max_authors_pubication_id)
    title = G.nodes[max_authors_pubication_id]["title"]

    return (
        title,
        list(authors),
        G.nodes[max_authors_pubication_id]['authors_counter']
    )

def get_author_with_most_collaborotors(G:nx.Graph)->tuple:
    authors = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}
    authors = list(authors)

    max = {"author": "None","collaborators":list()}
    for author in authors:
        collaborators = set()
        publication_ids = [publication_id[1] for publication_id in list(G.edges(author))]
        for publication_id in publication_ids:
            collaborators.update(
                [publication_id[1] for publication_id in list(G.edges(publication_id))]
            )
        collaborators = list(collaborators)
        collaborators.remove(author)
        if len(collaborators)  > len(max["collaborators"]):
            max["author"] = author
            max["collaborators"] = collaborators
    return(
        max['author'],
        len(max['collaborators']),
        max['collaborators']
    )

def calculate_B_i(G:nx.Graph, u:str, i:int):
    a = nx.eccentricity(G)
    F = list()
    for key in a.keys():
        if a[key] == i:
            F.append(key)
    B_i = 0
    for node in F:
        max = nx.eccentricity(G, v=node)
        if max > B_i:
            B_i = max
    return B_i

def calculate_starting_node(G:nx.Graph)->str:
    lcc = get_largest_connected_component(G)
    return list(lcc)[0]

def get_largest_connected_component(G:nx.Graph)->nx.Graph:
    return G.subgraph(
    sorted(nx.connected_components(G), key = len, reverse=True)[0]
    ).copy()

def iFub(G:nx.Graph,u:str)-> int:
    lcc = get_largest_connected_component(G)
    i = nx.eccentricity(lcc,v=u)

    lb = i
    ub = 2*lb

    while ub > lb:
        B_i = calculate_B_i(lcc,u,i)
        max = np.max([lb,B_i])
        if max > 2*(i-1):
            return max
        else:
            lb = max
            ub = 2*(i-1)
        i=i-1
    return lb

def answer_to_all_main_questions(G:nx.Graph,name:str)->None:
    print(f"-------------Graph: {name}--------------\n")

    title, authors, counter = get_publication_with_max_authors(G)
    print("The publication with most authors is:")
    print(f"{title} \n {list(authors)} \n wich has {counter} authors\n")

    diameter = iFub(G, calculate_starting_node(G))
    nx_diameter = nx.diameter(get_largest_connected_component(G))
    print(f"The graph has exact diameter: {diameter}")
    print(f"Diameter calculated with NetworkX is: {nx_diameter}\n")

    author, count, collaborators = get_author_with_most_collaborotors(G)
    print(f"The author with most collaborators is {author}, with {count} collaborators:\n{collaborators}\n")

def concatenate_DataFrame_from_list(df_list:list[pd.DataFrame])->pd.DataFrame:
    df = pd.concat(
        df_list,
        axis=0,
        ignore_index=True
    )
    df.drop_duplicates(
        subset='title',
        keep='first',
        inplace=True
    )
    return df

def build_union_graph_from_DataFrame_list(df_list:list[pd.DataFrame])->nx.Graph:
    return create_graph(
        concatenate_DataFrame_from_list(df_list)
    )

def build_authors_graph_from_DataFrame_list(df_list:list[pd.DataFrame])->nx.Graph:
    df = concatenate_DataFrame_from_list(df_list)
    df = df["author"]
    G = nx.Graph()
    for authors in df:
        authors_list = authors.split("|")
        for author_comb in itertools.combinations(authors_list,2):
            if G.has_edge(*author_comb):
                G[author_comb[0]][author_comb[1]]["weight"] += 1
            else:
                G.add_edge(*author_comb,weight = 1)
    return G

def find_most_collaborating_couple(G:nx.Graph)->tuple[str,str,dict[int]]:
    return  sorted(G.edges(data=True),key= lambda x: x[2]['weight'],reverse=True)[0]

def main():

    df_list = create_DataFrame_list(
        create_csv_list()
    )

    graph_list = create_graph_list(df_list)

    for idx, G in enumerate(graph_list):
        answer_to_all_main_questions(
            G,
            df_list[idx].name
        )
    
    Union_g = build_union_graph_from_DataFrame_list(df_list)

    answer_to_all_main_questions(
        Union_g,
        "Union"
        )

    author_1, author_2, weight = find_most_collaborating_couple(
        build_authors_graph_from_DataFrame_list(df_list)
        )

    print(f"The most collaborating authors are {author_1} and {author_2} with {weight['weight']} collaborations togheter ")



if __name__ == "__main__":
    main()