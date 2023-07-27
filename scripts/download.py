"""
Download ressources
"""
from typing import List, Tuple
import json
import os
import shutil
from urllib.request import urlopen
import tap
import networkx as nx
import gdown
import numpy as np
from tqdm import tqdm

BEAMSEARCH_LINKS = [
    (
        "data/beamsearch/beams_test.json",
        "https://dl.dropboxusercontent.com/s/gpnm54l903fms63/beams_test.json",
    ),
    (
        "data/beamsearch/beams_train.json",
        "https://dl.dropboxusercontent.com/s/ci47p5ybitahnqx/beams_train.json",
    ),
    (
        "data/beamsearch/beams_val_seen.json",
        "https://dl.dropboxusercontent.com/s/1o6xmjjv74mq8f8/beams_val_seen.json",
    ),
    (
        "data/beamsearch/beams_val_unseen.json",
        "https://dl.dropboxusercontent.com/s/5m5by9ralaim5nb/beams_val_unseen.json",
    ),
    (
        "data/beamsearch/scores_test.json",
        "https://dl.dropboxusercontent.com/s/49u8vp01sbh9nfl/scores_test.json",
    ),
    (
        "data/beamsearch/scores_val_seen.json",
        "https://dl.dropboxusercontent.com/s/fx4o7ob23h4ei9k/scores_val_seen.json",
    ),
    (
        "data/beamsearch/scores_val_unseen.json",
        "https://dl.dropboxusercontent.com/s/aggdqj19ruttkci/scores_val_unseen.json",
    ),
]

CONFIG_LINKS = [
    (
        "data/config/bert_base_6_layer_6_connect.json",
        "https://drive.google.com/uc?id=17mL0qCWnIjqL2GNku8A7CKAi6A8Scogh",
    )
]

SPEAKER_LINKS: List[Tuple[str, str]] = [
    (
        "data/task/aug+R2R_train.json",
        "https://drive.google.com/uc?id=1cA2GRF_EGB8cw_XIxk8b6TXSEaWZEDk7",
    ),
    (
        "data/beamsearch/aug_beams_train.json",
        "https://drive.google.com/uc?id=1ukpTRI6LelEl0_gk10azW_Td95XANL2e",
    ),
    (
        "data/task/aug+R2R_val_seen.json",
        "https://dl.dropboxusercontent.com/s/8ye4gqce7v8yzdm/R2R_val_seen.json",
    ),
    (
        "data/beamsearch/aug_beams_val_seen.json",
        "https://dl.dropboxusercontent.com/s/1o6xmjjv74mq8f8/beams_val_seen.json",
    ),
    (
        "data/task/aug+R2R_val_unseen.json",
        "https://dl.dropboxusercontent.com/s/p6hlckr70a07wka/R2R_val_unseen.json",
    ),
    (
        "data/beamsearch/aug_beams_val_unseen.json",
        "https://dl.dropboxusercontent.com/s/5m5by9ralaim5nb/beams_val_unseen.json",
    ),
]

CONNECTIVITY_ROOT_URL = "https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/master/connectivity"
CONNECTIVITY_FILES = [
    "17DRP5sb8fy_connectivity.json",
    "1LXtFkjw3qL_connectivity.json",
    "1pXnuDYAj8r_connectivity.json",
    "29hnd4uzFmX_connectivity.json",
    "2azQ1b91cZZ_connectivity.json",
    "2n8kARJN3HM_connectivity.json",
    "2t7WUuJeko7_connectivity.json",
    "5LpN3gDmAk7_connectivity.json",
    "5q7pvUzZiYa_connectivity.json",
    "5ZKStnWn8Zo_connectivity.json",
    "759xd9YjKW5_connectivity.json",
    "7y3sRwLe3Va_connectivity.json",
    "8194nk5LbLH_connectivity.json",
    "82sE5b5pLXE_connectivity.json",
    "8WUmhLawc2A_connectivity.json",
    "aayBHfsNo7d_connectivity.json",
    "ac26ZMwG7aT_connectivity.json",
    "ARNzJeq3xxb_connectivity.json",
    "B6ByNegPMKs_connectivity.json",
    "b8cTxDM8gDG_connectivity.json",
    "cV4RVeZvu5T_connectivity.json",
    "D7G3Y4RVNrH_connectivity.json",
    "D7N2EKCX4Sj_connectivity.json",
    "dhjEzFoUFzH_connectivity.json",
    "E9uDoFAP3SH_connectivity.json",
    "e9zR4mvMWw7_connectivity.json",
    "EDJbREhghzL_connectivity.json",
    "EU6Fwq7SyZv_connectivity.json",
    "fzynW3qQPVF_connectivity.json",
    "GdvgFV5R1Z5_connectivity.json",
    "gTV8FGcVJC9_connectivity.json",
    "gxdoqLR6rwA_connectivity.json",
    "gYvKGZ5eRqb_connectivity.json",
    "gZ6f7yhEvPG_connectivity.json",
    "HxpKQynjfin_connectivity.json",
    "i5noydFURQK_connectivity.json",
    "JeFG25nYj2p_connectivity.json",
    "JF19kD82Mey_connectivity.json",
    "jh4fc5c5qoQ_connectivity.json",
    "JmbYfDe2QKZ_connectivity.json",
    "jtcxE69GiFV_connectivity.json",
    "kEZ7cmS4wCh_connectivity.json",
    "mJXqzFtmKg4_connectivity.json",
    "oLBMNvg9in8_connectivity.json",
    "p5wJjkQkbXX_connectivity.json",
    "pa4otMbVnkk_connectivity.json",
    "pLe4wQe7qrG_connectivity.json",
    "Pm6F8kyY3z2_connectivity.json",
    "pRbA3pwrgk9_connectivity.json",
    "PuKPg4mmafe_connectivity.json",
    "PX4nDJXEHrG_connectivity.json",
    "q9vSo1VnCiC_connectivity.json",
    "qoiz87JEwZ2_connectivity.json",
    "QUCTc6BB5sX_connectivity.json",
    "r1Q1Z4BcV1o_connectivity.json",
    "r47D5H71a5s_connectivity.json",
    "rPc6DW4iMge_connectivity.json",
    "RPmz2sHmrrY_connectivity.json",
    "rqfALeAoiTq_connectivity.json",
    "s8pcmisQ38h_connectivity.json",
    "S9hNv5qa7GM_connectivity.json",
    "sKLMLpTHeUy_connectivity.json",
    "SN83YJsR3w2_connectivity.json",
    "sT4fr6TAbpF_connectivity.json",
    "TbHJrupSAjP_connectivity.json",
    "ULsKaCPVFJR_connectivity.json",
    "uNb9QFRL6hY_connectivity.json",
    "ur6pFq6Qu1A_connectivity.json",
    "UwV83HsGsw3_connectivity.json",
    "Uxmj2M2itWa_connectivity.json",
    "V2XKFyX4ASd_connectivity.json",
    "VFuaQ6m2Qom_connectivity.json",
    "VLzqgDo317F_connectivity.json",
    "Vt2qJdWjCF2_connectivity.json",
    "VVfe2KiqLaN_connectivity.json",
    "Vvot9Ly1tCj_connectivity.json",
    "vyrNrziPKCB_connectivity.json",
    "VzqfbhrpDEA_connectivity.json",
    "wc2JMjhGNzB_connectivity.json",
    "WYY7iVyf5p8_connectivity.json",
    "X7HyMhZNoso_connectivity.json",
    "x8F5xyUWy9e_connectivity.json",
    "XcA2TqTSSAj_connectivity.json",
    "YFuZgdQ5vWj_connectivity.json",
    "YmJkqBEsHnH_connectivity.json",
    "yqstnuAEVhm_connectivity.json",
    "YVUC4YcDtcY_connectivity.json",
    "Z6MFQCViBuw_connectivity.json",
    "ZMojNkEp431_connectivity.json",
    "zsNo4HB9uLZ_connectivity.json",
    "README.md",
    "scans.txt",
]

TASK_LINKS = [
    (
        "data/task/R2R_test.json",
        "https://dl.dropboxusercontent.com/s/w4pnbwqamwzdwd1/R2R_test.json",
    ),
    (
        "data/task/R2R_train.json",
        "https://dl.dropboxusercontent.com/s/hh5qec8o5urcztn/R2R_train.json",
    ),
    (
        "data/task/R2R_val_seen.json",
        "https://dl.dropboxusercontent.com/s/8ye4gqce7v8yzdm/R2R_val_seen.json",
    ),
    (
        "data/task/R2R_val_unseen.json",
        "https://dl.dropboxusercontent.com/s/p6hlckr70a07wka/R2R_val_unseen.json",
    ),
]



def _download_url_to_file(url, path):
    print(f"downloading {url}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)


    # with urlopen(url) as response, open(path, "wb") as file:
    #     shutil.copyfileobj(response, file)

    cmd = f'curl --socks5-hostname cpu0:1080 {url} --output {path}' # in case you need a proxy
    print('command: ', cmd)
    res = os.popen(cmd)
    output = res.read()
    print(output)
    print('saving path: ',path)


    print(f"downloading {url}... done!")


def _load_nav_graph(scan):
    """ Load connectivity graph for scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    with open(f"data/connectivity/{scan}_connectivity.json") as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i, item in enumerate(data):
            if item["included"]:
                for j, conn in enumerate(item["unobstructed"]):
                    if conn and data[j]["included"]:
                        positions[item["image_id"]] = np.array(
                            [item["pose"][3], item["pose"][7], item["pose"][11]]
                        )
                        assert data[j]["unobstructed"][i], "Graph should be undirected"
                        G.add_edge(
                            item["image_id"],
                            data[j]["image_id"],
                            weight=distance(item, data[j]),
                        )
        nx.set_node_attributes(G, values=positions, name="position")
    return G


def _generate_distances(scan):
    g = _load_nav_graph(scan)
    d = dict(nx.all_pairs_dijkstra_path_length(g))
    with open(f"data/distances/{scan}_distances.json", "w") as fid:
        json.dump(d, fid)


class Arguments(tap.Tap):
    beamsearch: bool = True  # only download beamsearch data
    config: bool = True  # only download configuration files
    connectivity: bool = True  # only download connectivity data
    distances: bool = True  # only generate distance data
    task: bool = True  # only download task data
    speaker: bool = True  # only speaker-based data augmented


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    download_all = (
         args.beamsearch
        and args.config
        and args.connectivity
        and args.distances
        and args.task
    )

    if download_all or args.beamsearch:
        for path, url in BEAMSEARCH_LINKS:
            _download_url_to_file(url, path)

    if download_all or args.config:
        for path, url in CONFIG_LINKS:
            _download_url_to_file(url, path)

    if download_all or args.speaker:
        for path, url in SPEAKER_LINKS:
            _download_url_to_file(url, path)

    if download_all or args.connectivity:
        for fname in CONNECTIVITY_FILES:
            path = f"data/connectivity/{fname}"
            url = f"{CONNECTIVITY_ROOT_URL}/{fname}"
            _download_url_to_file(url, path)

    if download_all or args.distances:
        print("generating distance data...")
        os.makedirs("data/distances", exist_ok=True)
        scans = open("data/connectivity/scans.txt", "r").read().splitlines()
        for scan in tqdm(scans):
            _generate_distances(scan)
        print("generating distance data... done!")

    if download_all or args.task:
        for path, url in TASK_LINKS:
            _download_url_to_file(url, path)

    # # complete the directory structure
    # os.makedirs("data/logs", exist_ok=True)
    # os.makedirs("data/models", exist_ok=True)
    # os.makedirs("data/runs", exist_ok=True)
