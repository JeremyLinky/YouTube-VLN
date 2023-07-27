"""
Evaluate path selection results.
Code adapted from https://github.com/peteanderson80/Matterport3DSimulator
"""

import argparse
import json
from collections import defaultdict

import networkx as nx
import numpy as np


def _load_nav_graphs(scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    graphs = {}
    for scan in scans:
        with open("data/connectivity/%s_connectivity.json" % scan) as f:
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
                            assert data[j]["unobstructed"][
                                i
                            ], "Graph should be undirected"
                            G.add_edge(
                                item["image_id"],
                                data[j]["image_id"],
                                weight=distance(item, data[j]),
                            )
            nx.set_node_attributes(G, values=positions, name="position")
            graphs[scan] = G
    return graphs


def _load_datasets(splits):
    data = []
    for split in splits:
        assert split in ["train", "val_seen", "val_unseen", "test"]
        with open("data/task/R2R_%s.json" % split) as f:
            data += json.load(f)
    return data


class Evaluation(object):
    """
    Results submission format: [
        {
            "instr_id": instruction_id,
            "trajectory": [(viewpoint_id, heading_rads, elevation_rads),]
        }
    ]
    """

    def __init__(self, splits):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for item in _load_datasets(splits):
            self.gt[item["path_id"]] = item
            self.scans.append(item["scan"])
            self.instr_ids += ["%d_%d" % (item["path_id"], i) for i in range(3)]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = _load_nav_graphs(self.scans)
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        """ Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). """
        gt = self.gt[int(instr_id.split("_")[0])]
        start = gt["path"][0]
        assert (
            start == path[0][0]
        ), "Result trajectories should include the start position"
        goal = gt["path"][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt["scan"], goal, path)
        self.scores["nav_errors"].append(
            self.distances[gt["scan"]][final_position][goal]
        )
        self.scores["oracle_errors"].append(
            self.distances[gt["scan"]][nearest_position][goal]
        )
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            if prev[0] != curr[0]:
                try:
                    self.graphs[gt["scan"]][prev[0]][curr[0]]
                except KeyError:
                    print(
                        "Error: The provided trajectory moves from %s to %s but the navigation graph contains no "
                        "edge between these viewpoints. Please ensure the provided navigation trajectories "
                        "are valid, so that trajectory length can be accurately calculated."
                        % (prev[0], curr[0])
                    )
                    raise
            distance += self.distances[gt["scan"]][prev[0]][curr[0]]
            prev = curr
        self.scores["trajectory_lengths"].append(distance)
        self.scores["shortest_path_lengths"].append(
            self.distances[gt["scan"]][start][goal]
        )

    def score(self, output_file):
        """ Evaluate each agent trajectory based on how close it got to the goal location """
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item["instr_id"] in instr_ids:
                    instr_ids.remove(item["instr_id"])
                    self._score_item(item["instr_id"], item["trajectory"])
        assert len(instr_ids) == 0, (
            "Trajectories not provided for %d instruction ids: %s"
            % (len(instr_ids), instr_ids)
        )
        assert len(self.scores["nav_errors"]) == len(self.instr_ids)

        num_successes = len(
            [i for i in self.scores["nav_errors"] if i < self.error_margin]
        )

        oracle_successes = len(
            [i for i in self.scores["oracle_errors"] if i < self.error_margin]
        )

        spls = []
        for err, length, sp in zip(
            self.scores["nav_errors"],
            self.scores["trajectory_lengths"],
            self.scores["shortest_path_lengths"],
        ):
            if err < self.error_margin:
                spls.append(sp / max(length, sp))
            else:
                spls.append(0)

        score_summary = {
            "length": np.average(self.scores["trajectory_lengths"]),
            "nav_error": np.average(self.scores["nav_errors"]),
            "oracle_success_rate": float(oracle_successes)
            / float(len(self.scores["oracle_errors"])),
            "success_rate": float(num_successes)
            / float(len(self.scores["nav_errors"])),
            "spl": np.average(spls),
        }

        assert score_summary["spl"] <= score_summary["success_rate"]
        return score_summary, self.scores


def eval():
    parser = argparse.ArgumentParser("Calculate standard VLN metrics")
    parser.add_argument("path", type=str, help="path to a results file")
    args = parser.parse_args()

    split = "val_unseen" if "val_unseen" in args.path else "val_seen"
    ev = Evaluation([split])
    score_summary, _ = ev.score(args.path)
    # 取小数点后4位
    if True:
        score_summary = {
            key: round(value, 4)
            for key, value in score_summary.items()
        }
    print(json.dumps(score_summary, indent=2))


if __name__ == "__main__":
    eval()
