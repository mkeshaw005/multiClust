__author__ = 'mshaw'

from math import pow, sqrt
import scipy

from sklearn import metrics
from pylab import plot, clf
import matplotlib.pyplot as pyplot
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from pygraph.algorithms.minmax import minimal_spanning_tree
import math
from pygraph.classes.graph import graph


def read_data(file_name):
    f = open(file_name, 'r')
    data = []
    identifiers = []
    coordinates_by_identifier = {}
    x_min = float('inf')
    x_max = -float('inf')
    y_min = float('inf')
    y_max = -float('inf')
    import math
    for line in f:
        if not line is "":
            line = line.rstrip()
            line_values = line.split(",")
            x_line = float(line_values[0])
            y_line = math.log(float(line_values[1]), 2)
            if x_line < x_min:
                x_min = x_line
            if x_line > x_max:
                x_max = x_line
            if y_line < y_min:
                y_min = y_line
            if y_line > y_max:
                y_max = y_line
            coordinates = [x_line, y_line]
            data.append(coordinates)
            identifiers.append(line_values[2])
            coordinates_by_identifier[line_values[2]] = coordinates

    return data, identifiers, coordinates_by_identifier, {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def read_data_generalized(file_name):
    f = open(file_name, 'r')
    data = []
    identifiers = []
    coordinates_by_identifier = {}
    min = []
    max = []
    for line in f:
        line = line.rstrip()
        first_line = line.split("\t")
        min = [float("inf") for x in range(len(first_line))]
        max = [float("-inf") for x in range(len(first_line))]
        break
    for line in f:
        if not line is "":
            line = line.rstrip()
            line_values = line.split("\t")
            values = []
            for column_idx in range(1, len(line_values)):
                v = float(line_values[column_idx])
                if v < min[column_idx]:
                    min[column_idx] = float(v)
                if v > max[column_idx]:
                    max[column_idx] = float(v)
                values.append(float(v))
            data.append(values)
            identifiers.append(line_values[0])

    scaled_data = []
    for y in range(len(data)):
        scaled_row = []
        for x in range(len(data[0])):
            scale_factor = max[x] - min[x]
            v = data[y][x]
            v = (v - min[x]) / scale_factor
            scaled_row.append(v)
        scaled_data.append(scaled_row)
    for x in range(len(data)):
        scaled_row = scaled_data[x]
        coordinates_by_identifier[identifiers[x]] = scaled_row

    return scaled_data, identifiers, coordinates_by_identifier


def read_data_generalized_using_array(file_name):
    f = open(file_name, 'r')
    file_data = f.readlines()
    identifiers = []
    coordinates_by_identifier = {}
    min = []
    max = []
    for line in file_data:
        line = line.rstrip()
        first_line = line.split("\t")
        min = [float("inf") for x in range(len(first_line) - 1)]
        max = [float("-inf") for x in range(len(first_line) - 1)]
        break

    data = np.arange(len(file_data) * (len(first_line) - 1)).reshape(len(file_data), len(first_line) - 1).astype(np.float32)

    for row_idx in range(len(file_data)):
        if not file_data[row_idx] is "":
            line = file_data[row_idx].rstrip()
            line_values = line.split("\t")
            values = []
            for column_idx in range(1, len(line_values)):
                v = float(line_values[column_idx])
                if v < min[column_idx - 1]:
                    min[column_idx - 1] = v
                if v > max[column_idx - 1]:
                    max[column_idx - 1] = v
                data[row_idx, column_idx - 1] = v
                values.append(v)
            identifiers.append(line_values[0])

    for y in range(len(data)):
        for x in range(len(data[0])):
            scale_factor = max[x] - min[x]
            v = data[y, x]
            v = (v - float(min[x])) / scale_factor
            data[y, x] = v

    for x in range(len(data)):
        scaled_row = data[x]
        coordinates_by_identifier[identifiers[x]] = scaled_row

    return data, identifiers, coordinates_by_identifier


def read_data_generalized_using_array_no_scaling(file_name, fold_change, number_of_samples):
    f = open(file_name, 'r')
    file_data = f.readlines()
    #print "fold_change"
    fold_change = float(fold_change)
    #print fold_change

    #print "number_of_samples"
    #print number_of_samples
    identifiers = []
    coordinates_by_identifier = {}
    line = file_data[0]
    line = line.rstrip()
    first_line = line.split("\t")

    data = np.arange(len(file_data) * (len(first_line) - 1)).reshape(len(file_data), len(first_line) - 1).astype(np.float32)
    for row_idx in range(len(file_data)):
        if file_data[row_idx] is "":
            True
        else:
            line = file_data[row_idx].rstrip()
            line_values = line.split("\t")
            #print "line_values"
            #print line_values
            values = []
            for column_idx in range(1, len(line_values)):

                v = float(line_values[column_idx])

                data[row_idx, column_idx - 1] = v
                values.append(v)
            identifiers.append(line_values[0])
    filtered_identifiers = []
    scaled_data = np.arange(len(file_data) * (len(first_line) - 1)).reshape(len(file_data), len(first_line) - 1).astype(np.float32)
    number_of_rows_to_keep = 0
    for row in range(len(data)):
        sorted_row = np.sort(data[row])
        ten_percent = math.floor(len(sorted_row) / 10)
        sorted_row = sorted_row[ten_percent:] # remove the lowest 10%
        sorted_row = sorted_row[:len(sorted_row) - ten_percent] # remove the highest 10%

        mean_expression = sorted_row.mean()
        #print "mean_expression"
        #print mean_expression
        s = np.log2(data[row] / mean_expression)
        c_count = 0
        for col in s:
            #print "col"
            #print abs(col)
            #print fold_change
            if abs(col) >= fold_change:
                #print "col is bigger"
                c_count += 1
        #print "c_count"
        #print c_count
        if c_count >= number_of_samples:
            scaled = (data[row] / data[row].mean()) - data[row].std()
            scaled_data[number_of_rows_to_keep] = scaled
            coordinates_by_identifier[identifiers[row]] = scaled
            filtered_identifiers.append(identifiers[row])
            number_of_rows_to_keep += 1

    #print "number_of_rows_to_keep"
    #print number_of_rows_to_keep
    return scaled_data[:number_of_rows_to_keep], filtered_identifiers, coordinates_by_identifier


def read_data_from_rutgers(file_name):
    f = open(file_name, 'r')
    transformed_data = ""
    for line in f:
        if not line is "":
            line = line.rstrip()
            line_values = line.split("\t")
            gene_identifier = line_values[0]
            for column_idx in range(1, len(line_values) - 1):
                transformed_data += str(column_idx) + "," + str(line_values[column_idx]) + "," + str(gene_identifier) + "\n"

    f_write = open(file_name + ".transformed", "w")
    f_write.write(transformed_data)
    f_write.close()


def calculate_adjacency_matrix_and_nearest_neighbors_list(data, identifiers):
    nearest_neighbors = {}
    adj_matrix = [[0.0 for x in xrange(len(data))] for x in xrange(len(data))]
    for point_idx in range(0, len(data)):
        distances = []
        for point_idy in range(0, len(data)):
            distance = 0.0
            for coordinate in range(len(data[point_idx])):
                distance += pow(data[point_idx][coordinate] - data[point_idy][coordinate], 2)
            distance = sqrt(distance)
            adj_matrix[point_idx][point_idy] = distance
            distances.append({'distance': distance, 'point_idx': str(identifiers[point_idy])})

        ordered_distances = sorted(distances, key=lambda k: k['distance'])
        nearest_neighbors[identifiers[point_idx]] = ordered_distances

    return adj_matrix, nearest_neighbors


def calculate_adjacency_matrix_and_nearest_neighbors_list_using_arrays(data, identifiers):
    nearest_neighbors = {}
    adj_matrix = np.arange(len(data) * len(data)).reshape(len(data), len(data)).astype(np.float64)
    for point_idx in range(0, len(data)):
        distances = []
        for point_idy in range(0, len(data)):
            d = data[point_idx] - data[point_idy]
            d = d ** 2
            d = d.sum()
            d = sqrt(d)
            adj_matrix[point_idx, point_idy] = d
            distances.append({'distance': d, 'point_idx': str(identifiers[point_idy])})

        ordered_distances = sorted(distances, key=lambda k: k['distance'])
        nearest_neighbors[identifiers[point_idx]] = ordered_distances

    return adj_matrix, nearest_neighbors


def create_graph_from_adjacency_matrix(adj_matrix, identifiers):
    num_points = len(identifiers)
    gr = graph()
    gr.add_nodes(identifiers)
    for idx in range(num_points):
        for idy in range(idx, num_points):
            if idx != idy:
                gr.add_edge((identifiers[idx], identifiers[idy]), adj_matrix[idx][idy])

    return gr


def calculate_edge_length_statistics(graph, coordinates_by_identifier):
    total_edge_lengths = 0.0
    shortest_edge = float("inf")
    longest_edge = float("-inf")
    for edge in graph.edges():
        edge_length = distance_between_points(coordinates_by_identifier[edge[0]], coordinates_by_identifier[edge[1]])
        total_edge_lengths += edge_length
        if edge_length > longest_edge:
            longest_edge = edge_length
        if edge_length < shortest_edge:
            shortest_edge = edge_length
    average_edge_length = total_edge_lengths / len(graph.edges())
    return average_edge_length, shortest_edge, longest_edge


def create_graph_from_k_nearest_neighbors(nearest_neighbors, identifiers, k):
    gr = graph()
    gr.add_nodes(identifiers)
    edges_seen = {}
    for root_node_idx, neighbors in nearest_neighbors.items():
        for neighbor_idx in range(1, k):
            neighbor = neighbors[neighbor_idx]
            edge_name = root_node_idx + "-" + neighbor['point_idx']
            reverse_edge_name = neighbor['point_idx'] + "-" + root_node_idx

            if edge_name in edges_seen or reverse_edge_name in edges_seen:
                True
            else:
                gr.add_edge((root_node_idx, neighbor['point_idx']), neighbor['distance'])
                edges_seen[edge_name] = True
                edges_seen[reverse_edge_name] = True

    return gr, edges_seen


def create_tree_from_forrest(graph, mst, edges_seen, coordinates_by_identifier):
    clusters = {}
    edges_seen = {}
    for key, value in mst.items():
        if value is None:
            clusters[key] = calculate_tree_cluster_centroid(graph, key, coordinates_by_identifier)

    second_clusters_seen = {}
    for first_cluster_key in clusters:
        first_cluster = clusters[first_cluster_key]
        distance_to_nearest_cluster = float('inf')
        second_cluster_idx = None
        for second_cluster_key in clusters:
            if second_cluster_key in second_clusters_seen:
                True
            else:
                second_cluster = clusters[second_cluster_key]
                if first_cluster_key != second_cluster_key:
                    distance = distance_between_points(first_cluster['centroid'], second_cluster['centroid'])
                    if distance < distance_to_nearest_cluster:
                        distance_to_nearest_cluster = distance
                        second_cluster_idx = second_cluster_key

        if second_cluster_idx != None:
            distance_between_closest_points = float('inf')
            first_point_idx = None
            second_point_idx = None
            for point_in_first_cluster in first_cluster["points"]:
                for point_in_second_cluster in clusters[second_cluster_idx]["points"]:
                    distance = distance_between_points(coordinates_by_identifier[point_in_first_cluster], coordinates_by_identifier[point_in_second_cluster])
                    if distance < distance_between_closest_points:
                        distance_between_closest_points = distance
                        first_point_idx = point_in_first_cluster
                        second_point_idx = point_in_second_cluster

            edge_name = first_point_idx + "-" + second_point_idx
            reverse_edge_name = second_point_idx + "-" + first_point_idx
            if edge_name in edges_seen or reverse_edge_name in edges_seen:
                True
            else:
                graph.add_edge((first_point_idx, second_point_idx), distance_between_closest_points)
                second_clusters_seen[second_cluster_key] = True
                second_clusters_seen[first_cluster_key] = True
                edges_seen[edge_name] = True
                edges_seen[reverse_edge_name] = True

    return graph


def create_tree_from_forrest_using_arrays(graph, mst, edges_seen, coordinates_by_identifier):
    clusters = {}
    edges_seen = {}
    for key, value in mst.items():
        if value is None:
            clusters[key] = calculate_tree_cluster_centroid_using_arrays(graph, key, coordinates_by_identifier)


    second_clusters_seen = {}
    for first_cluster_key in clusters:
        first_cluster = clusters[first_cluster_key]
        distance_to_nearest_cluster = float('inf')
        second_cluster_idx = None
        for second_cluster_key in clusters:
            if second_cluster_key in second_clusters_seen:
                True
            else:
                second_cluster = clusters[second_cluster_key]
                if first_cluster_key != second_cluster_key:
                    distance = distance_between_points_using_arrays(first_cluster['centroid'], second_cluster['centroid'])
                    if distance < distance_to_nearest_cluster:
                        distance_to_nearest_cluster = distance
                        second_cluster_idx = second_cluster_key

        if second_cluster_idx != None:
            distance_between_closest_points = float('inf')
            first_point_idx = None
            second_point_idx = None
            for point_in_first_cluster in first_cluster["points"]:
                for point_in_second_cluster in clusters[second_cluster_idx]["points"]:
                    distance = distance_between_points_using_arrays(coordinates_by_identifier[point_in_first_cluster], coordinates_by_identifier[point_in_second_cluster])
                    if distance < distance_between_closest_points:
                        distance_between_closest_points = distance
                        first_point_idx = point_in_first_cluster
                        second_point_idx = point_in_second_cluster

            edge_name = first_point_idx + "-" + second_point_idx
            reverse_edge_name = second_point_idx + "-" + first_point_idx
            if edge_name in edges_seen or reverse_edge_name in edges_seen:
                True
            else:
                graph.add_edge((first_point_idx, second_point_idx), distance_between_closest_points)
                second_clusters_seen[second_cluster_key] = True
                second_clusters_seen[first_cluster_key] = True
                edges_seen[edge_name] = True
                edges_seen[reverse_edge_name] = True

    return graph


def calculate_tree_cluster_centroid(graph, tree_root_node, coordinates_by_identifier):
    sub_mst = minimal_spanning_tree(graph, tree_root_node)
    point_idx = 0.0
    points_in_cluster = []
    coordinate_vals = [0.0 for x in range(len(coordinates_by_identifier[tree_root_node]))]
    for key, value in sub_mst.items():

        for coordinate in range(len(coordinates_by_identifier[key])):
            if coordinate in coordinate_vals:
                coordinate_vals[coordinate] += coordinates_by_identifier[key][coordinate]
        points_in_cluster.append(key)
        point_idx += 1.0

    for coordinate_idx in range(len(coordinate_vals)):
        coordinate_vals[coordinate_idx] = coordinate_vals[coordinate_idx] / point_idx

    return {"centroid": coordinate_vals, "points": points_in_cluster}


def calculate_tree_cluster_centroid_using_arrays(graph, tree_root_node, coordinates_by_identifier):
    sub_mst = minimal_spanning_tree(graph, tree_root_node)
    point_idx = 0.0
    points_in_cluster = []

    coordinate_vals = np.zeros(len(coordinates_by_identifier[tree_root_node])).astype(np.float64)
    for key, value in sub_mst.items():

        coordinate_vals += coordinates_by_identifier[key]
        points_in_cluster.append(key)
        point_idx += 1.0

    coordinate_vals = coordinate_vals / point_idx

    return {"centroid": coordinate_vals, "points": points_in_cluster}


def construct_reverse_lookup(graph):
    edges = graph.edges()
    reverse_map = {}
    for k, v in edges:
        if reverse_map.has_key(k):
            reverse_map[k].append(v)
        else:
            reverse_map[k] = [v]

    return reverse_map


def plot_data_with_mst_graph(data, graph, coordinates_by_identifier):
    array_of_data = array(data)
    clf()
    plot(array_of_data[:, 0], array_of_data[:, 1], 'sg', markersize=8)
    sol = []
    for idx in range(len(data)):
        sol.append(1)

    idx = 0
    for edge in graph.edges():
        if sol[idx] is 1:
            pyplot.plot([coordinates_by_identifier[edge[0]][0], coordinates_by_identifier[edge[1]][0]],
                        [coordinates_by_identifier[edge[0]][1], coordinates_by_identifier[edge[1]][1]])
        else:
            graph.del_edge(edge)

    plt.show()


def plot_data(data):
    clf()
    plot(data[:, 0], data[:, 1], 'sg', markersize=8)
    plt.show()


def add_node_to_cluster_visitor(cluster, node):
    cluster.append(node)


def traverse_edges(reverse_map, root_node_id, cluster):
    if root_node_id in reverse_map:
        for leaf_node_id in reverse_map[root_node_id]:
            cluster.append(leaf_node_id)
            traverse_edges(reverse_map, leaf_node_id, cluster)
    return cluster


def traverse_edges_to_find_tree(reverse_map, root_node_id, cluster, root_node_indexes, sol, nodes_seen):
    if root_node_id in reverse_map:
        for leaf_node_id in reverse_map[root_node_id]:
            edge_index = root_node_indexes[leaf_node_id]

            if edge_index == len(sol):
                if not (leaf_node_id in nodes_seen):
                    cluster.append(leaf_node_id)
                    nodes_seen[leaf_node_id] = True
                    return cluster
            else:
                if sol[edge_index] == 1:
                    if not (leaf_node_id in nodes_seen):
                        cluster.append(leaf_node_id)
                        traverse_edges_to_find_tree(reverse_map, leaf_node_id, cluster, root_node_indexes, sol, nodes_seen)

    return cluster


def generate_root_node_index_lookup(tree):
    root_node_indexes = {}
    for idx, root_node in enumerate(tree):
        root_node_indexes[root_node] = idx

    return root_node_indexes


def decode_solution(candidate, ordered_reverse_map, root_node_indexes, cluster, gst, identifiers):
    edges = gst.edges()
    nodes = gst.nodes()
    clusters = []
    for idx in range(0, len(candidate)):
        if candidate[idx] is 0:
            clusters.append({nodes[idx] : True})
        else:
            inserted_into_existing_cluster = False
            for cluster in clusters:
                if candidate[idx] in cluster or nodes[idx] in cluster:
                    cluster[candidate[idx]] = True
                    cluster[nodes[idx]] = True
                    inserted_into_existing_cluster = True
                    break

            if not inserted_into_existing_cluster:
                new_cluster = {nodes[idx]: True, candidate[idx]: True}
                clusters.append(new_cluster)

    while True:
        clusters_updated = False
        first_cluster_idx = 0
        while True:
            if first_cluster_idx >= len(clusters):
                break
            first_cluster = clusters[first_cluster_idx]
            second_cluster_idx = first_cluster_idx + 1
            while True:
                if second_cluster_idx >= len(clusters):
                    break
                second_cluster = clusters[second_cluster_idx]
                found_cluster_to_add_onto = False
                for node_id in second_cluster:
                    if node_id in first_cluster:
                        first_cluster.update(second_cluster)
                        found_cluster_to_add_onto = True
                        clusters_updated = True
                        del clusters[second_cluster_idx]
                        break
                if not found_cluster_to_add_onto:
                    second_cluster_idx += 1
            first_cluster_idx += 1
        if not clusters_updated:
            break

    clusters_of_points = []
    for cluster in clusters:
        clusters_of_points.append(cluster.keys())
    return clusters_of_points


def decode_solution2(candidate, ordered_reverse_map, root_node_indexes, cluster, gst, identifiers):
    clusters = []
    edges = gst.edges()
    number_of_zeros = 0
    for idx in range(len(candidate)):
        if candidate[idx] == 0:
            number_of_zeros += 1
            first_node_in_existing_cluster = False
            second_node_in_existing_cluster = False
            for cluster in clusters:
                if cluster:
                    if edges[idx][0] in cluster:
                        first_node_in_existing_cluster = True
                    if edges[idx][1] in cluster:
                        second_node_in_existing_cluster = True

            if not first_node_in_existing_cluster:
                clusters.append([edges[idx][0]])
            if not second_node_in_existing_cluster:
                clusters.append([edges[idx][1]])
        else:
            first_cluster_idx = -1
            second_cluster_idx = -1
            second_cluster = []
            cluster_idx = 0
            for cluster in clusters:
                if cluster:
                    if edges[idx][0] in cluster:
                        first_cluster_idx = cluster_idx
                    if edges[idx][1] in cluster:
                        second_cluster_idx = cluster_idx
                        second_cluster = cluster
                cluster_idx += 1
            if (first_cluster_idx != -1) and (second_cluster_idx != -1):
                clusters[first_cluster_idx] += clusters[second_cluster_idx]
                clusters[second_cluster_idx] = None
            elif (first_cluster_idx == -1) and (second_cluster_idx != -1):
                clusters[second_cluster_idx].append(edges[idx][0])
            elif (first_cluster_idx != -1) and (second_cluster_idx == -1):
                clusters[first_cluster_idx].append(edges[idx][1])
            elif (first_cluster_idx == -1) and (second_cluster_idx == -1):
                clusters.append([edges[idx][0], edges[idx][1]])

    clean_clusters = []
    for cluster in clusters:
        if cluster:
            clean_clusters.append(cluster)
    return clean_clusters


def decode_solution_1(candidate, ordered_reverse_map, root_node_indexes, cluster, gst, identifiers):
    gr = graph()
    gr.add_nodes(identifiers)

    cluster_idx = 0
    edges = gst.edges()
    nodes_seen = {}
    if candidate[cluster_idx] == 0:
        edge = edges[0]
        for node in ordered_reverse_map[cluster[cluster_idx]]:
            if node in edge:
                clusters = [[node]]
                nodes_seen[node] = True
                cluster_idx += 1

    clusters.append([cluster[0]])
    traverse_edges_to_find_tree(ordered_reverse_map, clusters[cluster_idx][0], clusters[cluster_idx], root_node_indexes, candidate, nodes_seen)
    for idx in range(1, len(candidate)):
        if candidate[idx] == 0:
            cluster_idx += 1
            new_cluster = [cluster[idx]]

            traverse_edges_to_find_tree(ordered_reverse_map, new_cluster[0], new_cluster, root_node_indexes, candidate, nodes_seen)
            clusters.append(new_cluster)

    return clusters


def plot_clustering_solution(solution, original_data_set):
    #plot_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plot_colors = ['b', 'g', 'y']
    #marker_types = ['-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    marker_types = ['o', 'v', '^', '<', '>']
    color_idx = 0
    marker_idx = 0
    clf()
    for cluster in solution:
        if cluster:
            for point in cluster:
                marker = plot_colors[color_idx] + marker_types[marker_idx]
                plt.plot(original_data_set[point][0], original_data_set[point][1], marker)
            color_idx += 1
            marker_idx += 1
            if color_idx == len(plot_colors):
                color_idx = 0
            if marker_idx == len(marker_types):
                marker_idx = 0
    plt.show()


def distance_between_points(initial_point, end_point):
    for coordinate in range(len(initial_point)):
        dist = pow(initial_point[coordinate] - end_point[coordinate], 2)
    dist = sqrt(dist)
    return dist


def distance_between_points_using_arrays(initial_point, end_point):
    distance = initial_point - end_point
    distance = distance ** 2
    distance = distance.sum()
    distance = sqrt(distance)
    return distance


def combine_degenerate_clusters(solution, minimum_cluster_size, coordinates_by_identifier):
    clusters = []
    degenerate_clusters = []
    for cluster in solution:
        if len(cluster) > minimum_cluster_size:
            clusters.append(compute_cluster_centroid(cluster, coordinates_by_identifier))
        else:
            degenerate_clusters.append(compute_cluster_centroid(cluster, coordinates_by_identifier))

    for cluster in degenerate_clusters:
        reassign_points_in_degenerate_cluster(clusters, cluster, coordinates_by_identifier)

    return(clusters)


def reassign_points_in_degenerate_cluster(clusters, degenerate_cluster, coordinates_by_identifier):
    for point in degenerate_cluster["points_in_cluster"]:
        nearest_cluster = {}
        distance_to_nearest_cluster = float("inf")
        for cluster in clusters:
            distance_to_centroid = distance_between_points_using_arrays(cluster["centroid"], coordinates_by_identifier[point])
            if distance_to_centroid < distance_to_nearest_cluster:
                distance_to_nearest_cluster = distance_to_centroid
                nearest_cluster = cluster
        if distance_to_nearest_cluster != float("inf"):
            nearest_cluster["points_in_cluster"].append(point)



def convert_clustering_solution_to_sklearn_format_and_calculate_silhouette_score(clusters, coordinates_by_identifier):
    points = np.array([[0.0] * len(coordinates_by_identifier[clusters[0]["points_in_cluster"][0]])] * len(coordinates_by_identifier), ndmin=2)
    labels = np.array([0] * len(coordinates_by_identifier))
    point_idx = 0
    cluster_idx = 0
    for cluster in clusters:
        for point_id in cluster["points_in_cluster"]:
            points[point_idx]  = coordinates_by_identifier[point_id]

            labels[point_idx] = cluster_idx
            point_idx += 1
        cluster_idx += 1


    return metrics.silhouette_score(points, labels, metric='euclidean')


def compute_cluster_centroid(cluster, coordinates_by_identifier):
    point_idx = 0.0
    points_in_cluster = []

    coordinate_vals = np.zeros(len(coordinates_by_identifier[cluster[0]])).astype(np.float64)
    for point_id in cluster:

        coordinate_vals += coordinates_by_identifier[point_id]
        points_in_cluster.append(point_id)
        point_idx += 1.0

    coordinate_vals = coordinate_vals / point_idx

    return({"centroid": coordinate_vals, "points_in_cluster":points_in_cluster})


