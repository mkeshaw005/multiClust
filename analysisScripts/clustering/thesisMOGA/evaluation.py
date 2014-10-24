__author__ = 'mshaw'

from utilities import decode_solution
from math import pow, sqrt
import numpy as np


def evaluator(candidates, args):
    emo = args.get('emo')
    ordered_reverse_map = args.get('ordered_reverse_map')
    root_node_indexes = args.get('root_node_indexes')
    cluster = args.get('cluster')
    coordinates_by_identifier = args.get('coordinates_by_identifier')
    num_edges = args.get('num_edges')
    average_edge_length = args.get('average_edge_length')
    longest_edge = args.get('longest_edge')
    use_arrays = args.get('use_arrays')

    gst = args.get('gst')
    identifiers = args.get('identifiers')
    fitness = []
    for c in candidates:
        clusters = decode_solution(c, ordered_reverse_map, root_node_indexes, cluster, gst, identifiers)

        #f0 = evaluate_by_number_of_clusters_high_cluster_granularity(clusters, num_edges)
        f1 = evaluate_by_average_distance_to_nearest_neighbor(clusters, coordinates_by_identifier)
        #f2 = evaluate_by_number_of_clusters_high_cluster_size(clusters, num_edges)
        #f3 = evaluate_by_separation_using_average_distance_to_nearest_centroid(clusters, coordinates_by_identifier, use_arrays, longest_edge)
        f4 = evaluate_by_compactness_using_average_distance_to_centroid(clusters, coordinates_by_identifier, use_arrays, average_edge_length, longest_edge)
        fitness.append(emo.Pareto([f1, f4]))
    return fitness


def evaluate_by_number_of_clusters_high_cluster_granularity(clustering_solution, total_number_of_points):
    return float(len(clustering_solution)) / float(total_number_of_points)


def evaluate_by_number_of_clusters_high_cluster_size(solutions, total_number_of_points):
    return float(total_number_of_points) / float(len(solutions)) * (1.0/float(total_number_of_points))


def evaluate_by_average_distance_to_nearest_neighbor(solution, coordinates_by_identifier):
    average_nearest_neighbors_for_all_clusters = 0.00001
    for cluster in solution:
        average_nearest_neighbor = calculate_average_nearest_neighbor(cluster, coordinates_by_identifier, True)

        average_nearest_neighbors_for_all_clusters += average_nearest_neighbor

    return 1.0 / (average_nearest_neighbors_for_all_clusters / len(solution))



def evaluate_by_compactness_using_average_distance_to_centroid(solution, coordinates_by_identifier, use_arrays, average_edge_length, longest_edge):
    average_distance_to_centroids = 0.0
    for cluster in solution:
        if len(cluster) == 1:
            average_distance_to_centroids += longest_edge ** 2
        else:
            centroid = calculate_centroid(cluster, coordinates_by_identifier)
            avg_dist_to_centroids = 0.0
            for point in cluster:
                if use_arrays:
                    avg_dist_to_centroids += distance_between_points_using_arrays(centroid, coordinates_by_identifier[point])
                else:
                    avg_dist_to_centroids += distance_between_points(centroid, coordinates_by_identifier[point])
            average_distance_to_centroids += avg_dist_to_centroids / len(cluster)

    return 1.0 / (average_distance_to_centroids / len(solution))



def evaluate_by_separation_using_average_distance_to_nearest_centroid(solution, coordinates_by_identifier, use_arrays, longest_edge):
    average_distance_between_nearest_centroids = 0.0
    centroids = []
    for cluster in solution:
        if use_arrays:
            centroids.append(calculate_centroid_using_arrays(cluster, coordinates_by_identifier))
        else:
            centroids.append(calculate_centroid(cluster, coordinates_by_identifier))
    for centroid_one_idx in range(0, len(centroids)):
        distance_to_nearest_centroid = float('inf')
        for centroid_two_idx in range(0, len(centroids)):
            if centroid_one_idx != centroid_two_idx:
                if use_arrays:
                    distance = distance_between_points_using_arrays(centroids[centroid_one_idx], centroids[centroid_two_idx])
                else:
                    distance = distance_between_points(centroids[centroid_one_idx], centroids[centroid_two_idx])
                if distance < distance_to_nearest_centroid:
                    distance_to_nearest_centroid = distance
        average_distance_between_nearest_centroids += distance_to_nearest_centroid

    # penalize degenerate clusters containing just a single element
    if average_distance_between_nearest_centroids is 0.0:
        average_distance_between_nearest_centroids = longest_edge ** 2
    return 1.0 / (average_distance_between_nearest_centroids / len(solution))



def calculate_average_nearest_neighbor(cluster, coordinates_by_identifier, use_arrays):
    sum_of_all_shortest_distances = 0.0
    for initial_point in cluster:
        shortest_distance = float("inf")
        for end_point in cluster:
            if initial_point != end_point:
                if use_arrays:
                    distance = distance_between_points_using_arrays(coordinates_by_identifier[initial_point], coordinates_by_identifier[end_point])
                else:
                    distance = distance_between_points(coordinates_by_identifier[initial_point], coordinates_by_identifier[end_point])
                if distance < shortest_distance:
                    shortest_distance = distance
        if shortest_distance != float("inf"):
            sum_of_all_shortest_distances += shortest_distance

    return sum_of_all_shortest_distances / len(cluster)


def evaluate_by_density(solution, coordinates_by_identifier):
    average_distances_for_all_clusters = 0
    for cluster in solution:
        if len(cluster) == 1:
            average_distances_for_all_clusters += 100
        else:
            cluster_centroid = calculate_centroid(cluster, coordinates_by_identifier)
            average_distances_for_all_clusters += calculate_average_distance_from_centroid(cluster, cluster_centroid)

    return 1.0 / (average_distances_for_all_clusters / len(solution))



def calculate_centroid(cluster, point_coordinates):
    values = [0.0 for x in range(len(point_coordinates[cluster[0]]))]

    for point in cluster:
        for coordinate in range(len(point_coordinates[point])):
            values[coordinate] += point_coordinates[point][coordinate]

    for coordinate_idx in range(len(values)):
        values[coordinate_idx] = values[coordinate_idx] / len(cluster)
    return values


def calculate_centroid_using_arrays(cluster, point_coordinates):
    values = np.zeros(len(point_coordinates[cluster[0]])).astype(np.float64)
    point_idx = 0.0
    for point in cluster:
        values += point_coordinates[point]
        point_idx += 1.0

    values / len(cluster)
    return values


def calculate_average_distance_from_centroid(cluster, centroid, use_arrays):
    distances_from_centroid = 0
    for point in cluster:
        if use_arrays:
            distances_from_centroid += distance_between_points_using_arrays(point, centroid)
        else:
            distances_from_centroid += distance_between_points(point, centroid)

    return distances_from_centroid / len(cluster)


def distance_between_points(initial_point, end_point):
    dist = 0.0
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