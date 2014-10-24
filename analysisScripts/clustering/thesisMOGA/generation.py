__author__ = 'mshaw'


import evaluation
import math

def generate_candidates(random, args):
    ordered_reverse_map = args.get('ordered_reverse_map')
    nearest_neighbors = args.get('nearest_neighbors')
    coordinates_by_identifier = args.get('coordinates_by_identifier')
    cluster = args.get('cluster')
    bounding_box = args.get('bounding_box')
    gst = args.get('gst')
    average_edge_length = args.get('average_edge_length')
    use_arrays = args.get('use_arrays')
    p = []

    coin = random.randint(0, 6)
    if coin == 10:
        upper_bound_for_k = 10
        value_for_k = random.randint(5, upper_bound_for_k)
        clusters, point_to_cluster_idx = generate_clusters_by_kmeans(coordinates_by_identifier, value_for_k, random, bounding_box)
        p = recluster_using_mst(point_to_cluster_idx, cluster)
    elif coin == 1:
        #p = generate_candidates_randomly(len(cluster) - 1, random)
        p = generate_candidate_randomly_using_locus(gst, nearest_neighbors, random)
    #elif coin == 0:
        #p = [0 for x in range(len(cluster) - 1)]
    #elif coin == 2:
        #p = generate_candidates_by_removing_longer_than_average_edges(gst, coordinates_by_identifier, average_edge_length, random, use_arrays)
    else:
        value_for_k = random.randint(10, 20)
        #p = generate_candidates_by_removing_interesting_edges(gst, nearest_neighbors, value_for_k, cluster, random, ordered_reverse_map)
        p = generate_candidate_by_mst_interesting(gst, ordered_reverse_map, nearest_neighbors, value_for_k, random)

    return p


def generate_candidate_by_mst_interesting(gst, ordered_reverse_map, nearest_neighbors, neighborhood_size, random):
    edges = gst.edges()
    nodes = gst.nodes()
    solution = [0 for idx in range(len(edges)  + 1)]
    for edge in edges:
        n = 0
        value_to_append = edge[0]
        if edge[0] in ordered_reverse_map and edge[1] in ordered_reverse_map:
            for nearest_neighbor in nearest_neighbors[edge[0]]:
                if n < neighborhood_size:
                    if nearest_neighbor['point_idx'] == edge[1]:
                        break
                else:
                    coin = random.randint(0, 1)
                    if coin == 0:
                        value_to_append = 0
                    break
                n += 1

            n = 0
            for nearest_neighbor in nearest_neighbors[edge[1]]:
                if n < neighborhood_size:
                    if nearest_neighbor['point_idx'] == edge[0]:
                        break
                else:
                    coin = random.randint(0, 1)
                    if coin == 0:
                        value_to_append = 0
                    break
                n += 1
        solution[nodes.index(edge[1])] = value_to_append
    return(solution)


def generate_candidate_randomly_using_locus(gst, nearest_neighbors, random):
    edges = gst.edges()
    nodes = gst.nodes()
    upper_bound = math.floor(len(nodes) / 2)
    solution = [0 for idx in range(len(edges)  + 1)]
    for idx in range(len(solution)):
        coin = random.randint(0, upper_bound)
        if coin > upper_bound / 2:
            solution[idx] = 0
        else:
            solution[idx] = nearest_neighbors[nodes[idx]][coin]['point_idx']
    return(solution)


def generate_candidates_randomly(number_of_edges, rand):
    c = []
    for idx in range(number_of_edges):
        coin = rand.randint(0,1)
        if coin == 1:
            c.append(1)
        else:
            c.append(0)
    return c


def generate_candidates_by_removing_longer_than_average_edges(gst, coordinates_by_identifier, average_edge_length, random, use_arrays):
    solution = []
    num_clusters = 0
    edges = gst.edges()
    for edge in edges:
        value_to_append = 1
        if use_arrays:
            dist = evaluation.distance_between_points_using_arrays(coordinates_by_identifier[edge[0]], coordinates_by_identifier[edge[1]])
        else:
            dist = evaluation.distance_between_points(coordinates_by_identifier[edge[0]], coordinates_by_identifier[edge[1]])
        if dist > average_edge_length:
            coin = random.randint(0, 10)
            if coin == 0:
                value_to_append = 0
                num_clusters += 1
        solution.append(value_to_append)

    return solution


def generate_candidates_by_removing_interesting_edges(gst, nearest_neighbors, neighborhood_size, cluster, random, ordered_reverse_map):
    solution = []
    number_of_clusters = 0
    for edges in gst.edges():
        n = 0
        value_to_append = 1
        found_connection = False
        if edges[0] in ordered_reverse_map and edges[1] in ordered_reverse_map:

            for nearest_neighbor in nearest_neighbors[edges[0]]:
                if n < neighborhood_size:
                    if nearest_neighbor['point_idx'] == edges[1]:
                        found_connection = True
                        break
                else:
                    coin = random.randint(0, 1)
                    if coin == 0:
                        value_to_append = 0
                    break
                n += 1

            n = 0
            for nearest_neighbor in nearest_neighbors[edges[1]]:
                if n < neighborhood_size:
                    if nearest_neighbor['point_idx'] == edges[0]:
                        found_connection = True
                        break
                else:
                    coin = random.randint(0, 1)
                    if coin == 0:
                        value_to_append = 0
                    break
                n += 1

        if value_to_append is 0:
            number_of_clusters += 1
        solution.append(value_to_append)

    return solution


def generate_clusters_by_kmeans(data, number_of_clusters, random, bounding_box):
    centers = []
    for idx in range(number_of_clusters):
        centers.append([random.uniform(bounding_box["x_min"], bounding_box["x_max"]), random.uniform(bounding_box["y_min"], bounding_box["y_max"])])

    for idx in range(5):
        clusters, point_to_cluster_idx = assign_points_to_clusters(centers, data)
        centers = update_cluster_centers(clusters, data)

    assign_points_to_clusters(centers, data)

    return clusters, point_to_cluster_idx


def update_cluster_centers(clusters, data):
    centers = []
    for cluster in clusters:
        if len(cluster) > 0:
            coordinate_vals = []
            for point in cluster:
                for coordinate in range(len(data[point])):
                    if coordinate in coordinate_vals:
                        coordinate_vals[coordinate] += data[point][coordinate]
                    else:
                        coordinate_vals.append(data[point][coordinate])

            for coordinate_idx in range(len(coordinate_vals)):
                coordinate_vals[coordinate_idx] = coordinate_vals[coordinate_idx] / len(cluster)
            centers.append(coordinate_vals)
    return centers


def assign_points_to_clusters(centers, data, use_arrays):
    new_clusters = [[] for i in range(len(centers))]
    point_to_cluster_idx = {}
    for point in data:
        nearest_cluster_distance = float('inf')
        idx = 0
        cluster_idx = idx
        for centroid in centers:
            if use_arrays:
                d = evaluation.distance_between_points_using_arrays(centroid, data[point])
            else:
                d = evaluation.distance_between_points(centroid, data[point])
            if d < nearest_cluster_distance:
                nearest_cluster_distance = d
                cluster_idx = idx
            idx += 1

        new_clusters[cluster_idx].append(point)
        point_to_cluster_idx[point] = cluster_idx
    return new_clusters, point_to_cluster_idx


def recluster_using_mst(point_to_cluster_idx, mst):
    cluster_idx = point_to_cluster_idx[mst[0]]
    candidate = []
    for idx in range(1, len(mst)):
        if point_to_cluster_idx[mst[idx]] == cluster_idx:
            candidate.append(1)
        else:
            candidate.append(0)
        cluster_idx = point_to_cluster_idx[mst[idx]]

    return candidate

