__author__ = 'mshaw'
import utilities
import generation
import evaluation
from pygraph.algorithms.minmax import minimal_spanning_tree
from pygraph.classes.digraph import digraph
from collections import OrderedDict
from random import Random
from inspyred import ec
from time import time

import math
import sys
import json


def cluster_data(file_path, file_name, fold_change, number_of_samples, population_size, num_iterations, mutation_rate):
    start = time()
    path = file_path + "/" + file_name
    data, identifiers, coordinates_by_identifier = utilities.read_data_generalized_using_array_no_scaling(path, fold_change, number_of_samples)
    end = time()
    load_data_time = end - start

    start = time()
    adj_mat, nearest_neighbors = utilities.calculate_adjacency_matrix_and_nearest_neighbors_list_using_arrays(data, identifiers)
    end = time()
    construct_adjacency_matrix_time = end - start

    start = time()
    k_neighbors = int(math.floor(len(coordinates_by_identifier) / 10))
    graph, edges_seen = utilities.create_graph_from_k_nearest_neighbors(nearest_neighbors, identifiers, k_neighbors)
    end = time()
    create_graph_using_k_nearest_neighbors_time = end - start

    start = time()
    mst = minimal_spanning_tree(graph)
    gst = digraph()
    gst.add_spanning_tree(mst)
    graph = utilities.create_tree_from_forrest_using_arrays(graph, mst, edges_seen, coordinates_by_identifier)
    average_edge_length, shortest_edge, longest_edge = utilities.calculate_edge_length_statistics(graph, coordinates_by_identifier)
    mst = minimal_spanning_tree(graph, graph.nodes()[0])
    gst = digraph()
    gst.add_spanning_tree(mst)

    end = time()
    compute_mst_time = end - start

    start = time()
    rm = utilities.construct_reverse_lookup(gst)
    ordered_reverse_map = OrderedDict(sorted(rm.items(), key=lambda t: t[0]))

    cluster = [graph.nodes()[0]]
    utilities.traverse_edges(ordered_reverse_map, cluster[0], cluster)
    root_node_indexes = utilities.generate_root_node_index_lookup(cluster)
    end = time()
    precompute_lookups_time = end - start

    rand = Random()
    rand.seed(int(time()))


    ea = ec.emo.NSGA2(rand)
    #ea.variator = [ec.variators.blend_crossover, ec.variators.gaussian_mutation]
    #ec.variator = [ec.variators.bit_flip_mutation]
    ea.terminator = ec.terminators.generation_termination
    #population_size = 50
    #max_generations = 100
    #mutation_rate = 0.01
    start = time()

    final_pop = ea.evolve(generator=generation.generate_candidates,
                          evaluator=evaluation.evaluator,
                          pop_size=population_size,
                          maximize=True,
                          max_generations=num_iterations,
                          mutation_rate=mutation_rate,
                          num_elites=0,
                          cluster=cluster,
                          ordered_reverse_map=ordered_reverse_map,
                          root_node_indexes=root_node_indexes,
                          emo=ec.emo,
                          num_edges=len(coordinates_by_identifier),
                          coordinates_by_identifier=coordinates_by_identifier,
                          nearest_neighbors=nearest_neighbors,
                          bounding_box= {},
                          gst=gst,
                          average_edge_length=average_edge_length,
                          longest_edge=longest_edge,
                          identifiers=identifiers,
                          use_arrays=True)
    end = time()
    run_ga_time = end - start

    # Sort and print the best individual, who will be at index 0.
    final_pop.sort(reverse=True)
    highest_final_clustering_silhouette_score = float("-inf")
    best_solution = {}
    final_clustering = {}
    for solution in final_pop:
        sol = utilities.decode_solution(solution.candidate, ordered_reverse_map, root_node_indexes, cluster, gst, identifiers)
        min_cluster_size = math.floor(len(data) * 0.02)
        #min_cluster_size = 5
        fc = utilities.combine_degenerate_clusters(sol, min_cluster_size, coordinates_by_identifier)
        if len(fc) > 1:
            silhouette_score = utilities.convert_clustering_solution_to_sklearn_format_and_calculate_silhouette_score(fc, coordinates_by_identifier)
            if silhouette_score > highest_final_clustering_silhouette_score:
                highest_final_clustering_silhouette_score = silhouette_score
                best_solution = sol
                final_clustering = fc




    final_arc = ea.archive

    import pylab
    import matplotlib.pyplot as plt
    x = []
    y = []
    z = []
    for f in final_arc:
        x.append(f.fitness[0])
        y.append(f.fitness[1])
        sol = utilities.decode_solution(f.candidate, ordered_reverse_map, root_node_indexes, cluster, gst, identifiers)
        fc = utilities.combine_degenerate_clusters(sol, math.floor(len(data) / 20), coordinates_by_identifier)
        silhouette_score = 0
        if len(fc) > 1:
            silhouette_score = utilities.convert_clustering_solution_to_sklearn_format_and_calculate_silhouette_score(fc, coordinates_by_identifier)
        z.append(silhouette_score)


    pylab.scatter(x, y, color='b')
    pareto_plot_file_name = '{}/ParetoFront.svg'.format(file_path)
    pylab.savefig(pareto_plot_file_name, format='svg')
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x, y, z, color='b')
    #
    #ax.set_xlabel('Avg. Sep. Between Centroids')
    #ax.set_ylabel('Avg. Dist. Between Points and Centroid')
    #ax.set_zlabel('Silhouette Score')
    #plt.savefig('{}ParetoFrontWithSilhouetteScore.pdf'.format(file_path), format='pdf')
    #print "final_clustering"
    #print final_clustering
    clusters = []
    for cluster in final_clustering:
        c = {'centroid': str(cluster['centroid']),'points_in_cluster': cluster['points_in_cluster']}

        clusters.append(c)
    output = {
        'number_of_points_clustered': len(coordinates_by_identifier),
        'silhoute_score': highest_final_clustering_silhouette_score,
        'number_of_clusters': len(final_clustering),
        'population_size': population_size,
        'max_generations': num_iterations,
        'mutation_rate': mutation_rate,
        'clustering_solution': clusters,
        'pareto_plot_file_name': "ParetoFront.svg"
    }

    print(json.dumps(output))


if __name__ == '__main__':
    data_file_directory = sys.argv[1]
    file_name = sys.argv[2]
    fold_change = float(sys.argv[3])
    number_of_experiments_exceeding_fold_change = int(sys.argv[4])
    population_size = int(sys.argv[5])
    num_iterations = int(sys.argv[6])
    mutation_rate = float(sys.argv[7])
    cluster_data(data_file_directory, file_name, fold_change, number_of_experiments_exceeding_fold_change, population_size, num_iterations, mutation_rate)
