// Bridge from a stable file interface to the AGP-Dynamic reference Graph class.
#include "graph/Graph.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: agp_query_bridge GRAPH_BINARY QUERY_TEXT\n";
        return 2;
    }

    FILE *graph_file = std::fopen(argv[1], "rb");
    if (graph_file == nullptr) {
        std::cerr << "cannot open graph file\n";
        return 3;
    }
    unsigned int n = 0;
    unsigned int endpoint_count = 0;
    if (std::fread(&n, sizeof(unsigned int), 1, graph_file) != 1 ||
        std::fread(&endpoint_count, sizeof(unsigned int), 1, graph_file) != 1 ||
        endpoint_count % 2 != 0) {
        std::cerr << "invalid graph header\n";
        std::fclose(graph_file);
        return 4;
    }
    std::vector<int> endpoints(endpoint_count);
    if (endpoint_count > 0 &&
        std::fread(endpoints.data(), sizeof(int), endpoint_count, graph_file) != endpoint_count) {
        std::cerr << "incomplete graph data\n";
        std::fclose(graph_file);
        return 5;
    }
    std::fclose(graph_file);

    MyVector<dynagp::Vertex *> vertices;
    vertices.reserve(n);
    for (unsigned int i = 0; i < n; ++i) {
        vertices.push_back(new dynagp::Vertex(static_cast<int>(i + 1)));
    }
    Graph graph(vertices);
    vertices.release_space();
    for (unsigned int i = 0; i < endpoint_count; i += 2) {
        graph.insertEdge(endpoints[i], endpoints[i + 1]);
    }

    std::ifstream query_file(argv[2]);
    double a = 0.0;
    double b = 1.0;
    int L = 0;
    double epsilon = 0.0;
    char query_type = 'S';
    if (!(query_file >> a >> b >> L >> epsilon >> query_type) || L < 0) {
        std::cerr << "invalid query header\n";
        return 6;
    }
    std::vector<double> weights(static_cast<size_t>(L + 1));
    for (double &weight : weights) {
        if (!(query_file >> weight)) {
            std::cerr << "invalid weight sequence\n";
            return 7;
        }
    }
    unsigned int seed_count = 0;
    if (!(query_file >> seed_count) || seed_count == 0) {
        std::cerr << "query requires at least one seed\n";
        return 8;
    }
    std::vector<double> x(n, 0.0);
    for (unsigned int i = 0; i < seed_count; ++i) {
        unsigned int vertex = 0;
        double mass = 0.0;
        if (!(query_file >> vertex >> mass) || vertex < 1 || vertex > n) {
            std::cerr << "invalid seed entry\n";
            return 9;
        }
        x[vertex - 1] += mass;
    }

    double *scores = graph.query(
        x.data(), static_cast<float>(a), static_cast<float>(b), L,
        weights.data(), epsilon, query_type);
    std::cout << std::setprecision(17);
    for (unsigned int i = 0; i < n; ++i) {
        std::cout << "SCORE\t" << (i + 1) << "\t" << scores[i] << "\n";
    }
    delete[] scores;
    return 0;
}

