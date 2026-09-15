#include "agp_api.h"
#include "graph/Graph.h"

#include <cmath>
#include <exception>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

struct PersistentGraph {
    Graph *graph;
    unsigned int vertex_count;
    float a;
    float b;
    std::vector<dynagp::Vertex *> vertices;

    PersistentGraph(
        unsigned int count,
        const int *endpoints,
        unsigned int endpoint_count,
        float a,
        float b,
        bool prepare_approximate)
        : graph(nullptr), vertex_count(count), a(a), b(b) {
        MyVector<dynagp::Vertex *> vertex_list;
        vertex_list.reserve(count);
        vertices.reserve(count);
        for (unsigned int index = 0; index < count; ++index) {
            auto *vertex = new dynagp::Vertex(static_cast<int>(index + 1));
            vertices.push_back(vertex);
            vertex_list.push_back(vertex);
        }
        graph = new Graph(vertex_list);
        for (unsigned int index = 0; index < endpoint_count; index += 2) {
            graph->insertEdge(endpoints[index], endpoints[index + 1]);
        }
        if (prepare_approximate) {
            graph->construct_DS_static(a, b);
        }
    }

    ~PersistentGraph() {
        delete graph;
        for (auto *vertex : vertices) {
            delete vertex;
        }
    }
};

void fail(const std::string &message) {
    last_error = message;
}

}  // namespace

extern "C" agp_handle agp_create(
    unsigned int vertex_count,
    const int *edge_endpoints,
    unsigned int endpoint_count,
    float a,
    float b,
    int prepare_approximate) {
    last_error.clear();
    if (vertex_count == 0) {
        fail("vertex_count must be positive");
        return nullptr;
    }
    if (endpoint_count % 2 != 0 || (endpoint_count > 0 && edge_endpoints == nullptr)) {
        fail("edge endpoints must contain complete pairs");
        return nullptr;
    }
    for (unsigned int index = 0; index < endpoint_count; ++index) {
        if (edge_endpoints[index] < 1 ||
            edge_endpoints[index] > static_cast<int>(vertex_count)) {
            fail("edge endpoint is outside the graph");
            return nullptr;
        }
    }
    try {
        return new PersistentGraph(
            vertex_count,
            edge_endpoints,
            endpoint_count,
            a,
            b,
            prepare_approximate != 0);
    } catch (const std::exception &error) {
        fail(error.what());
    } catch (...) {
        fail("unknown error while creating AGP graph");
    }
    return nullptr;
}

extern "C" int agp_query(
    agp_handle handle,
    const int *seed_ids,
    const double *seed_masses,
    unsigned int seed_count,
    int depth,
    const double *weights,
    double epsilon,
    char query_type,
    double *output_scores,
    unsigned int output_count) {
    last_error.clear();
    auto *state = static_cast<PersistentGraph *>(handle);
    if (state == nullptr) {
        fail("AGP graph handle is null");
        return 1;
    }
    if (seed_count == 0 || seed_ids == nullptr || seed_masses == nullptr) {
        fail("at least one seed is required");
        return 2;
    }
    if (depth < 0 || weights == nullptr) {
        fail("depth and weights are invalid");
        return 3;
    }
    if (query_type != 'N' && query_type != 'S') {
        fail("query_type must be N or S");
        return 4;
    }
    if (output_scores == nullptr || output_count != state->vertex_count) {
        fail("output buffer size must equal vertex count");
        return 5;
    }
    std::vector<double> seeds(state->vertex_count, 0.0);
    for (unsigned int index = 0; index < seed_count; ++index) {
        int vertex_id = seed_ids[index];
        if (vertex_id < 1 || vertex_id > static_cast<int>(state->vertex_count) ||
            !std::isfinite(seed_masses[index])) {
            fail("seed entry is invalid");
            return 6;
        }
        seeds[static_cast<unsigned int>(vertex_id - 1)] += seed_masses[index];
    }
    try {
        double *scores = state->graph->query(
            seeds.data(), state->a, state->b, depth,
            const_cast<double *>(weights), epsilon, query_type);
        for (unsigned int index = 0; index < output_count; ++index) {
            output_scores[index] = scores[index];
        }
        delete[] scores;
        return 0;
    } catch (const std::exception &error) {
        fail(error.what());
    } catch (...) {
        fail("unknown error while running AGP query");
    }
    return 7;
}

extern "C" unsigned int agp_vertex_count(agp_handle handle) {
    auto *state = static_cast<PersistentGraph *>(handle);
    return state == nullptr ? 0U : state->vertex_count;
}

extern "C" const char *agp_last_error(void) {
    return last_error.c_str();
}

extern "C" void agp_destroy(agp_handle handle) {
    delete static_cast<PersistentGraph *>(handle);
}
