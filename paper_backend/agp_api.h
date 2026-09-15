#ifndef AGP_RESEARCH_API_H
#define AGP_RESEARCH_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void *agp_handle;

agp_handle agp_create(
    unsigned int vertex_count,
    const int *edge_endpoints,
    unsigned int endpoint_count,
    float a,
    float b,
    int prepare_approximate);

int agp_query(
    agp_handle handle,
    const int *seed_ids,
    const double *seed_masses,
    unsigned int seed_count,
    int depth,
    const double *weights,
    double epsilon,
    char query_type,
    double *output_scores,
    unsigned int output_count);

unsigned int agp_vertex_count(agp_handle handle);
const char *agp_last_error(void);
void agp_destroy(agp_handle handle);

#ifdef __cplusplus
}
#endif

#endif
