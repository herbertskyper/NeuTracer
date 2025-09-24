#include <curand.h>
#include <curand_kernel.h>

extern "C" __global__ void lesinn(const double *incoming_data, const double *all_data, int incoming_window, int all_data_window, int dimension, int t, int phi, double *similarity)
{
    // Thread ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curandState_t state;
    /* we have to initialize the state */
    curand_init(0,   /* the seed controls the sequence of random values that are produced */
                tid, /* the sequence number is only important with multiple cores */
                0,   /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);

    const double *current_data = incoming_data + dimension * tid;

    double score = 0;
    for (size_t sample_id = 0; sample_id < t; sample_id++)
    {
        double max_sim = 0;
        for (size_t s = 0; s < phi; s++)
        {
            int sample = curand(&state) % all_data_window;
            double tmp = 0;
            // Calculate similarity
            for (size_t data_id = 0; data_id < dimension; data_id++)
            {
                tmp += (current_data[data_id] - all_data[sample * dimension + data_id]) * (current_data[data_id] - all_data[sample * dimension + data_id]);
            }
            tmp = sqrt(tmp);
            tmp = 1 / (1 + tmp);
            max_sim = max(max_sim, tmp);
        }
        score += max_sim;
    }

    // Store result
    if (score > 0)
    {
        similarity[tid] = t / score;
    }
}