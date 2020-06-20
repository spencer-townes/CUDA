/**
 *
 * Include file for MT19937, with initialization improved 2002/1/26.
 * Coded by Takuji Nishimura and Makoto Matsumoto.
 *
 * Adapted to run with OpenCL from Jasper Bedaux's 2003/1/1 (see
 * http://www.bedaux.net/mtrand/).  The generators returning floating
 * point numbers are based on a version by Isaku Wada, 2002/01/09
 *
 * Adapted again to run with CUDA by Spencer Townes from Miami 
 * University under the guidance of Professor DJ Rao.
 */

#ifdef HOST

 // If the "HOST" compiler flag is set then this code is being compiled
 // on the host and not on the GPU device.
#include <string.h>
#include <stdio.h>
// Remove CUDA directives to make C compatible with host
#define CUDA_DEVICE 

#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define CUDA_DEVICE __device__

#endif  // HOST

/** Some of the predefined compile time constants used for random
    number generation.  Since this is used to create an array further
    below, it cannot be a 'constant int' but has to be #define.
*/
#define MaxStates 624

/* An approximate middle-ish point used for random number
   generation */
#define m 397

   /**
    * Simplified C struct to hold the key information used for generating
    * Mersenne Twister (MT) random numbers.  This structure is meant to
    * encapsulate the necessary information so that multiple/independent
    * versions of the random number generator can be used a thread-safe
    * manner.
    */
struct MTrand_Info {
    // The states used for random number generation with warp-around
    // after MaxState entries are used.
    unsigned long state[MaxStates];
    // The position in the state array
    int p;
};

/**
   The top-level random number generation initialization function.

   \param[out] rndGen The members of this structure are initialized to
   default initial values based on the supplied seed.

   \param[in] seed The seed to be used to initialize the random number
   generator.  The default value is 5489UL.
*/
CUDA_DEVICE void inline MTrand_init(struct MTrand_Info* rndGen, unsigned long seed) {
    // First, initialized the structure members to standard defaults.
    unsigned long* const state = rndGen->state;
#ifdef HOST
    //bzero(state, sizeof(unsigned long) * MaxStates);
#endif
    // Initialize the states
    state[0] = seed & 0xFFFFFFFFUL; // for > 32 bit machines
    for (int i = 1; i < MaxStates; ++i) {
        state[i] = 1812433253UL * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
        // see Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier in the
        // previous versions, MSBs of the seed affect only MSBs of the
        // array state 2002/01/09 modified by Makoto Matsumoto
        state[i] &= 0xFFFFFFFFUL;  // for > 32 bit machines
    }
    // force gen_state() to be called for next random number
    rndGen->p = MaxStates;
}

CUDA_DEVICE unsigned long inline MTrand_twiddle(unsigned long u, unsigned long v) {
    return (((u & 0x80000000UL) | (v & 0x7FFFFFFFUL)) >> 1)
        ^ ((v & 1UL) ? 0x9908B0DFUL : 0x0UL);
}

// generate new state vector
CUDA_DEVICE void inline MTrand_gen_state(struct MTrand_Info* rndGen) {
    // A shortcut reference to states for this random number generator.
    unsigned long* const state = rndGen->state;
    const int range = MaxStates - m;
    for (int i = 0; (i < range); ++i) {
        state[i] = state[i + m] ^ MTrand_twiddle(state[i], state[i + 1]);
    }
    for (int i = range; i < (MaxStates - 1); ++i) {
        state[i] = state[i + m - MaxStates] ^
            MTrand_twiddle(state[i], state[i + 1]);
    }
    state[MaxStates - 1] = state[m - 1] ^
        MTrand_twiddle(state[MaxStates - 1], state[0]);
    rndGen->p = 0; // reset position
}

// generate 32 bit random integer
CUDA_DEVICE unsigned long inline MTrand_int32(struct MTrand_Info* rndGen) {
    if (rndGen->p == MaxStates) {
        MTrand_gen_state(rndGen); // new state vector needed
    }
    // gen_state() is split off to be non-inline, because it is only
    // called once in every 624 calls and otherwise irand() would
    // become too big to get inlined
    unsigned long x = rndGen->state[rndGen->p++];
    x ^= (x >> 11);
    x ^= (x << 7) & 0x9D2C5680UL;
    x ^= (x << 15) & 0xEFC60000UL;
    return x ^ (x >> 18);
}

/** Convenience method to map a 32-bit Mersenne Twister number to the
    range 0.0---1.0 (inclusive).

    \return A uniformly distributed random number in the range
    0.0--1.0 (inclusive).
*/
CUDA_DEVICE double inline MTrand_get(struct MTrand_Info* rndGen) {
    return (1.0 / 4294967296.0) * MTrand_int32(rndGen);  // divided by 2^32
}

//int main() {
//    struct MTrand_Info rndGen;
//    MTrand_init(&rndGen, 5489);
//    for (int i = 0; (i < 100); i++) {
//        printf("%f\n", MTrand_get(&rndGen));
//    }
//    return 0;
//}


