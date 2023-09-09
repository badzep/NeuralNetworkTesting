#include <random>
#include <chrono>

const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine RNG (seed);
