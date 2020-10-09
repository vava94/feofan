#include "utils.hpp"

using namespace utils;

__device__ __host__ unsigned int utils::divUp(unsigned int a, unsigned int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}