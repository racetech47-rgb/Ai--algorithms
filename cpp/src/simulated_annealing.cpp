#include "simulated_annealing.hpp"
#include <cmath>
#include <stdexcept>

namespace ai {

CoolingSchedule::CoolingSchedule(CoolingType type, double alpha)
    : type_(type), alpha_(alpha) {}

double CoolingSchedule::operator()(double initial_temp, int iteration) const {
    switch (type_) {
        case CoolingType::LINEAR:
            return initial_temp * (1.0 - alpha_ * iteration);
        case CoolingType::EXPONENTIAL:
            return initial_temp * std::pow(alpha_, iteration);
        case CoolingType::LOGARITHMIC:
            return initial_temp / (1.0 + std::log(1.0 + iteration));
        default:
            throw std::invalid_argument("Unknown cooling type");
    }
}

} // namespace ai
