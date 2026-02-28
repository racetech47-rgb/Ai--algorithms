#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <tuple>
#include <string>

namespace ai {

// Cooling schedule types
enum class CoolingType {
    LINEAR,
    EXPONENTIAL,
    LOGARITHMIC
};

// Cooling schedule
class CoolingSchedule {
public:
    explicit CoolingSchedule(CoolingType type = CoolingType::EXPONENTIAL,
                              double alpha = 0.995);
    double operator()(double initial_temp, int iteration) const;

private:
    CoolingType type_;
    double alpha_;
};

// History entry
struct AnnealingStep {
    int iteration;
    double temperature;
    double current_cost;
    double best_cost;
};

// Simulated Annealing class
template<typename Solution>
class SimulatedAnnealing {
public:
    using ObjectiveFn = std::function<double(const Solution&)>;
    using NeighborFn = std::function<Solution(const Solution&)>;

    SimulatedAnnealing(ObjectiveFn objective,
                       NeighborFn neighbor,
                       double initial_temp = 1000.0,
                       double min_temp = 1e-8,
                       int max_iterations = 100000,
                       CoolingSchedule cooling = CoolingSchedule{});

    std::tuple<Solution, double, std::vector<AnnealingStep>>
    optimize(const Solution& initial_solution);

    static double acceptance_probability(double current_cost, double new_cost, double temp);

private:
    ObjectiveFn objective_;
    NeighborFn neighbor_;
    double initial_temp_;
    double min_temp_;
    int max_iterations_;
    CoolingSchedule cooling_;
    std::mt19937 rng_;
};

// Template implementation
template<typename Solution>
SimulatedAnnealing<Solution>::SimulatedAnnealing(
    ObjectiveFn objective, NeighborFn neighbor,
    double initial_temp, double min_temp, int max_iterations,
    CoolingSchedule cooling)
    : objective_(std::move(objective))
    , neighbor_(std::move(neighbor))
    , initial_temp_(initial_temp)
    , min_temp_(min_temp)
    , max_iterations_(max_iterations)
    , cooling_(cooling)
    , rng_(std::random_device{}())
{}

template<typename Solution>
double SimulatedAnnealing<Solution>::acceptance_probability(
    double current_cost, double new_cost, double temp) {
    if (new_cost < current_cost) return 1.0;
    return std::exp(-(new_cost - current_cost) / temp);
}

template<typename Solution>
std::tuple<Solution, double, std::vector<AnnealingStep>>
SimulatedAnnealing<Solution>::optimize(const Solution& initial_solution) {
    Solution current = initial_solution;
    Solution best = initial_solution;
    double current_cost = objective_(current);
    double best_cost = current_cost;

    std::vector<AnnealingStep> history;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int iter = 0; iter < max_iterations_; ++iter) {
        double temp = cooling_(initial_temp_, iter);
        if (temp < min_temp_) break;

        Solution neighbor = neighbor_(current);
        double neighbor_cost = objective_(neighbor);

        if (dist(rng_) < acceptance_probability(current_cost, neighbor_cost, temp)) {
            current = std::move(neighbor);
            current_cost = neighbor_cost;
        }

        if (current_cost < best_cost) {
            best = current;
            best_cost = current_cost;
        }

        if (iter % 1000 == 0) {
            history.push_back({iter, temp, current_cost, best_cost});
        }
    }

    return {best, best_cost, history};
}

} // namespace ai
