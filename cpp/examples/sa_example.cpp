#include "simulated_annealing.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

int main() {
    // Minimize f(x) = (x - 3)^2
    auto objective = [](const double& x) {
        return (x - 3.0) * (x - 3.0);
    };

    // Neighbour: add a small random perturbation
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> perturb(0.0, 0.5);
    auto neighbor = [&rng, &perturb](const double& x) {
        return x + perturb(rng);
    };

    ai::CoolingSchedule cooling(ai::CoolingType::EXPONENTIAL, 0.995);
    ai::SimulatedAnnealing<double> sa(
        objective, neighbor,
        /*initial_temp=*/100.0,
        /*min_temp=*/1e-8,
        /*max_iterations=*/50000,
        cooling
    );

    double initial_solution = 0.0;
    std::cout << "Minimizing f(x) = (x - 3)^2 starting from x = "
              << initial_solution << "\n\n";

    auto [best, best_cost, history] = sa.optimize(initial_solution);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Result:\n";
    std::cout << "  Best x    = " << best      << "\n";
    std::cout << "  Best f(x) = " << best_cost << "\n";
    std::cout << "  Expected x ≈ 3.0\n\n";

    std::cout << "Annealing history (every 1000 iterations):\n";
    std::cout << std::setw(10) << "Iteration"
              << std::setw(14) << "Temperature"
              << std::setw(14) << "Current f(x)"
              << std::setw(14) << "Best f(x)" << "\n";
    for (const auto& step : history) {
        std::cout << std::setw(10) << step.iteration
                  << std::setw(14) << step.temperature
                  << std::setw(14) << step.current_cost
                  << std::setw(14) << step.best_cost << "\n";
    }

    return 0;
}
