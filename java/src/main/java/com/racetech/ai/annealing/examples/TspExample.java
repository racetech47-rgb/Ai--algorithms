package com.racetech.ai.annealing.examples;

import com.racetech.ai.annealing.AnnealingResult;
import com.racetech.ai.annealing.ExponentialCooling;
import com.racetech.ai.annealing.SimulatedAnnealing;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Demonstrates simulated annealing on a 5-city Travelling Salesman Problem.
 *
 * <p>City coordinates are hard-coded.  A random tour is used as the initial
 * solution and 2-opt swaps are used to generate neighbours.
 */
public class TspExample {

    /** Hard-coded (x, y) coordinates for 5 cities. */
    private static final double[][] CITIES = {
        {0.0,  0.0},   // city 0
        {3.0,  4.0},   // city 1
        {6.0,  1.0},   // city 2
        {5.0,  7.0},   // city 3
        {2.0,  8.0},   // city 4
    };

    /** Computes the Euclidean distance between two cities. */
    private static double dist(int a, int b) {
        double dx = CITIES[a][0] - CITIES[b][0];
        double dy = CITIES[a][1] - CITIES[b][1];
        return Math.sqrt(dx * dx + dy * dy);
    }

    /** Computes the total tour length for a given permutation of city indices. */
    private static double tourLength(List<Integer> tour) {
        double length = 0.0;
        int n = tour.size();
        for (int i = 0; i < n; i++) {
            length += dist(tour.get(i), tour.get((i + 1) % n));
        }
        return length;
    }

    /**
     * Generates a 2-opt neighbour by reversing a random sub-segment of the tour.
     *
     * @param tour current tour
     * @return new tour with a reversed sub-segment
     */
    private static List<Integer> twoOptNeighbor(List<Integer> tour) {
        Random rng = new Random();
        int n  = tour.size();
        int i  = rng.nextInt(n);
        int j  = rng.nextInt(n);
        if (i == j) j = (j + 1) % n;
        if (i > j)  { int tmp = i; i = j; j = tmp; }

        List<Integer> newTour = new ArrayList<>(tour);
        // Reverse segment [i, j].
        while (i < j) {
            Collections.swap(newTour, i, j);
            i++;
            j--;
        }
        return newTour;
    }

    /** Entry point. */
    public static void main(String[] args) {
        System.out.println("=== TSP Example (5 cities) ===");
        System.out.println("City coordinates:");
        for (int i = 0; i < CITIES.length; i++) {
            System.out.printf("  City %d: (%.1f, %.1f)%n", i, CITIES[i][0], CITIES[i][1]);
        }

        // Initial tour: [0, 1, 2, 3, 4]
        List<Integer> initialTour = new ArrayList<>();
        for (int i = 0; i < CITIES.length; i++) initialTour.add(i);

        System.out.printf("%nInitial tour  : %s%n", initialTour);
        System.out.printf("Initial length: %.4f%n", tourLength(initialTour));

        SimulatedAnnealing<List<Integer>> sa = new SimulatedAnnealing<>(
            TspExample::tourLength,
            TspExample::twoOptNeighbor,
            /* initialTemp    */ 100.0,
            /* minTemp        */ 0.001,
            /* maxIterations  */ 10_000,
            new ExponentialCooling(0.9995)
        );

        AnnealingResult<List<Integer>> result = sa.optimize(initialTour);

        System.out.printf("%nBest tour     : %s%n", result.getBestSolution());
        System.out.printf("Best length   : %.4f%n", result.getBestCost());
        System.out.printf("Iterations    : %d%n", result.getCostHistory().size());
    }
}
