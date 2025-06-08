#include "pch.h"
#include "MainWindow.xaml.h"
#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

template<typename T>
const T& clamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : (value > high) ? high : value;
}

double X4_MAX;
bool flag = true;
double D, Cw, Cs, Ch;
double k1, k2, k3, k4;

double ValueMax,ValueMin;

mt19937_64 global_generator(random_device{}());

double rand_uniform(double min, double max) {
    uniform_real_distribution<double> dist(min, max);
    return dist(global_generator);
}

double rand_normal(double mean = 0.0, double stddev = 1.0) {
    normal_distribution<double> dist(mean, stddev);
    return dist(global_generator);
}

void calculateCoefficients(double& k1, double& k2, double& k3, double& k4) {
    const double pi = M_PI;
    const double cos30 = cos(30.0 * pi / 180.0); 

    double Vl = 2.0 * pi * pow(1.0 / cos30, 2) * (60.0 / 360.0);
    k1 = Vl * D * Cw; 

    double Vs = 4.0 * pi * pi * pow(1.0 / cos30, 2) * (60.0 / 360.0);
    k2 = Vs * D * Cw;

    k3 = 2 * pi * D * Cs;  

    k4 = 4 * pi * D * Ch;  
}

double f(double x1, double x2, double x3, double x4) {
	if(flag) {
        return 3.1611 * pow(x1, 2) * x4 + 19.84 * pow(x1, 2) * x3 + 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * pow(x3, 2) ;
	}
    else {
        return k1 * pow(x1, 2) * x4 + k2 * pow(x1, 2) * x3 + k3 * x1 * x3 * x4 + k4 * x2 * pow(x3, 2);
    }
}

bool checkConstraints(double x1, double x2, double x3, double x4) {
  
    if (flag) {
        return (x1 >= 0.0193 * x3) &&
            (x2 >= 0.00954 * x3) &&
            ((4.0 / 3.0) * M_PI * pow(x3, 3) + M_PI * pow(x3, 2) * x4 >= 1296000) &&
            (x4 - X4_MAX <= 0);
    }
    else {
        int a = ValueMin * 1728;
        return (x1 >= 0.0193 * x3) &&
            (x2 >= 0.00954 * x3) &&
            ((4.0 / 3.0) * M_PI * pow(x3, 3) + M_PI * pow(x3, 2) * x4 >= a) &&
            (x4 - X4_MAX <= 0);
    }
}

vector<double> randomSolution() {
    vector<double> solution(4);
    do {
        solution[0] = rand_uniform(0.0625, 6.1875);
        solution[1] = rand_uniform(0.0625, 6.1875);
        solution[2] = rand_uniform(10.0, 200.0);
        solution[3] = rand_uniform(10.0, X4_MAX); 
    } while (!checkConstraints(solution[0], solution[1], solution[2], solution[3]));
    return solution;
}

class CuckooSearch {
public:
    CuckooSearch(int numNests, int maxGenerations, int maxIter, double pa = 0.25)
        : numNests(numNests), maxGenerations(maxGenerations), maxIter(maxIter),
        pa(pa), bestFitness(numeric_limits<double>::infinity()) {
    }

    void search() {
        auto start = chrono::high_resolution_clock::now(); 

        bestSolution = randomSolution();
        bestFitness = evaluateFitness(bestSolution);

        double prevBest = bestFitness;
        int noImprove = 0;
        const double epsilon = 1e-6; 


        for (int iter = 0; iter < maxIter; ++iter) {
            generateNewSolutions();
            replaceAbandonedNests();


            if (abs(prevBest - bestFitness) < epsilon) {
                break; 
            }
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start; 

        printResults(elapsed);
    }

private:
    int numNests;
    int maxGenerations;
    int maxIter;
    double pa;
    double bestFitness;
    vector<double> bestSolution;

    void generateNewSolutions() {
        for (int i = 0; i < numNests; ++i) {
            vector<double> newSolution = levyFlight(bestSolution);
            if (checkAndRepair(newSolution)) {
                double newFit = evaluateFitness(newSolution);
                if (newFit < bestFitness) {
                    bestFitness = newFit;
                    bestSolution = newSolution;
                }
            }
        }
    }

    void replaceAbandonedNests() {
        for (int i = 0; i < numNests; ++i) {
            if (rand_uniform(0, 1) < pa) {
                vector<double> newSolution = randomSolution();
                double newFit = evaluateFitness(newSolution);
                if (newFit < bestFitness) {
                    bestFitness = newFit;
                    bestSolution = newSolution;
                }
            }
        }
    }

    vector<double> levyFlight(const vector<double>& current) {
        vector<double> newSolution = current;
        double beta = 1.5;
        double sigma = pow(

            (tgamma(1 + beta) * sin(M_PI * beta / 2) /

                (tgamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2))),

            1 / beta

        );

        for (size_t i = 0; i < 4; ++i) {
            double u = rand_normal(0, 1) * sigma;
            double v = rand_normal(0, 1);
            double step = 0.01 * u / pow(abs(v), 1 / beta);
            newSolution[i] += step;
        }
        return newSolution;
    }

    bool checkAndRepair(vector<double>& solution) {
        const vector<pair<double, double>> bounds = {
            {0.0625, 6.1875}, {0.0625, 6.1875}, {10.0, 200.0}, {10.0, X4_MAX}  };// отут менять (вторую скобку)

        for (size_t i = 0; i < 4; ++i) {
            solution[i] = ::clamp(solution[i], bounds[i].first, bounds[i].second);
        }
        return checkConstraints(solution[0], solution[1], solution[2], solution[3]);
    }

    double evaluateFitness(const vector<double>& solution) {
        return f(solution[0], solution[1], solution[2], solution[3]);
    }

    void printResults(const chrono::duration<double>& elapsed) {
    
        cout << "Execution Time: " << elapsed.count() << " seconds\n";
        cout << "Best Fitness: " << bestFitness << "\n";
        cout << "Best Solution: (" << bestSolution[0] << ", " << bestSolution[1] << ", "
            << bestSolution[2] << ", " << bestSolution[3] << ")\n";
    }
};

class PSO {
public:
    PSO(size_t maxIter, size_t swarmSize, double cognitiveAttraction, double socialAttraction)
        : maxIter(maxIter), swarmSize(swarmSize),
        c1(cognitiveAttraction), c2(socialAttraction) {
    }

    void search() {
        const size_t dim = 4;

        struct Particle {
            vector<double> position;
            vector<double> velocity;
            vector<double> personalBest;
            double cost;
            double bestCost;
        };

        vector<Particle> swarm(swarmSize);
        vector<double> globalBestPosition(dim);
        double globalBestCost = numeric_limits<double>::infinity();

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dist(0.0, 1.0);

 
        for (auto& p : swarm) {
            p.position = randomSolution();

            // 10 % від діапазону для кожної координати
            const double vMax[4] = { 0.61, 0.61, 19.0, 23.0 };

            p.velocity.resize(dim);
            for (size_t d = 0; d < dim; ++d) {
                p.velocity[d] = rand_uniform(-vMax[d], vMax[d]);   // випадкова стартова швидкість
            }

            p.cost = f(p.position[0], p.position[1], p.position[2], p.position[3]);
            p.personalBest = p.position;
            p.bestCost = p.cost;

            if (p.cost < globalBestCost) {
                globalBestCost = p.cost;
                globalBestPosition = p.position;
            }
        }


        double inertia = 0.7;
        double inertiaDrop = 0.99;

        auto start = chrono::high_resolution_clock::now();

        for (size_t iter = 0; iter < maxIter; ++iter) {
            for (auto& p : swarm) {
                for (size_t d = 0; d < dim; ++d) {
                    double r1 = dist(gen);
                    double r2 = dist(gen);

                    p.velocity[d] = inertia * p.velocity[d]
                        + c1 * r1 * (p.personalBest[d] - p.position[d])
                            + c2 * r2 * (globalBestPosition[d] - p.position[d]);

                        p.position[d] += p.velocity[d];

    
                        if (d == 0 || d == 1) p.position[d] = ::clamp(p.position[d], 0.0625, 6.1875);
                        else if (d == 2) p.position[d] = ::clamp(p.position[d], 10.0, 200.0);
                        else if (d == 3) p.position[d] = ::clamp(p.position[d], 10.0, X4_MAX);
                }

                if (checkConstraints(p.position[0], p.position[1], p.position[2], p.position[3])) {
                    p.cost = f(p.position[0], p.position[1], p.position[2], p.position[3]);

                    if (p.cost < p.bestCost) {
                        p.personalBest = p.position;
                        p.bestCost = p.cost;

                        if (p.bestCost < globalBestCost) {
                            globalBestCost = p.bestCost;
                            globalBestPosition = p.personalBest;
                        }
                    }
                }
            }

            inertia *= inertiaDrop;
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        cout << fixed << setprecision(8);

        cout << "Execution Time: " << elapsed.count() << " seconds\n";
        cout << "Best Fitness: " << globalBestCost << "\n";
        cout << "Best Solution: (" << globalBestPosition[0] << ", " << globalBestPosition[1]
            << ", " << globalBestPosition[2] << ", " << globalBestPosition[3] << ")\n";
    }

private:
    size_t maxIter;
    size_t swarmSize;
    double c1, c2;
};

class SimulatedAnnealing {
public:
    SimulatedAnnealing(double initialTemp, double coolingRate, int maxIter)
        : temperature(initialTemp), coolingRate(coolingRate), maxIter(maxIter), bestFitness(numeric_limits<double>::infinity()) {
    }

    void search() {

        auto start = chrono::high_resolution_clock::now();
        vector<double> currentSolution = randomSolution();
        double currentFitness = f(currentSolution[0], currentSolution[1], currentSolution[2], currentSolution[3]);
        bestSolution = currentSolution;
        bestFitness = currentFitness;

        for (int i = 0; i < maxIter; ++i) {
            vector<double> newSolution = neighborSolution(currentSolution);
            double newFitness = f(newSolution[0], newSolution[1], newSolution[2], newSolution[3]);

            if (newFitness < currentFitness || acceptWorseSolution(currentFitness, newFitness)) {
                currentSolution = newSolution;
                currentFitness = newFitness;
            }

            if (currentFitness < bestFitness) {
                bestSolution = currentSolution;
                bestFitness = currentFitness;
            }

            temperature *= coolingRate;
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Execution Time: " << elapsed.count() << " seconds\n";
        cout << "Best Fitness: " << bestFitness << "\n";
        cout << "Best Solution: (" << bestSolution[0] << ", " << bestSolution[1] << ", " << bestSolution[2] << ", " << bestSolution[3] << ")\n";
    }

private:
    double temperature;
    double coolingRate;
    int maxIter;
    double bestFitness;
    vector<double> bestSolution;

    vector<double> neighborSolution(const vector<double>& current) {
        vector<double> newSolution(4);
        for (int i = 0; i < 4; ++i) {
            double step = ((rand() % 2 == 0 ? 1 : -1) * (0.001 + static_cast<double>(rand()) / RAND_MAX * 0.1));
            newSolution[i] = current[i] + step;
        }
        return checkConstraints(newSolution[0], newSolution[1], newSolution[2], newSolution[3]) ? newSolution : current;
    }

    bool acceptWorseSolution(double currentFitness, double newFitness) {
        double probability = exp((currentFitness - newFitness) / temperature);
        return (static_cast<double>(rand()) / RAND_MAX) < probability;
    }
};

class HookeJeeves {
public:
    HookeJeeves(vector<double> start,
        vector<double> initDelta,
        double tol,
        double lambda = 2.0,
        int maxIter = 10000)
        : delta(initDelta), tol(tol), lambda(lambda), maxIter(maxIter) {
        x = start;
    }

    void search() {
        const int n = static_cast<int>(x.size());
        int iter = 0;
        vector<double> xBase = x;
        double fBase = eval(xBase);
        auto t0 = chrono::high_resolution_clock::now();

        while (!converged() && iter < maxIter) {
            // 1) Exploratory step
            vector<double> xExp = exploratory(xBase);
            double fExp = eval(xExp);

            if (fExp < fBase) {
                // 2) Pattern move
                vector<double> xPattern(n);
                for (int i = 0; i < n; ++i)
                    xPattern[i] = xExp[i] + lambda * (xExp[i] - xBase[i]);

                if (!checkConstraints(xPattern[0], xPattern[1], xPattern[2], xPattern[3]))
                    xPattern = xExp; // відкотити, якщо вийшли за межі

                xBase = exploratory(xPattern);
                fBase = eval(xBase);
            }
            else {
                // зменшуємо кроки
                for (double& d : delta) d *= 0.5;
            }
            ++iter;
        }

        auto t1 = chrono::high_resolution_clock::now();
        //cout << "Iterations: " << iter << "\n"

        cout << "Execution Time: " << chrono::duration<double>(t1 - t0).count() << " seconds\n";
        cout << "Best Fitness: " << fBase << "\n";
        cout << "Best Solution: (" << xBase[0] << ", " << xBase[1] << ", " << xBase[2] << ", " << xBase[3] << ")\n";
    }

private:
    vector<double> x;           
    vector<double> delta;       
    double tol;                 
    double lambda;              
    int maxIter;

    double eval(const vector<double>& v) { return f(v[0], v[1], v[2], v[3]); }

    vector<double> exploratory(const vector<double>& base) {
        vector<double> trial = base;
        for (size_t i = 0; i < base.size(); ++i) {
            trial[i] = base[i] + delta[i];
            if (!checkConstraints(trial[0], trial[1], trial[2], trial[3]) || eval(trial) >= eval(base)) {
                trial[i] = base[i] - delta[i];
                if (!checkConstraints(trial[0], trial[1], trial[2], trial[3]) || eval(trial) >= eval(base)) {
                    trial[i] = base[i]; 
                }
            }
        }
        return trial;
    }

    bool converged() const {
        return all_of(delta.begin(), delta.end(), [this](double d) { return d < tol; });
    }
};

class FireflyAlgorithm {
private:
    static const int DIM = 4;
    const int POP_SIZE;
    const int MAX_GEN;
    const double alpha;
    const double beta0;
    const double gamma;

    struct Firefly {
        vector<double> x;
        double fitness;
    };

    vector<Firefly> population;

    bool is_feasible(const vector<double>& x) {
        return checkConstraints(x[0], x[1], x[2], x[3]);
    }

    double distance(const vector<double>& x1, const vector<double>& x2) {
        double sum = 0.0;
        for (int i = 0; i < DIM; ++i)
            sum += pow(x1[i] - x2[i], 2);
        return sqrt(sum);
    }

    void move(Firefly& a, const Firefly& b) {
        double r = distance(a.x, b.x);
        vector<double> new_x(DIM);

        for (int i = 0; i < DIM; ++i) {
            double beta = beta0 * exp(-gamma * r * r);
            double rand_factor = alpha * rand_normal(0.0, 1.0);
            new_x[i] = a.x[i] + beta * (b.x[i] - a.x[i]) + rand_factor;
        }

        if (is_feasible(new_x)) {
            a.x = new_x;
            a.fitness = f(a.x[0], a.x[1], a.x[2], a.x[3]);
        }
        else {
            a.x = randomSolution();
            a.fitness = f(a.x[0], a.x[1], a.x[2], a.x[3]);
        }
    }

    void initializePopulation() {
        population.clear();
        while (population.size() < POP_SIZE) {
            vector<double> sol = randomSolution();
            population.push_back({ sol, f(sol[0], sol[1], sol[2], sol[3]) });
        }
    }

public:
    FireflyAlgorithm(int popSize, int maxGen, double a, double b0, double g)
        : POP_SIZE(popSize), MAX_GEN(maxGen), alpha(a), beta0(b0), gamma(g) {
    }

    void run() {
        auto start = chrono::high_resolution_clock::now();

        initializePopulation();

        for (int gen = 0; gen < MAX_GEN; ++gen) {
            for (int i = 0; i < POP_SIZE; ++i) {
                for (int j = 0; j < POP_SIZE; ++j) {
                    if (population[j].fitness < population[i].fitness)
                        move(population[i], population[j]);
                }
            }
        }

        Firefly best = population[0];
        for (const auto& ffly : population) {
            if (ffly.fitness < best.fitness)
                best = ffly;
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        cout << "Execution Time: " << elapsed.count() << " seconds\n";
        cout << "Best Fitness: " << best.fitness << "\n";
        cout << "Best Solution: (" << best.x[0] << ", " << best.x[1] << ", " << best.x[2] << ", " << best.x[3] << ")\n";
    }
};

class FlowerPollinationAlgorithm {
public:
    FlowerPollinationAlgorithm(int populationSize, int maxIterations, double p = 0.8)
        : popSize(populationSize), maxIter(maxIterations), switchProbability(p) {
        dim = 4;
        initPopulation();
    }

    void search() {
        auto start = chrono::high_resolution_clock::now();

        for (int iter = 0; iter < maxIter; ++iter) {
            for (int i = 0; i < popSize; ++i) {
                vector<double> newSol(dim);
                if (rand01(rng) < switchProbability) {
                    double L = levy();
                    for (int d = 0; d < dim; ++d) {
                        newSol[d] = population[i][d] + L * (bestSolution[d] - population[i][d]);
                    }
                }
                else {
                    int j = randInt(rng) % popSize;
                    int k = randInt(rng) % popSize;
                    for (int d = 0; d < dim; ++d) {
                        newSol[d] = population[i][d] + rand01(rng) * (population[j][d] - population[k][d]);
                    }
                }

                if (!checkConstraints(newSol[0], newSol[1], newSol[2], newSol[3])) {
                    newSol = randomSolution();
                }

                double newFit = fitness(newSol);
                if (newFit < fitness(population[i])) {
                    population[i] = newSol;
                }
                if (newFit < bestFitness) {
                    bestSolution = newSol;
                    bestFitness = newFit;
                }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        cout << "Execution Time: " << elapsed.count() << " seconds\n";
        cout << "Best Fitness: " << bestFitness << "\n";
        cout << "Best Solution: (" << bestSolution[0] << ", " << bestSolution[1]
            << ", " << bestSolution[2] << ", " << bestSolution[3] << ")\n";
    }

private:
    int popSize;
    int maxIter;
    int dim;
    double switchProbability;
    vector<vector<double>> population;
    vector<double> bestSolution;
    double bestFitness = numeric_limits<double>::infinity();

    default_random_engine rng{ random_device{}() };
    uniform_real_distribution<double> rand01{ 0.0, 1.0 };
    uniform_int_distribution<int> randInt{ 0, 1000000 };

    void initPopulation() {
        population.resize(popSize);
        for (auto& ind : population) {
            ind = randomSolution();
            double fit = fitness(ind);
            if (fit < bestFitness) {
                bestFitness = fit;
                bestSolution = ind;
            }
        }
    }

    double fitness(const vector<double>& x) {
        return checkConstraints(x[0], x[1], x[2], x[3]) ? f(x[0], x[1], x[2], x[3]) : numeric_limits<double>::infinity();
    }

    double levy() {
        double epsilon = rand_uniform(1e-10, 1.0);
        double L = (1.0 / sqrt(2.0 * M_PI)) *
            pow(epsilon, -1.7) *
            exp(-1.0 / (2.0 * epsilon));

        return L;
    }

};

class GravitationalSearchAlgorithm {
public:
    GravitationalSearchAlgorithm(int populationSize, int maxIter, double G0, double alpha = 20.0)
        : N(populationSize), MaxIter(maxIter), G0(G0), alpha(alpha), dim(4) {
        X.resize(N, vector<double>(dim));
        V.assign(N, vector<double>(dim, 0.0));
        fit.resize(N); mass.resize(N);
        bestFit = numeric_limits<double>::infinity();
    }

    void search() {
        initAgents();
        auto t0 = chrono::high_resolution_clock::now();

        for (int iter = 0; iter < MaxIter; ++iter) {
            evaluateFitness();
            double Gt = G0 * exp(-alpha * double(iter) / MaxIter); 
            applyForcesAndMove(Gt, iter);
        }

        auto t1 = chrono::high_resolution_clock::now();

        cout << "Execution Time: " << chrono::duration<double>(t1 - t0).count() << " seconds\n";
        cout << "Best Fitness: " << bestFit << "\n";
        cout << "Best Solution: (" << bestSol[0] << ", " << bestSol[1] << ", " << bestSol[2] << ", " << bestSol[3] << ")\n";
    }

private:

    int N, MaxIter, dim; double G0, alpha;
    vector<vector<double>> X, V;        
    vector<double> fit, mass;           
    vector<double> bestSol; double bestFit;
   


    void initAgents() { for (auto& x : X) x = randomSolution(); }

    void evaluateFitness() {
        double worst = -numeric_limits<double>::infinity();
        double best = numeric_limits<double>::infinity();
        for (int i = 0; i < N; ++i) {
            fit[i] = f(X[i][0], X[i][1], X[i][2], X[i][3]);
            worst = max(worst, fit[i]); best = min(best, fit[i]);
            if (fit[i] < bestFit && checkConstraints(X[i][0], X[i][1], X[i][2], X[i][3])) {
                bestFit = fit[i]; bestSol = X[i];
            }
        }
        double sum = 0, eps = 1e-16;
        for (int i = 0; i < N; ++i) {
            mass[i] = (worst - fit[i]) / (worst - best + eps);
            sum += mass[i];
        }
        for (auto& m : mass) m /= (sum + eps); 
    }

    void applyForcesAndMove(double Gt, int iter) {
        const double eps = 1e-12;
        int K = int(round(N - double(iter) * (N - 1) / MaxIter)); 
        vector<int> idx(N); iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b) {return fit[a] < fit[b]; });
        vector<int> kbest(idx.begin(), idx.begin() + K);

        vector<vector<double>> acc(N, vector<double>(dim, 0.0));
        for (int i = 0; i < N; ++i) {
            for (int j : kbest) if (j != i) {
                double R = euclidian(X[i], X[j]) + eps;
                double common = rand_uniform(0.0, 1.0) * Gt * (mass[i] * mass[j]) / R;
                for (int d = 0; d < dim; ++d) acc[i][d] += common * (X[j][d] - X[i][d]);
            }
            for (int d = 0; d < dim; ++d) acc[i][d] /= (mass[i] + eps); 
        }

        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < dim; ++d) {
                V[i][d] = rand_uniform(0.0, 1.0) * V[i][d] + acc[i][d];
                X[i][d] += V[i][d];
            }
        }
    }

    static double euclidian(const vector<double>& a, const vector<double>& b) {
        double s = 0; for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
        return sqrt(s);
    }
};
