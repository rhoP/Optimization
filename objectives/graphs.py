import networkx as nx
import cvxpy as cp
import numpy as np


class GraphOptimizationProblem:
    def __init__(self, graph, **kwargs):
        self.graph = graph
        self.params = kwargs

    def setup_problem(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def solve_problem(self, solver):
        self.problem = self.setup_problem()
        return solver.solve(self.problem)


class GraphCutMinimization(GraphOptimizationProblem):
    def setup_problem(self):
        W = nx.to_numpy_array(self.graph, weight='weight')
        n = self.graph.number_of_nodes()
        x = cp.Variable(n, boolean=True)
        cut_value = cp.sum([W[i, j] * cp.abs(x[i] - x[j]) for i in range(n) for j in range(n) if W[i, j] > 0])
        return cp.Problem(cp.Minimize(cut_value))


class GraphDenoising(GraphOptimizationProblem):
    def setup_problem(self):
        W = nx.to_numpy_array(self.graph)
        y = self.params['y']
        lambda_param = self.params['lambda_param']
        f = cp.Variable(len(y))
        tv = cp.sum([cp.abs(f[i] - f[j]) for i in range(len(y)) for j in range(len(y)) if W[i, j] > 0])
        fidelity = cp.sum_squares(f - y)
        return cp.Problem(cp.Minimize(tv + lambda_param * fidelity))


class ConstrainedTotalVariationMinimization(GraphOptimizationProblem):
    def setup_problem(self):
        W = nx.to_numpy_array(self.graph)
        a = self.params['a']
        b = self.params['b']
        f = cp.Variable(len(a))
        tv = cp.sum([cp.abs(f[i] - f[j]) for i in range(len(a)) for j in range(len(a)) if W[i, j] > 0])
        constraints = [f >= a, f <= b]
        return cp.Problem(cp.Minimize(tv), constraints)


class ImageSegmentation(GraphOptimizationProblem):
    def setup_problem(self):
        W = nx.to_numpy_array(self.graph)
        y = self.params['y']
        lambda_param = self.params['lambda_param']
        f = cp.Variable(len(y), boolean=True)
        tv = cp.sum([W[i, j] * cp.abs(f[i] - f[j]) for i in range(len(y)) for j in range(len(y)) if W[i, j] > 0])
        fidelity = cp.sum([cp.abs(f[i] - y[i]) for i in range(len(y))])
        return cp.Problem(cp.Minimize(tv + lambda_param * fidelity))


class SemiSupervisedLearning(GraphOptimizationProblem):
    def setup_problem(self):
        W = nx.to_numpy_array(self.graph)
        L = self.params['L']
        y = self.params['y']
        f = cp.Variable(W.shape[0])
        tv = cp.sum(
            [W[i, j] * cp.abs(f[i] - f[j]) for i in range(W.shape[0]) for j in range(W.shape[0]) if W[i, j] > 0])
        constraints = [f[L] == y]
        return cp.Problem(cp.Minimize(tv), constraints)
