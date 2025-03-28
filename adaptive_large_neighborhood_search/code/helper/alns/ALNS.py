import warnings
from collections import OrderedDict

import numpy as np
import numpy.random as rnd

from .Result import Result
from .State import State  # pylint: disable=unused-import
from .Statistics import Statistics
from .criteria import AcceptanceCriterion  # pylint: disable=unused-import
from .select_operator import select_operator
from .tools.warnings import OverwriteWarning

# progress bar
from tqdm import tqdm

# Weights
_IS_BEST = 0
_IS_BETTER = 1
_IS_ACCEPTED = 2
_IS_REJECTED = 3

# Callbacks
_ON_BEST = 0


class ALNS:

    def __init__(self, rnd_state=rnd.RandomState()):
        """
        Implements the adaptive large neighbourhood search (ALNS) algorithm.
        The implementation optimises for a minimisation problem, as explained
        in the text by Pisinger and Røpke (2010).

        Parameters
        ----------
        rnd_state : rnd.RandomState
            Optional random state to use for random number generation. When
            passed, this state is used for operator selection and general
            computations requiring random numbers. It is also passed to the
            destroy and repair operators, as a second argument.

        References
        ----------
        - Pisinger, D., and Røpke, S. (2010). Large Neighborhood Search. In M.
          Gendreau (Ed.), *Handbook of Metaheuristics* (2 ed., pp. 399-420).
          Springer.
        """
        super().__init__()

        self._destroy_operators = OrderedDict()
        self._repair_operators = OrderedDict()
        self._callbacks = {}

        self._rnd_state = rnd_state

    @property
    def destroy_operators(self):
        """
        Returns the destroy operators set for the ALNS algorithm.

        Returns
        -------
        list
            A list of (name, operator) tuples. Their order is the same as the
            one in which they were passed to the ALNS instance.
        """
        return list(self._destroy_operators.items())

    @property
    def repair_operators(self):
        """
        Returns the repair operators set for the ALNS algorithm.

        Returns
        -------
        list
            A list of (name, operator) tuples. Their order is the same as the
            one in which they were passed to the ALNS instance.
        """
        return list(self._repair_operators.items())

    def add_destroy_operator(self, operator, name=None):
        """
        Adds a destroy operator to the heuristic instance.

        Parameters
        ----------
        operator : Callable[[State, RandomState], State]
            An operator that, when applied to the current state, returns a new
            state reflecting its implemented destroy action. The second argument
            is the random state constructed from the passed-in seed.
        name : str
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        self._add_operator(self._destroy_operators, operator, name)

    def add_repair_operator(self, operator, name=None):
        """
        Adds a repair operator to the heuristic instance.

        Parameters
        ----------
        operator : Callable[[State, RandomState], State]
            An operator that, when applied to the destroyed state, returns a
            new state reflecting its implemented repair action. The second
            argument is the random state constructed from the passed-in seed.
        name : str
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        self._add_operator(self._repair_operators, operator, name)

    # def iterate(self, initial_solution, weights, operator_decay, criterion,
    #            iterations=10000, collect_stats=True, time_limit=None, max_non_improving=None, callbacks=None):
    #     """
    #     Runs the adaptive large neighbourhood search heuristic [1], using the
    #     previously set destroy and repair operators. The first solution is set
    #     to the passed-in initial solution, and then subsequent solutions are
    #     computed by iteratively applying the operators.

    #     Parameters
    #     ----------
    #     initial_solution : State
    #         The initial solution, as a State object.
    #     weights: array_like
    #         Initial list of four non-negative elements for weight updates.
    #         Can be updated by callbacks during iteration.
    #     operator_decay : float
    #         Initial operator decay parameter in the unit interval [0, 1].
    #         Can be updated by callbacks during iteration.
    #     criterion : AcceptanceCriterion
    #         Initial acceptance criterion. Can be updated by callbacks.
    #     iterations : int
    #         The number of iterations. Default 10000.
    #     collect_stats : bool
    #         Should statistics be collected during iteration? Default True.
    #     time_limit : float
    #         Maximum runtime in seconds. Default None.
    #     max_non_improving : int
    #         Maximum number of non-improving iterations. Default None.
    #     callbacks : dict
    #         Dictionary of callback functions. Supported callbacks:
    #         - 'on_iteration': Called after each iteration with current iteration number.
    #           Should return a dict with 'weights', 'criterion', and 'adaptation' keys
    #           for dynamic updates.

    #     Returns
    #     -------
    #     Result
    #         A result object, containing the best solution and statistics.
    #     """
    #     import time 

    #     weights = np.asarray(weights, dtype=np.float16)
    #     self._validate_parameters(weights, operator_decay, iterations)
        
    #     current = best = initial_solution
    #     d_weights = np.ones(len(self.destroy_operators), dtype=np.float16)
    #     r_weights = np.ones(len(self.repair_operators), dtype=np.float16)
    #     statistics = Statistics() if collect_stats else None
        
    #     if collect_stats:
    #         statistics.collect_objective(initial_solution.objective())

    #     start_time = time.time()
    #     non_improving_count = 0
    #     current_criterion = criterion
    #     current_weights = weights
    #     current_decay = operator_decay

    #     for iteration in tqdm(range(iterations)):
    #         # Check time limit
    #         if time_limit is not None and time.time() - start_time > time_limit:
    #             break
                
    #         # Check non-improving iterations limit
    #         if max_non_improving is not None and non_improving_count >= max_non_improving:
    #             break

    #         # Handle callbacks for phase switching and updates
    #         if callbacks and 'on_iteration' in callbacks:
    #             phase_info = callbacks['on_iteration'](iteration)
    #             if isinstance(phase_info, dict):
    #                 if 'weights' in phase_info:
    #                     current_weights = np.asarray(phase_info['weights'], dtype=np.float16)
    #                 if 'criterion' in phase_info:
    #                     current_criterion = phase_info['criterion']
    #                 if 'adaptation' in phase_info:
    #                     current_decay = phase_info['adaptation']

    #         # Select and apply operators
    #         d_idx = select_operator(self.destroy_operators, d_weights, self._rnd_state)
    #         r_idx = select_operator(self.repair_operators, r_weights, self._rnd_state)

    #         d_name, d_operator = self.destroy_operators[d_idx]
    #         destroyed = d_operator(current, self._rnd_state)

    #         r_name, r_operator = self.repair_operators[r_idx]
    #         candidate = r_operator(destroyed, self._rnd_state)

    #         # Consider the candidate solution
    #         prev_best_objective = best.objective()
    #         best, current, weight_idx = self._consider_candidate(
    #             best, current, candidate, current_criterion
    #         )
            
    #         # Update non-improving counter
    #         if max_non_improving is not None:
    #             if best.objective() < prev_best_objective:
    #                 non_improving_count = 0
    #             else:
    #                 non_improving_count += 1

    #         # Update operator weights using current phase weights and decay
    #         d_weights[d_idx] *= current_decay
    #         d_weights[d_idx] += (1 - current_decay) * current_weights[weight_idx]

    #         r_weights[r_idx] *= current_decay
    #         r_weights[r_idx] += (1 - current_decay) * current_weights[weight_idx]

    #         # Collect statistics
    #         if collect_stats:
    #             statistics.collect_objective(current.objective())
    #             statistics.collect_destroy_operator(d_name, weight_idx)
    #             statistics.collect_repair_operator(r_name, weight_idx)

    #     return Result(best, statistics)
    
    def iterate(self, initial_solution, weights, operator_decay, criterion,
            iterations=10000, collect_stats=True, time_limit=280, max_non_improving=100):
        """
        Runs the adaptive large neighbourhood search heuristic [1], using the
        previously set destroy and repair operators. The first solution is set
        to the passed-in initial solution, and then subsequent solutions are
        computed by iteratively applying the operators.

        Parameters
        ----------
        initial_solution : State
            The initial solution, as a State object.
        weights: array_like
            A list of four non-negative elements, representing the weight
            updates when the candidate solution results in a new global best
            (idx 0), is better than the current solution (idx 1), the solution
            is accepted (idx 2), or rejected (idx 3).
        operator_decay : float
            The operator decay parameter, as a float in the unit interval,
            [0, 1] (inclusive).
        criterion : AcceptanceCriterion
            The acceptance criterion to use for candidate states. See also
            the `alns.criteria` module for an overview.
        iterations : int
            The number of iterations. Default 10000.
        collect_stats : bool
            Should statistics be collected during iteration? Default True, but
            may be turned off for long runs to reduce memory consumption.

        Raises
        ------
        ValueError
            When the parameters do not meet requirements.

        Returns
        -------
        Result
            A result object, containing the best solution and some additional
            statistics.

        References
        ----------
        [1]: Pisinger, D., & Røpke, S. (2010). Large Neighborhood Search. In M.
        Gendreau (Ed.), *Handbook of Metaheuristics* (2 ed., pp. 399-420).
        Springer.

        [2]: S. Røpke and D. Pisinger (2006). A unified heuristic for a large
        class of vehicle routing problems with backhauls. *European Journal of
        Operational Research*, 171: 750–775, 2006.
        """
        import time 

        weights = np.asarray(weights, dtype=np.float16)

        self._validate_parameters(weights, operator_decay, iterations)

        current = best = initial_solution

        d_weights = np.ones(len(self.destroy_operators), dtype=np.float16)
        r_weights = np.ones(len(self.repair_operators), dtype=np.float16)

        statistics = Statistics()

        if collect_stats:
            statistics.collect_objective(initial_solution.objective())

        start_time = time.time()
        non_improving_count = 0
        for iteration in tqdm(range(iterations)):
            if time_limit is not None and time.time() - start_time > time_limit:
                print(f"\nStopping: Time limit of {time_limit}s reached")
                break
                
            # Check non-improving iterations limit
            if max_non_improving is not None and non_improving_count >= max_non_improving:
                print(f"\nStopping: {max_non_improving} non-improving iterations reached")
                break

            d_idx = select_operator(self.destroy_operators, d_weights,
                                    self._rnd_state)

            r_idx = select_operator(self.repair_operators, r_weights,
                                    self._rnd_state)

            d_name, d_operator = self.destroy_operators[d_idx]
            destroyed = d_operator(current, self._rnd_state)

            r_name, r_operator = self.repair_operators[r_idx]
            candidate = r_operator(destroyed, self._rnd_state)

            prev_best_objective = best.objective()
            best, current, weight_idx = self._consider_candidate(best,
                                                                 current,
                                                                 candidate,
                                                                 criterion)
            
            if max_non_improving is not None:
                if best.objective() > prev_best_objective:
                    non_improving_count = 0  # Reset counter if new best found
                else:
                    non_improving_count += 1

            # The weights are updated as convex combinations of the current
            # weight and the update parameter. See eq. (2), p. 12.
            d_weights[d_idx] *= operator_decay
            d_weights[d_idx] += (1 - operator_decay) * weights[weight_idx]

            r_weights[r_idx] *= operator_decay
            r_weights[r_idx] += (1 - operator_decay) * weights[weight_idx]

            if collect_stats:
                statistics.collect_objective(current.objective())
                statistics.collect_destroy_operator(d_name, weight_idx)
                statistics.collect_repair_operator(r_name, weight_idx)

        return Result(best, statistics if collect_stats else None)

    def on_best(self, func):
        """
        Sets a callback function to be called when ALNS finds a new global best
        solution state.

        Parameters
        ----------
        func : callable
            A function that should take a solution State as its first parameter,
            and a numpy RandomState as its second (cf. the operator signature).
            It should return a (new) solution State.

        Warns
        -----
        OverwriteWarning
            When a callback has already been set.
        """
        self._set_callback(_ON_BEST, func)

    @staticmethod
    def _add_operator(operators, operator, name=None):
        """
        Internal helper that adds an operator to the passed-in operator
        dictionary. See `add_destroy_operator` and `add_repair_operator` for
        public methods that use this helper.

        Parameters
        ----------
        operators : dict
            Dictionary of (name, operator) key-value pairs.
        operator : Callable[[State, RandomState], State]
            Callable operator function.
        name : str
            Optional operator name.

        Warns
        -----
        OverwriteWarning
            When the operator name already maps to an operator on this ALNS
            instance.
        """
        if name is None:
            name = operator.__name__

        if name in operators:
            warnings.warn("The ALNS instance already knows an operator by the"
                          " name `{0}'. This operator will now be replaced with"
                          " the newly passed-in operator. If this is not what"
                          " you intended, consider explicitly naming your"
                          " operators via the `name' argument.".format(name),
                          OverwriteWarning)

        operators[name] = operator

    def _consider_candidate(self, best, current, candidate, criterion):
        """
        Considers the candidate solution by comparing it against the best and
        current solutions. Returns the new solution when it is better or
        accepted, or the current in case it is rejected. Candidate solutions
        are accepted based on the passed-in acceptance criterion.

        Parameters
        ----------
        best : State
            Best solution encountered so far.
        current : State
            Current solution.
        candidate : State
            Candidate solution.
        criterion : AcceptanceCriterion
            The chosen acceptance criterion.

        Returns
        -------
        State
            The (possibly new) best state.
        State
            The (possibly new) current state.
        int
            The weight index to use when updating the operator weights.
        """
        if criterion.accept(self._rnd_state, best, current, candidate):
            if candidate.objective() > current.objective():
                weight = _IS_BETTER
            else:
                weight = _IS_ACCEPTED

            current = candidate
        else:
            weight = _IS_REJECTED

        if candidate.objective() > best.objective():
            # Is a new global best, so we might want to do something to further
            # improve the solution.
            if _ON_BEST in self._callbacks:
                callback = self._callbacks[_ON_BEST]
                candidate = callback(candidate, self._rnd_state)

            # Global best solution becomes the new starting point for further
            # iterations.
            return candidate, candidate, _IS_BEST

        # Best has not been updated if we get here, but the current state might
        # have (if the candidate was accepted).
        return best, current, weight

    def _validate_parameters(self, weights, operator_decay, iterations):
        """
        Helper method to validate the passed-in ALNS parameters.
        """
        if len(self.destroy_operators) == 0 or len(self.repair_operators) == 0:
            raise ValueError("Missing at least one destroy or repair operator.")

        if not (0 <= operator_decay <= 1):
            raise ValueError("Operator decay parameter outside unit interval"
                             " is not understood.")

        if any(weight < 0 for weight in weights):
            raise ValueError("Negative weights are not understood.")

        if len(weights) < 4:
            # More than four is not explicitly problematic, as we only use the
            # first four anyways.
            raise ValueError("Unsupported number of weights: expected 4,"
                             " found {0}.".format(len(weights)))

        if iterations < 0:
            raise ValueError("Negative number of iterations.")

    def _set_callback(self, flag, func):
        """
        Sets the passed-in callback func for the passed-in flag. Warns if this
        would overwrite an existing callback.
        """
        if flag in self._callbacks:
            warnings.warn("A callback function has already been set for the"
                          " `{0}' flag. This callback will now be replaced by"
                          " the newly passed-in callback.".format(flag),
                          OverwriteWarning)

        self._callbacks[flag] = func
