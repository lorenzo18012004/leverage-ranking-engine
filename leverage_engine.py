"""Moteur de classement de stratégies optionnelles par levier (Ω, Rachev, L, L^rob)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from mock_engine import BachelierPricer, MarketContext, ScenarioP, SimulationResult, Strategy, discretize_gaussian_mixture


@dataclass(frozen=True)
class EngineConfig:
    lambda1: float = 1.0
    lambda2: float = 1.0
    alpha: float = 0.05
    beta: float = 0.05
    pi_min: float = 1e-3
    epsilon: float = 0.05
    f_max: float = 0.85
    min_gross_premium: float = 0.0025


class TailStatistics:

    @staticmethod
    def _sorted(values: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(values)
        return values[order], probs[order]

    @classmethod
    def upper_tail_expectation(cls, values: np.ndarray, probs: np.ndarray, level: float) -> float:
        v, p = cls._sorted(values, probs)
        cum = np.cumsum(p)
        target = 1.0 - level
        idx = min(int(np.searchsorted(cum, target)), len(v) - 1)
        frac = 0.0 if p[idx] <= 0 else np.clip((cum[idx] - target) / p[idx], 0.0, 1.0)
        weights = np.zeros_like(p)
        weights[idx + 1:] = p[idx + 1:]
        weights[idx] = frac * p[idx]
        mass = weights.sum()
        return float(v[idx]) if mass <= 0 else float(np.dot(weights, v) / mass)

    @classmethod
    def lower_tail_expectation(cls, values: np.ndarray, probs: np.ndarray, level: float) -> float:
        v, p = cls._sorted(values, probs)
        cum = np.cumsum(p)
        idx = min(int(np.searchsorted(cum, level)), len(v) - 1)
        prev_cum = cum[idx - 1] if idx > 0 else 0.0
        frac = 0.0 if p[idx] <= 0 else np.clip((level - prev_cum) / p[idx], 0.0, 1.0)
        weights = np.zeros_like(p)
        weights[:idx] = p[:idx]
        weights[idx] = frac * p[idx]
        mass = weights.sum()
        return float(v[idx]) if mass <= 0 else float(np.dot(weights, v) / mass)

    @staticmethod
    def moments(values: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, float]:
        mean = float(np.dot(probs, values))
        var = float(np.dot(probs, (values - mean) ** 2))
        std = np.sqrt(var)
        if std <= 0:
            return mean, std, 0.0, 0.0
        skew = float(np.dot(probs, (values - mean) ** 3)) / std**3
        kurt = float(np.dot(probs, (values - mean) ** 4)) / std**4 - 3.0
        return mean, std, skew, kurt


class ScenarioDiscretizer:

    def __init__(self, grid: np.ndarray):
        self.grid = grid

    def probabilities(self, scenario: ScenarioP) -> np.ndarray:
        src_prices = np.asarray(scenario.prices)
        src_probs = np.asarray(scenario.probs)
        order = np.argsort(src_prices)
        src_prices = src_prices[order]
        src_cdf = np.cumsum(src_probs[order])
        src_cdf /= src_cdf[-1]

        midpoints = (self.grid[:-1] + self.grid[1:]) / 2
        edges = np.concatenate(([self.grid[0]], midpoints, [self.grid[-1]]))
        cdf_at_edges = np.interp(edges, src_prices, src_cdf, left=0.0, right=1.0)
        probs = np.clip(np.diff(cdf_at_edges), 0.0, None)
        probs /= probs.sum()
        return probs

    @staticmethod
    def default_for(
        market: MarketContext, scenario: ScenarioP, n_points: int = 2001, n_std: float = 6.0
    ) -> "ScenarioDiscretizer":
        market_spread = market.sigma * np.sqrt(market.T)
        prices = np.asarray(scenario.prices)
        lo, hi = float(prices.min()), float(prices.max())
        center = (lo + hi) / 2
        half_width = max((hi - lo) / 2 + n_std * market_spread, n_std * market_spread, 1e-6)
        grid = np.linspace(center - half_width, center + half_width, n_points)
        return ScenarioDiscretizer(grid)


class LeverageEngine:
    """Calcule Ω(0), Rachev, f, L et L^rob pour une stratégie sous un scénario donné."""

    def __init__(self, market: MarketContext, config: EngineConfig, discretizer: ScenarioDiscretizer, sigma_N: float):
        self.market = market
        self.config = config
        self.discretizer = discretizer
        self.sigma_N = sigma_N

    def omega_ratio(self, X: np.ndarray, probs: np.ndarray) -> tuple[float, float, float]:
        gain = float(np.dot(probs, np.maximum(X, 0.0)))
        loss = float(np.dot(probs, np.maximum(-X, 0.0)))
        denom = max(loss, self.config.pi_min * gain)
        omega = gain / denom if denom > 0 else 0.0
        return omega, gain, loss

    def rachev_ratio(self, X: np.ndarray, probs: np.ndarray) -> tuple[float, float, float | None]:
        etr = TailStatistics.upper_tail_expectation(X, probs, self.config.alpha)
        if etr <= 0:
            return 0.0, etr, None
        es = -TailStatistics.lower_tail_expectation(X, probs, self.config.beta)
        denom = max(es, self.config.pi_min * etr)
        rachev = etr / denom if denom > 0 else 0.0
        return rachev, etr, es

    def prudence_factor(self, kappa_e: float) -> float:
        t_n = 1.0 - np.exp(-max(0.0, kappa_e) / 4.0)
        f = 1.0 / (1.0 + self.config.lambda1 * self.sigma_N + self.config.lambda2 * t_n)
        return min(f, self.config.f_max)

    def _q_quantiles(self) -> dict[float, float]:
        F0, sigma, T = self.market.F0, self.market.sigma, self.market.T
        return {lvl: float(F0 + norm.ppf(lvl) * sigma * np.sqrt(T)) for lvl in (0.01, 0.99)}

    def _score_on_grid(self, strategy: Strategy, probs: np.ndarray) -> dict:
        grid = self.discretizer.grid
        X = strategy.payoff(grid) - strategy.pi_q_total

        omega, gain, loss = self.omega_ratio(X, probs)
        omega_tiebreak = gain / strategy.gross_pi_q_total if strategy.gross_pi_q_total > 0 else 0.0
        rachev, etr, es = self.rachev_ratio(X, probs)
        if etr <= 0:
            return {
                "omega": omega, "rachev": 0.0, "f": None, "L": 0.0,
                "etr": etr, "es": es, "gain": gain, "loss": loss, "kappa_e": None,
                "omega_tiebreak": omega_tiebreak,
            }

        _, _, _, kappa_e = TailStatistics.moments(grid, probs)
        f = self.prudence_factor(kappa_e)
        L = (omega**f) * (rachev ** (1 - f)) if omega > 0 else 0.0
        return {
            "omega": omega, "rachev": rachev, "f": f, "L": L,
            "etr": etr, "es": es, "gain": gain, "loss": loss, "kappa_e": kappa_e,
            "omega_tiebreak": omega_tiebreak,
        }

    def score(self, strategy: Strategy, scenario: ScenarioP) -> dict:
        probs = self.discretizer.probabilities(scenario)
        return self._score_on_grid(strategy, probs)

    def _gaussian_bump_on_grid(self, mu: float, sigma: float) -> np.ndarray:
        grid = self.discretizer.grid
        midpoints = (grid[:-1] + grid[1:]) / 2
        edges = np.concatenate(([-np.inf], midpoints, [np.inf]))
        probs = np.diff(norm.cdf(edges, loc=mu, scale=sigma))
        return probs / probs.sum()

    def robust_score(self, strategy: Strategy, scenario: ScenarioP, sigma_stress: float | None = None) -> dict:
        client_probs = self.discretizer.probabilities(scenario)
        nominal = self._score_on_grid(strategy, client_probs)

        if sigma_stress is None:
            _, sigma_stress, _, _ = TailStatistics.moments(self.discretizer.grid, client_probs)

        q = self._q_quantiles()
        eps = self.config.epsilon
        results = {"nominal": nominal}
        for label, center in (("stress_down", q[0.01]), ("stress_up", q[0.99])):
            stress_bump = self._gaussian_bump_on_grid(center, sigma_stress)
            mixed_probs = (1 - eps) * client_probs + eps * stress_bump
            results[label] = self._score_on_grid(strategy, mixed_probs)

        results["L_rob"] = min(results["nominal"]["L"], results["stress_down"]["L"], results["stress_up"]["L"])
        return results

    def rank(self, strategies: list[Strategy], scenario: ScenarioP, allow_open_risk: bool = True) -> pd.DataFrame:
        rows = []
        for strategy in strategies:
            if strategy.gross_pi_q_total < self.config.min_gross_premium:
                continue
            loss_info = strategy.loss_profile
            if not allow_open_risk and not loss_info["bounded"]:
                continue
            res = self.robust_score(strategy, scenario)
            n = res["nominal"]
            if n["gain"] < self.config.min_gross_premium and n["loss"] < self.config.min_gross_premium:
                continue
            rows.append(
                {
                    "name": strategy.name,
                    "pi_q_total": strategy.pi_q_total,
                    "gain": n["gain"], "loss": n["loss"], "etr": n["etr"], "es": n["es"],
                    "omega": n["omega"], "rachev": n["rachev"], "f": n["f"], "L": n["L"],
                    "omega_tiebreak": n["omega_tiebreak"],
                    "L_stress_down": res["stress_down"]["L"], "L_stress_up": res["stress_up"]["L"],
                    "L_rob": res["L_rob"],
                    "max_loss": loss_info["max_loss"], "open_risk_direction": loss_info["direction"],
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["L_rob", "omega_tiebreak"], ascending=False)
            .reset_index(drop=True)
        )


class AmericanPricer:

    @staticmethod
    def binomial_forward_price(F: float, K: float, sigma: float, T: float, r: float, option_type: str, n_steps: int = 200) -> float:
        dt = T / n_steps
        step = sigma * np.sqrt(dt)
        disc = np.exp(-r * dt)

        idx = np.arange(n_steps + 1)
        prices = F + step * (2 * idx - n_steps)
        values = np.maximum(prices - K, 0.0) if option_type == "C" else np.maximum(K - prices, 0.0)

        for i in range(n_steps - 1, -1, -1):
            idx = np.arange(i + 1)
            prices = F + step * (2 * idx - i)
            continuation = disc * 0.5 * (values[:-1] + values[1:])
            intrinsic = np.maximum(prices - K, 0.0) if option_type == "C" else np.maximum(K - prices, 0.0)
            values = np.maximum(continuation, intrinsic)

        return float(values[0]) * np.exp(r * T)

    @classmethod
    def assignment_charge(cls, strategy: Strategy, market: MarketContext, n_steps: int = 200) -> float:
        if market.margined:
            return 0.0 
        pricer = BachelierPricer(market)
        charge = 0.0
        for leg in strategy.legs:
            if leg.qty >= 0:
                continue 
            euro_price = pricer.forward_price(leg.strike, leg.option_type)
            amer_price = cls.binomial_forward_price(
                market.F0, leg.strike, market.sigma, market.T, market.r, leg.option_type, n_steps
            )
            charge += abs(leg.qty) * max(amer_price - euro_price, 0.0)
        return charge


class Top3Refiner:

    def __init__(self, engine: LeverageEngine, n_steps: int = 200):
        self.engine = engine
        self.n_steps = n_steps

    def refine(
        self, ranked_df: pd.DataFrame, strategies_by_name: dict[str, Strategy], scenario: ScenarioP, top_n: int = 3
    ) -> pd.DataFrame:
        top = ranked_df.head(top_n)
        records = []
        for _, row in top.iterrows():
            strategy = strategies_by_name[row["name"]]
            charge = AmericanPricer.assignment_charge(strategy, self.engine.market, self.n_steps)
            adjusted = Strategy(
                strategy.name, strategy.legs, strategy.pi_q_total + charge,
                strategy.p_paid, strategy.gross_pi_q_total,
            )
            res = self.engine.robust_score(adjusted, scenario)
            records.append(
                {
                    "name": row["name"],
                    "assignment_charge": charge,
                    "L_rob_nominal": row["L_rob"],
                    "L_rob_adjusted": res["L_rob"],
                }
            )
        out = pd.DataFrame(records).sort_values("L_rob_adjusted", ascending=False).reset_index(drop=True)
        out.attrs["rank_changed"] = out["name"].tolist() != top["name"].tolist()
        return out


class Calibrator:

    def __init__(self, market: MarketContext, strategies: list[Strategy]):
        self.market = market
        self.strategies = strategies

    def synthetic_library(self, F0: float, spread: float, n_points: int = 80) -> dict[str, ScenarioP]:
        narratives = {
            "bimodal": ((0.5, 0.5), (F0 - spread, F0 + spread), (spread * 0.25, spread * 0.25)),
            "heavy_tailed": (
                (0.7, 0.2, 0.1),
                (F0, F0 - 2.5 * spread, F0 + 2.5 * spread),
                (spread * 0.3, spread * 0.6, spread * 0.6),
            ),
            "uniform_like": (
                tuple(1 / 9 for _ in range(9)),
                tuple(float(F0 + spread * k) for k in np.linspace(-1.5, 1.5, 9)),
                tuple(spread * 0.35 for _ in range(9)),
            ),
        }
        library = {}
        for name, (weights, mus, sigmas) in narratives.items():
            extent = max(abs(mu - F0) + 5 * sig for mu, sig in zip(mus, sigmas))
            grid = np.linspace(F0 - extent, F0 + extent, n_points)
            probs = discretize_gaussian_mixture(weights, mus, sigmas, grid)
            library[name] = ScenarioP(tuple(float(p) for p in grid), tuple(float(p) for p in probs))
        return library

    def score(
        self,
        lambda1: float,
        lambda2: float,
        library: dict[str, ScenarioP],
        sigma_N: float,
        n_perturbations: int = 5,
        jitter: float = 0.1,
        seed: int = 0,
        allow_open_risk: bool = True,
    ):
        rng = np.random.default_rng(seed)
        config = EngineConfig(lambda1=lambda1, lambda2=lambda2)
        discrimination, stability, open_risk_penalty = [], [], []

        for scenario in library.values():
            discretizer = ScenarioDiscretizer.default_for(self.market, scenario)
            engine = LeverageEngine(self.market, config, discretizer, sigma_N)
            df = engine.rank(self.strategies, scenario, allow_open_risk=allow_open_risk)
            discrimination.append(df["L_rob"].std())

            base_top3 = set(df.head(3)["name"])
            matches = 0
            for _ in range(n_perturbations):
                jittered = ScenarioP(
                    tuple(p + rng.normal(0, jitter) for p in scenario.prices),
                    scenario.probs,
                )
                jdf = engine.rank(self.strategies, jittered, allow_open_risk=allow_open_risk)
                matches += len(base_top3 & set(jdf.head(3)["name"]))
            stability.append(matches / (n_perturbations * 3))

            open_mask = df["open_risk_direction"].notna()
            if open_mask.any() and (~open_mask).any():
                open_risk_penalty.append(df.loc[~open_mask, "L_rob"].mean() - df.loc[open_mask, "L_rob"].mean())

        return {
            "discrimination": float(np.mean(discrimination)),
            "stability": float(np.mean(stability)),
            "open_risk_penalty": float(np.mean(open_risk_penalty)) if open_risk_penalty else 0.0,
        }

    def calibrate(
        self,
        library: dict[str, ScenarioP],
        sigma_N: float,
        lambda_grid: np.ndarray = np.linspace(0.0, 4.0, 5),
        **kwargs,
    ) -> tuple[tuple[float, float], pd.DataFrame]:
        records = []
        for l1 in lambda_grid:
            for l2 in lambda_grid:
                scores = self.score(l1, l2, library, sigma_N, **kwargs)
                composite = scores["discrimination"] + scores["stability"] + scores["open_risk_penalty"]
                records.append({"lambda1": l1, "lambda2": l2, **scores, "composite": composite})

        results = pd.DataFrame(records).sort_values("composite", ascending=False).reset_index(drop=True)
        best = results.iloc[0]
        return (float(best["lambda1"]), float(best["lambda2"])), results


class SimulationReport:

    def __init__(
        self,
        simulation: SimulationResult,
        ranked: pd.DataFrame,
        refined: pd.DataFrame,
        calibration: dict | None = None,
    ):
        self.simulation = simulation
        self.ranked = ranked
        self.refined = refined
        self.calibration = calibration

    def _chain_table(self, strikes: list[float]) -> pd.DataFrame:
        rows = [
            {
                "strike": K,
                "call_PiQ_fwd": self.simulation.chain[(K, "C")],
                "put_PiQ_fwd": self.simulation.chain[(K, "P")],
            }
            for K in strikes
        ]
        return pd.DataFrame(rows)

    def print(self, top_n: int = 10):
        m = self.simulation.market
        sc = self.simulation.scenario

        if self.calibration is not None:
            print("=" * 90)
            print("0. Calibration de (lambda1, lambda2) par recherche sur grille (section 7)")
            print("=" * 90)
            print(
                f"Echantillon de calibration : {self.calibration['n_strategies_used']} strategies "
                f"(sur {len(self.simulation.strategies)} au total dans l'univers)"
            )
            print(f"lambda1={self.calibration['lambda1']:.2f}  lambda2={self.calibration['lambda2']:.2f}")
            print(self.calibration["results"].head(5).to_string(index=False))
            print()

        print("=" * 90)
        print("1. Marche simule (moteur amont)")
        print("=" * 90)
        print(
            f"F0={m.F0:.4f}  r={m.r:.2%}  T={m.T:.2f} an  sigma={m.sigma:.3f}  "
            f"margined={m.margined}  sigma_N={self.simulation.sigma_N:.2f}"
        )
        print(f"Contrainte client - perte illimitee autorisee : {self.simulation.allow_unlimited_loss}")

        prices = np.array(sc.prices)
        probs = np.array(sc.probs)
        mean, std, skew, kurt = TailStatistics.moments(prices, probs)
        print(f"\nScenario client P (distribution discrete, {len(sc.prices)} points) :")
        print(f"  moyenne={mean:.4f}  ecart-type={std:.4f}  skewness={skew:.3f}  exces_kurtosis={kurt:.3f}")
        preview_idx = np.linspace(0, len(prices) - 1, min(10, len(prices))).round().astype(int)
        order = np.argsort(prices)
        preview = pd.DataFrame({"prix": prices[order][preview_idx], "proba": probs[order][preview_idx]})
        print(preview.to_string(index=False))

        strikes = sorted(self.simulation.strikes)
        lo, hi, step = strikes[0], strikes[-1], strikes[1] - strikes[0]
        print(
            f"\nChaine d'options : {len(strikes)} strikes, de {lo:.4f} a {hi:.4f} "
            f"(pas {step:.4f}), valeurs forward ΠQ. Fenetre autour de l'ATM :"
        )
        atm_idx = int(np.argmin(np.abs(np.array(strikes) - m.F0)))
        window = strikes[max(0, atm_idx - 10) : atm_idx + 11]
        print(self._chain_table(window).to_string(index=False))

        n_open_risk = sum(1 for s in self.simulation.strategies if not s.loss_profile["bounded"])
        print(
            f"\nUnivers de strategies genere : {len(self.simulation.strategies)} "
            f"(dont {n_open_risk} a risque ouvert) -> {len(self.ranked)} conservees apres filtre client"
        )

        print("\n" + "=" * 90)
        print("2. Resolution : Omega(0), Rachev, f, L, L_rob par strategie (Top {})".format(top_n))
        print("=" * 90)
        cols = [
            "name", "pi_q_total", "gain", "loss", "omega", "omega_tiebreak", "etr", "es", "rachev", "f", "L",
            "L_stress_down", "L_stress_up", "L_rob", "max_loss", "open_risk_direction",
        ]
        print(self.ranked[cols].head(top_n).to_string(index=False))

        print("\n" + "=" * 90)
        print("3. Raffinement Top-3 : impact du style americain (arbre binomial)")
        print("=" * 90)
        print(self.refined.to_string(index=False))
        print(f"\nOrdre du Top-3 modifie par le raffinement : {self.refined.attrs.get('rank_changed')}")


if __name__ == "__main__":
    import sys

    from mock_engine import MarketSimulator

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    simulation = MarketSimulator().simulate()
    market = simulation.market

    calib_rng = np.random.default_rng()
    n_calib = min(300, len(simulation.strategies))
    calib_idx = calib_rng.choice(len(simulation.strategies), size=n_calib, replace=False)
    calib_strategies = [simulation.strategies[i] for i in calib_idx]

    calibrator = Calibrator(market, calib_strategies)
    spread = market.sigma * np.sqrt(market.T)
    library = calibrator.synthetic_library(market.F0, spread)
    (lambda1, lambda2), calib_results = calibrator.calibrate(
        library, simulation.sigma_N, lambda_grid=np.linspace(0.0, 4.0, 4), n_perturbations=3
    )

    config = EngineConfig(lambda1=lambda1, lambda2=lambda2)
    discretizer = ScenarioDiscretizer.default_for(market, simulation.scenario)
    engine = LeverageEngine(market, config, discretizer, simulation.sigma_N)

    ranked = engine.rank(simulation.strategies, simulation.scenario, allow_open_risk=simulation.allow_unlimited_loss)
    strategies_by_name = {s.name: s for s in simulation.strategies}
    refined = Top3Refiner(engine).refine(ranked, strategies_by_name, simulation.scenario)

    calibration_info = {
        "lambda1": lambda1, "lambda2": lambda2,
        "n_strategies_used": n_calib, "results": calib_results,
    }
    SimulationReport(simulation, ranked, refined, calibration_info).print()
