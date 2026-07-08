"""Simulateur du moteur amont : marché, chaîne d'options, univers de stratégies, scénario client."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class MarketContext:
    F0: float
    r: float
    T: float
    sigma: float
    margined: bool = True


@dataclass(frozen=True)
class Leg:
    option_type: str 
    strike: float
    qty: float

    def intrinsic(self, prices: np.ndarray):
        if self.option_type == "C":
            return self.qty * np.maximum(prices - self.strike, 0.0)
        return self.qty * np.maximum(self.strike - prices, 0.0)


@dataclass(frozen=True)
class Strategy:
    name: str
    legs: tuple[Leg, ...]
    pi_q_total: float
    p_paid: float
    gross_pi_q_total: float

    def payoff(self, prices: np.ndarray):
        total = np.zeros_like(prices, dtype=float)
        for leg in self.legs:
            total += leg.intrinsic(prices)
        return total

    @property
    def upside_unbounded(self) -> bool:
        return sum(leg.qty for leg in self.legs if leg.option_type == "C") < 0

    @property
    def downside_unbounded(self) -> bool:
        return sum(leg.qty for leg in self.legs if leg.option_type == "P") < 0

    @property
    def loss_profile(self) -> dict:
        if self.upside_unbounded or self.downside_unbounded:
            direction = "/".join(
                d for d, flag in (("down", self.downside_unbounded), ("up", self.upside_unbounded)) if flag
            )
            return {"bounded": False, "max_loss": None, "direction": direction}
        strikes = np.array(sorted({leg.strike for leg in self.legs})) if self.legs else np.array([0.0])
        worst_payoff = float(self.payoff(strikes).min())
        return {"bounded": True, "max_loss": self.pi_q_total - worst_payoff, "direction": None}


@dataclass(frozen=True)
class ScenarioP:
    """Distribution client discrète : au moins 50 points (prix, probabilité)."""

    prices: tuple[float, ...]
    probs: tuple[float, ...]


def discretize_gaussian_mixture(
    weights: tuple[float, ...], mus: tuple[float, ...], sigmas: tuple[float, ...], grid: np.ndarray
) -> np.ndarray:
    """Discrétise une mixture de gaussiennes sur `grid` par différence de CDF (section 7)."""
    midpoints = (grid[:-1] + grid[1:]) / 2
    edges = np.concatenate(([-np.inf], midpoints, [np.inf]))
    probs = np.zeros_like(grid, dtype=float)
    for w, mu, sig in zip(weights, mus, sigmas):
        cdf = norm.cdf(edges, loc=mu, scale=sig)
        probs += w * np.diff(cdf)
    probs /= probs.sum()
    return probs


@dataclass(frozen=True)
class SimulationResult:
    market: MarketContext
    chain: dict[tuple[float, str], float]
    strikes: list[float]
    strategies: list[Strategy]
    scenario: ScenarioP
    sigma_N: float
    allow_unlimited_loss: bool  # contrainte client : perte illimitée autorisée (oui/non)


class BachelierPricer:
    """Pricer forward (non actualisé) sous le modèle normal de Bachelier."""

    def __init__(self, market: MarketContext):
        self.market = market

    def forward_price(self, K: float, option_type: str) -> float:
        F, sigma, T = self.market.F0, self.market.sigma, self.market.T
        if T <= 0 or sigma <= 0:
            return max(F - K, 0.0) if option_type == "C" else max(K - F, 0.0)
        scaled = sigma * np.sqrt(T)
        d = (F - K) / scaled
        if option_type == "C":
            return float((F - K) * norm.cdf(d) + scaled * norm.pdf(d))
        return float((K - F) * norm.cdf(-d) + scaled * norm.pdf(d))

    def build_chain(self, strikes: list[float]) -> dict[tuple[float, str], float]:
        return {
            (K, option_type): self.forward_price(K, option_type)
            for K in strikes
            for option_type in ("C", "P")
        }


@dataclass(frozen=True)
class _Template:
    name: str
    leg_specs: tuple[tuple[int, str, float], ...]  # (offset, type, qty)


class StrategyUniverseGenerator:
    """Génère l'univers de stratégies par templates (2 à 4 jambes) glissés sur la grille de strikes."""

    TEMPLATES: tuple[_Template, ...] = (
        _Template("naked_call_long", ((0, "C", 1),)),
        _Template("naked_call_short", ((0, "C", -1),)),
        _Template("naked_put_long", ((0, "P", 1),)),
        _Template("naked_put_short", ((0, "P", -1),)),
        _Template("call_spread_debit", ((0, "C", 1), (1, "C", -1))),
        _Template("call_spread_credit", ((0, "C", -1), (1, "C", 1))),
        _Template("put_spread_debit", ((0, "P", -1), (1, "P", 1))),
        _Template("put_spread_credit", ((0, "P", 1), (1, "P", -1))),
        _Template("call_ladder_1x2", ((0, "C", 1), (1, "C", -2))),
        _Template("put_ladder_1x2", ((0, "P", -2), (1, "P", 1))),
        _Template("call_fly", ((0, "C", 1), (1, "C", -2), (2, "C", 1))),
        _Template("put_fly", ((0, "P", 1), (1, "P", -2), (2, "P", 1))),
        _Template("call_condor", ((0, "C", 1), (1, "C", -1), (2, "C", -1), (3, "C", 1))),
        _Template("put_condor", ((0, "P", 1), (1, "P", -1), (2, "P", -1), (3, "P", 1))),
        _Template("straddle_long", ((0, "C", 1), (0, "P", 1))),
        _Template("straddle_short", ((0, "C", -1), (0, "P", -1))),
        _Template("strangle_long", ((0, "P", 1), (1, "C", 1))),
        _Template("strangle_short", ((0, "P", -1), (1, "C", -1))),
        _Template("risk_reversal", ((0, "P", -1), (1, "C", 1))),
    )

    def __init__(self, widths: tuple[int, ...] = (1, 2, 3)):
        self.widths = widths

    def generate(self, market: MarketContext, chain: dict, strikes: list[float]) -> list[Strategy]:
        strikes_sorted = sorted(strikes)
        n = len(strikes_sorted)
        seen: set[tuple] = set()
        strategies: list[Strategy] = []

        for template in self.TEMPLATES:
            for width in self.widths:
                for start_idx in range(n):
                    leg_idxs = [start_idx + spec[0] * width for spec in template.leg_specs]
                    if max(leg_idxs) >= n:
                        continue
                    legs = tuple(
                        Leg(option_type, strikes_sorted[idx], qty)
                        for (_, option_type, qty), idx in zip(template.leg_specs, leg_idxs)
                    )
                    key = tuple(sorted((leg.strike, leg.option_type, leg.qty) for leg in legs))
                    if key in seen:
                        continue
                    seen.add(key)

                    pi_q_total = sum(leg.qty * chain[(leg.strike, leg.option_type)] for leg in legs)
                    gross_pi_q_total = sum(abs(leg.qty * chain[(leg.strike, leg.option_type)]) for leg in legs)
                    p_paid = 0.0 if market.margined else pi_q_total * np.exp(-market.r * market.T)
                    name = f"{template.name}_{strikes_sorted[start_idx]:.4f}_w{width}"
                    strategies.append(
                        Strategy(
                            name=name, legs=legs, pi_q_total=pi_q_total,
                            p_paid=p_paid, gross_pi_q_total=gross_pi_q_total,
                        )
                    )

        return strategies


class MarketSimulator:
    """Simule le moteur amont : tire un marché, un scénario client et un univers de stratégies.

    Sans seed explicite, chaque appel à `simulate()` puise une nouvelle entropie
    (numpy `default_rng()`), donc chaque lancement du script produit un contexte différent.
    """

    def __init__(
        self,
        f0_range: tuple[float, float] = (95.0, 99.0),
        r_range: tuple[float, float] = (0.03, 0.06),
        t_range: tuple[float, float] = (0.25, 1.0),
        sigma_range: tuple[float, float] = (0.30, 0.70),
        chain_half_width_stdevs: tuple[float, float] = (2.0, 3.5),  # bornes de liquidite reelle, pas la queue extreme
        n_strikes_range: tuple[int, int] = (80, 160),  # chaine dense comme sur un vrai listing SOFR/Euribor
        n_scenario_narratives_range: tuple[int, int] = (2, 4),
        n_scenario_points_range: tuple[int, int] = (60, 120),
        sigma_N_range: tuple[float, float] = (0.05, 0.95),
        widths: tuple[int, ...] = (1, 2, 3),
        unlimited_loss_allowed_prob: float = 0.7,
        rng: np.random.Generator | None = None,
    ):
        self.f0_range = f0_range
        self.r_range = r_range
        self.t_range = t_range
        self.sigma_range = sigma_range
        self.chain_half_width_stdevs = chain_half_width_stdevs
        self.n_strikes_range = n_strikes_range
        self.n_scenario_narratives_range = n_scenario_narratives_range
        self.n_scenario_points_range = n_scenario_points_range
        self.sigma_N_range = sigma_N_range
        self.widths = widths
        self.unlimited_loss_allowed_prob = unlimited_loss_allowed_prob
        self.rng = rng if rng is not None else np.random.default_rng()

    def _random_market(self) -> MarketContext:
        return MarketContext(
            F0=float(self.rng.uniform(*self.f0_range)),
            r=float(self.rng.uniform(*self.r_range)),
            T=float(self.rng.uniform(*self.t_range)),
            sigma=float(self.rng.uniform(*self.sigma_range)),
            margined=bool(self.rng.integers(0, 2)),
        )

    def _random_scenario(self, market: MarketContext) -> ScenarioP:
        # Le client exprime sa vue comme quelques scenarios narratifs distincts (ex. "deux
        # baisses de la Fed" vs "statu quo", section 6.1) ; on discretise ensuite ce melange
        # latent sur une grille fine (>= 50 points, section 7) pour obtenir la distribution
        # reellement transmise au moteur de levier.
        n_narratives = int(self.rng.integers(*self.n_scenario_narratives_range))
        weights = tuple(float(w) for w in self.rng.dirichlet(np.ones(n_narratives)))
        market_spread = market.sigma * np.sqrt(market.T)
        offsets = self.rng.normal(0.0, market_spread * 0.7, n_narratives)
        sigmas = tuple(float(s) for s in self.rng.uniform(0.4, 1.1, n_narratives) * market_spread)
        mus = tuple(float(market.F0 + o) for o in offsets)

        n_points = int(self.rng.integers(*self.n_scenario_points_range))
        extent = max(abs(mu - market.F0) + 5 * sig for mu, sig in zip(mus, sigmas))
        grid = np.linspace(market.F0 - extent, market.F0 + extent, n_points)
        probs = discretize_gaussian_mixture(weights, mus, sigmas, grid)

        return ScenarioP(tuple(float(p) for p in grid), tuple(float(p) for p in probs))

    def simulate(self) -> SimulationResult:
        market = self._random_market()

        half = int(self.rng.integers(*self.n_strikes_range)) // 2
        # Largeur de chaine indexee sur sigma*sqrt(T) (pas une plage fixe) : au-dela de
        # ~8 sigma sous Q, les prix Bachelier sous-flottent a zero et les ratios gain/perte
        # deviennent du bruit numerique divise par du bruit numerique (aucune bourse ne
        # liste des strikes aussi loin de toute facon).
        market_std = market.sigma * np.sqrt(market.T)
        half_width = float(self.rng.uniform(*self.chain_half_width_stdevs)) * market_std
        # step et ancre arrondis avant construction de la grille (pas après) pour que
        # l'espacement reste exactement uniforme : sinon un arrondi indépendant par strike
        # introduit un résidu qui casse la parité call/put des flies/condors.
        strike_step = round(half_width / max(half, 1), 4)
        anchor = round(market.F0 / strike_step) * strike_step
        # Pas de round() final ici : un arrondi independant par strike, meme fin (1e-6),
        # recasse l'uniformite de l'espacement et redevient visible en erreur relative
        # sur les structures a prime quasi nulle (flies tres loin de la monnaie).
        strikes = [anchor + (i - half) * strike_step for i in range(2 * half + 1)]

        pricer = BachelierPricer(market)
        chain = pricer.build_chain(strikes)

        generator = StrategyUniverseGenerator(widths=self.widths)
        strategies = generator.generate(market, chain, strikes)

        scenario = self._random_scenario(market)
        sigma_N = float(self.rng.uniform(*self.sigma_N_range))
        allow_unlimited_loss = bool(self.rng.random() < self.unlimited_loss_allowed_prob)

        return SimulationResult(market, chain, strikes, strategies, scenario, sigma_N, allow_unlimited_loss)
