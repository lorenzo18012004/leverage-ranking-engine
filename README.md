# Leverage Ranking Engine

Moteur de classement de stratégies optionnelles par levier (Ω, Rachev, L, L^rob), pour futures STIR (SOFR, Euribor...). Le cadre théorique complet est détaillé dans [`exercice.pdf`](exercice.pdf).

## Lancer le projet

Prérequis : Python 3, `numpy`, `pandas`, `scipy`.

```bash
python leverage_engine.py
```

Chaque exécution simule un marché et un univers de stratégies aléatoires (pas de seed par défaut), calibre le moteur, classe les stratégies, et affiche un rapport complet dans le terminal.

## Architecture

Le projet a **2 fichiers Python**, avec des rôles bien séparés :

- **`mock_engine.py`** = génère des **données de test factices** (un marché fictif, une chaîne d'options fictive, un univers de stratégies, un scénario client fictif). Le nom "mock" veut dire "simulé/factice" — ce n'est pas une vraie source de données de marché, juste de quoi alimenter le moteur de classement pour le tester/démontrer.
- **`leverage_engine.py`** = le **vrai moteur** : calcule et classe les stratégies par levier (Ω, Rachev, L, L^rob), conformément à `exercice.pdf`.

---

## `mock_engine.py`

### `MarketContext`
Une boîte qui contient les paramètres du marché simulé : prix forward (`F0`), taux sans risque (`r`), maturité (`T`), volatilité (`sigma`), et si le contrat est sous appel de marge (`margined`). Sert à faire circuler ces 5 valeurs ensemble plutôt que séparément partout dans le code.

### `Leg`
Une **jambe d'option** : type (call/put), strike, quantité signée (positif = achetée, négatif = vendue). Sait calculer son propre payoff intrinsèque (`intrinsic()`) pour un tableau de prix finaux.

### `Strategy`
Une **stratégie complète** = un nom + un tuple de `Leg` (ses jambes) + ses primes calculées (`pi_q_total` net, `gross_pi_q_total` brut, `p_paid`). Sait calculer son payoff total (`payoff()`, somme des jambes) et son profil de risque (`loss_profile` : perte plafonnée ou illimitée, et dans quelle direction).

### `ScenarioP`
La **vue du client sur l'avenir** : une distribution discrète (prix, probabilité). C'est la mesure P du PDF (subjective), par opposition à Q (risque-neutre, utilisée pour le pricing).

### `discretize_gaussian_mixture()`
Fonction utilitaire : transforme un mélange de plusieurs courbes en cloche (les "narratifs" du client) en probabilités discrètes sur une grille de prix donnée.

### `SimulationResult`
Une boîte qui regroupe **tout ce que produit une simulation** (marché, chaîne, strikes, stratégies, scénario, `sigma_N`, contrainte de risque ouvert) — pour passer un seul objet plutôt que 7 variables séparées.

### `BachelierPricer`
Calcule le **prix théorique** d'une option sous le modèle de Bachelier (modèle normal, adapté aux taux STIR qui peuvent être négatifs) — par opposition à Black-Scholes (lognormal, adapté aux actions). `build_chain()` calcule ce prix pour tous les strikes × {call, put} d'un coup.

### `_Template`
Le "patron" abstrait d'une stratégie : un nom et une liste de `leg_specs` (décalage relatif, type, quantité) — pas encore de strike concret.

### `StrategyUniverseGenerator`
Contient les **19 templates** de structures classiques (naked, spreads, ladders, flies, condors, straddles/strangles, risk reversal). `generate()` fait glisser chaque template sur toute la chaîne de strikes, pour plusieurs largeurs, dédoublonne, et produit la liste complète des `Strategy` de l'univers.

### `MarketSimulator`
**L'orchestrateur.** `simulate()` tire un marché aléatoire, construit la chaîne de strikes autour du forward, calcule les prix, génère l'univers de stratégies, et fabrique un scénario client — puis renvoie tout ça dans un `SimulationResult`. Sans seed explicite, chaque exécution donne un contexte différent.

---

## `leverage_engine.py`

### `EngineConfig`
Les **hyperparamètres** du moteur : `lambda1`/`lambda2` (poids de calibration de `f`), `alpha`/`beta` (seuils de queue à 5%), `pi_min` (plancher anti-division-par-zéro), `epsilon` (poids du stress, 5%), `f_max` (plafond de `f` à 0.85), `min_gross_premium` (tick minimum de matérialité).

### `TailStatistics`
Boîte à outils statistique : `upper_tail_expectation`/`lower_tail_expectation` calculent l'espérance conditionnelle dans une queue de distribution (méthode Rockafellar-Uryasev, avec pondération fractionnaire au point frontière pour éviter les sauts). `moments()` calcule moyenne/écart-type/skewness/kurtosis.

### `ScenarioDiscretizer`
Rééchantillonne le scénario du client (qui a sa propre grille de prix) sur la grille de calcul interne du moteur, par interpolation de la fonction de répartition (CDF) — indépendant de la forme du scénario source.

### `LeverageEngine`
**Le cœur du moteur.** Pour une stratégie et un scénario donnés :
- `omega_ratio()` → Ω(0), le levier central (gain/perte moyens sous P)
- `rachev_ratio()` → le levier de queue (ETR/ES à 5%)
- `prudence_factor()` → calcule `f`, le poids qui arbitre entre Ω et Rachev selon la volatilité et la kurtosis du scénario
- `_score_on_grid()` → assemble tout ça en `L = Ω^f · Rachev^(1-f)`
- `robust_score()` → recalcule `L` sous deux scénarios de stress (baisse/hausse, ancrés sur les quantiles 1%/99% du marché) et prend le minimum → `L^rob`, le score final robuste
- `rank()` → calcule `L^rob` pour tout l'univers de stratégies, filtre celles interdites par le client (risque ouvert non autorisé, prime trop petite), et trie

### `AmericanPricer`
Calcule l'écart de prix entre une option **américaine** (exerçable avant l'échéance) et son équivalent **européen**, via un arbre binomial — pour évaluer le coût réel d'une éventuelle assignation anticipée sur les jambes vendues, quand le contrat n'est pas margé.

### `Top3Refiner`
Reprend les 3 meilleures stratégies du classement, leur ajoute la charge d'exercice anticipé calculée par `AmericanPricer`, et recalcule leur score — pour voir si l'ordre du podium change une fois cet effet pris en compte. Fait uniquement sur le Top 3 (pas tout l'univers) pour rester rapide.

### `Calibrator`
Trouve les meilleures valeurs de `lambda1`/`lambda2` par recherche sur grille : teste chaque combinaison sur une bibliothèque de 3 scénarios synthétiques (bimodal, à queues lourdes, uniforme), et note chaque combinaison sur 3 critères — la dispersion des scores (discrimination), la stabilité du Top-3 sous perturbation de prix, et la pénalité anti-risque-ouvert (favorise les combinaisons qui classent les stratégies bornées au-dessus des stratégies à risque illimité).

### `SimulationReport`
S'occupe uniquement de l'**affichage** : imprime la calibration, le marché simulé, le scénario client, la chaîne d'options, le classement (Top N), et le raffinement Top-3 — aucune logique de calcul, juste de la présentation en tableaux dans le terminal.

### Bloc `if __name__ == "__main__":`
Le point d'entrée quand tu lances `python leverage_engine.py` : simule un marché (`mock_engine.py`), calibre `(λ1,λ2)`, construit le moteur, classe toutes les stratégies, raffine le Top-3, et affiche le rapport complet.
