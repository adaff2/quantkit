# Changelog

All notable changes to this project will be documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-03-30

### Added
- Stochastic processes: `StandardBM`, `BrownianMotion`, `GBM`, `RandomWalk`,
  `OrnsteinUhlenbeck`, `PoissonProcess`, `CompoundPoissonProcess`, `LevyFlight`,
  `FractionalBrownianMotion`, `StableProcess`, `HestonModel`, `CEV`,
  `MertonJumpDiffusion`, `VarianceGamma`, `ThreeTwoModel`
- `GBM.compute_option_price` for European and Asian barrier options via Monte Carlo
- `RandomEvents` with `dice_roll` and `coin_flip` helpers
- Real-time SDL2 visualiser: `Renderer` and `PathPlot` with zoom/pan and crosshair
- Numba-accelerated simulation paths for `StandardBM`, `BrownianMotion`, and `GBM`
