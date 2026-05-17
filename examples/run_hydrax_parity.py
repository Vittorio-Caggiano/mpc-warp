from __future__ import annotations

import argparse

from mpc_warp.envs.hydrax_parity import run_manifest_parity


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MPC parity checks against a local HydraX task manifest")
    parser.add_argument("manifest", help="Path to hydrax task manifest JSON")
    args = parser.parse_args()

    results = run_manifest_parity(args.manifest)
    failures = []
    for r in results:
        print(
            f"{r.name}: initial={r.initial_norm:.4f} final={r.final_norm:.4f} "
            f"target={r.target_norm:.4f} solved={r.solved}"
        )
        if not r.solved:
            failures.append(r)

    if failures:
        print("\nConvergence failures:")
        for r in failures:
            print(
                f"- {r.name}: convergence error={r.final_norm - r.target_norm:.6f} "
                f"(final {r.final_norm:.4f} > target {r.target_norm:.4f})"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
