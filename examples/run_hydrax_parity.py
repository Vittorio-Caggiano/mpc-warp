from __future__ import annotations

import argparse
import json
import tempfile

from mpc_warp.envs.hydrax_parity import discover_hydrax_tasks, run_manifest_parity


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MPC parity checks against HydraX task manifests")
    parser.add_argument("manifest", nargs="?", help="Path to hydrax task manifest JSON")
    parser.add_argument("--hydrax-root", help="Path to local hydrax checkout")
    parser.add_argument("--fail-on-unsupported", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest
    temp_path = None
    if args.hydrax_root:
        tasks = discover_hydrax_tasks(args.hydrax_root)
        fd = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(tasks, fd)
        fd.close()
        manifest_path = fd.name
        temp_path = manifest_path

    if not manifest_path:
        raise SystemExit("Provide either a manifest path or --hydrax-root")

    results = run_manifest_parity(manifest_path)
    counts = {"solved": 0, "mismatch": 0, "unsupported": 0, "error": 0}
    failures = []
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        print(
            f"{r.name}: status={r.status} initial={r.initial_norm} final={r.final_norm} "
            f"target={r.target_norm} msg={r.message}"
        )
        if r.status in {"mismatch", "error"} or (r.status == "solved" and not r.solved):
            failures.append(r)

    print(
        f"summary: solved={counts['solved']} mismatch={counts['mismatch']} "
        f"unsupported={counts['unsupported']} error={counts['error']}"
    )

    if temp_path:
        print(f"generated_manifest={temp_path}")

    if failures:
        print("\nConvergence failures:")
        for r in failures:
            if r.final_norm is not None and r.target_norm is not None:
                print(
                    f"- {r.name}: convergence error={r.final_norm - r.target_norm:.6f} "
                    f"(final {r.final_norm:.4f} > target {r.target_norm:.4f})"
                )
            else:
                print(f"- {r.name}: {r.message or r.status}")
        return 1

    if args.fail_on_unsupported and counts["unsupported"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
