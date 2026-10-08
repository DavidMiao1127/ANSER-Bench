"""Named baseline entry points with the common benchmark CLI."""
import sys


def baseline_main(method):
    import run_benchmark
    sys.argv[1:1]=['--method',method]
    raise SystemExit(run_benchmark.main())
