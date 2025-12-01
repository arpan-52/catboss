#!/usr/bin/env python3
"""
CLI entry point for catboss
"""

import argparse
import sys
import json


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    # Main parser
    parser = argparse.ArgumentParser(
        description="CATBOSS - Radio Astronomy RFI Flagging Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  catboss --cat pooh data.ms --apply-flags
  catboss --cat pooh data.ms --config config.json
  catboss --cat pooh -h  (for POOH-specific help)
        """,
    )

    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )
    parser.add_argument(
        "--cat", type=str, choices=["pooh"], help="Select RFI flagging algorithm"
    )

    # Parse to check if --cat is specified
    args, remaining = parser.parse_known_args()

    # Handle version
    if args.version:
        print("catboss version 0.1.0")
        return 0

    # If --cat pooh is specified, create POOH-specific parser
    if args.cat == "pooh":
        pooh_parser = argparse.ArgumentParser(
            prog="catboss --cat pooh",
            description="POOH: Parallelized Optimized Outlier Hunter",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Algorithm Details:
  POOH is a multi-scale SumThreshold RFI flagger with bandpass normalization.
  It uses polynomial fitting to normalize the bandpass, then applies adaptive
  thresholding at multiple window sizes to detect RFI.

Examples:
  # Basic usage
  catboss --cat pooh data.ms --apply-flags

  # With all options
  catboss --cat pooh data.ms \\
    --combinations 1,2,4,8,16,32,64 \\
    --sigma 6.0 --rho 1.5 \\
    --poly-degree 5 --deviation-threshold 3.0 \\
    --polarizations 0,3 \\
    --apply-flags --diagnostic-plots \\
    --output-dir rfi_plots --verbose

  # Using config file
  catboss --cat pooh data.ms --config config.json
            """,
        )

        # Required arguments
        pooh_parser.add_argument("ms_path", help="Path to Measurement Set file")

        # Configuration file
        pooh_parser.add_argument(
            "--config",
            type=str,
            help="Path to JSON configuration file (overrides other options)",
        )

        # Multi-scale parameters
        pooh_parser.add_argument(
            "--combinations",
            type=str,
            default="1,2,4,8,16,32,64",
            help="Comma-separated window sizes for multi-scale detection (default: 1,2,4,8,16,32,64)",
        )
        pooh_parser.add_argument(
            "--sigma",
            type=float,
            default=6.0,
            help="Threshold multiplier (lower = more aggressive) (default: 6.0)",
        )
        pooh_parser.add_argument(
            "--rho",
            type=float,
            default=1.5,
            help="Reduction factor for larger windows (default: 1.5)",
        )

        # Bandpass normalization parameters
        pooh_parser.add_argument(
            "--poly-degree",
            type=int,
            default=5,
            help="Polynomial degree for bandpass fitting (3-9) (default: 5)",
        )
        pooh_parser.add_argument(
            "--deviation-threshold",
            type=float,
            default=3.0,
            help="RFI channel detection threshold in sigma (default: 3.0)",
        )

        # Correlation selection
        pooh_parser.add_argument(
            "--polarizations",
            type=str,
            default=None,
            help="Comma-separated correlation indices (default: 0 and last)",
        )

        # Output options
        pooh_parser.add_argument(
            "--apply-flags",
            action="store_true",
            help="Write flags to MS file (default: False)",
        )
        pooh_parser.add_argument(
            "--diagnostic-plots",
            action="store_true",
            help="Generate diagnostic plots (default: False)",
        )
        pooh_parser.add_argument(
            "--output-dir",
            type=str,
            default="outputs",
            help="Directory for diagnostic plots (default: outputs)",
        )

        # Performance options
        pooh_parser.add_argument(
            "--max-threads",
            type=int,
            default=16,
            help="Max threads for bandpass processing (default: 16)",
        )
        pooh_parser.add_argument(
            "--max-memory-usage",
            type=float,
            default=0.8,
            help="Fraction of memory to use (0.0-1.0) (default: 0.8)",
        )
        pooh_parser.add_argument(
            "--chunk-size",
            type=int,
            default=200000,
            help="I/O chunk size in rows for reading MS data (default: 200000, larger = faster I/O)",
        )

        # Verbosity
        pooh_parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output (default: False)",
        )

        # Parse POOH arguments
        pooh_args = pooh_parser.parse_args(remaining)

        # Load config if specified
        if pooh_args.config:
            print(f"Loading configuration from {pooh_args.config}")
            config_options = load_config(pooh_args.config)
        else:
            # Build options from command line arguments
            config_options = {
                "combinations": [
                    int(x.strip()) for x in pooh_args.combinations.split(",")
                ],
                "sigma_factor": pooh_args.sigma,
                "rho": pooh_args.rho,
                "poly_degree": pooh_args.poly_degree,
                "deviation_threshold": pooh_args.deviation_threshold,
                "corr_to_process": [
                    int(x.strip()) for x in pooh_args.polarizations.split(",")
                ]
                if pooh_args.polarizations
                else None,
                "apply_flags": pooh_args.apply_flags,
                "diagnostic_plots": pooh_args.diagnostic_plots,
                "output_dir": pooh_args.output_dir,
                "max_threads": pooh_args.max_threads,
                "max_memory_usage": pooh_args.max_memory_usage,
                "chunk_size": pooh_args.chunk_size,
                "verbose": pooh_args.verbose,
            }

        # Run POOH
        from catboss.pooh.pooh import print_gpu_info, hunt_ms

        print("=" * 70)
        print("POOH: Parallelized Optimized Outlier Hunter")
        print("Developed by Arpan Pal, NCRA-TIFR, 2025")
        print("=" * 70)

        print_gpu_info()

        print(f"\nProcessing: {pooh_args.ms_path}")
        print(f"Configuration:")
        print(f"  Window sizes: {config_options['combinations']}")
        print(f"  Sigma factor: {config_options['sigma_factor']}")
        print(f"  Rho: {config_options['rho']}")
        print(f"  Polynomial degree: {config_options['poly_degree']}")
        print(f"  Deviation threshold: {config_options['deviation_threshold']}")
        print(f"  Correlations: {config_options['corr_to_process']}")
        print(f"  Apply flags: {config_options['apply_flags']}")
        print(f"  Diagnostic plots: {config_options['diagnostic_plots']}")
        print("=" * 70)

        # Process
        results = hunt_ms(pooh_args.ms_path, config_options)

        # Summary
        print("\n" + "=" * 70)
        print("POOH Flagging Summary")
        print("=" * 70)
        print(f"Measurement Set: {pooh_args.ms_path}")
        print(f"Processing time: {results['total_processing_time']:.2f} seconds")
        print(f"Total flagged: {results['overall_percent_flagged']:.2f}%")
        print(f"New flags: {results['new_percent_flagged']:.2f}%")
        print(f"Baselines processed: {results['baselines_processed']}")
        print(f"Baselines skipped: {results['baselines_skipped']}")
        print("=" * 70)

        return 0

    # If no --cat specified, show help
    if not args.cat:
        parser.print_help()
        print("\nAvailable cats:")
        print("  pooh    Parallelized Optimized Outlier Hunter")
        print("\nUse 'catboss --cat pooh -h' for POOH-specific help")
        return 0


if __name__ == "__main__":
    sys.exit(main())
