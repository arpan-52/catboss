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
    # Check for POOH-specific help first
    if "--cat" in sys.argv and "pooh" in sys.argv and ("-h" in sys.argv or "--help" in sys.argv):
        # Create POOH parser just for help
        help_parser = argparse.ArgumentParser(
            prog="catboss --cat pooh",
            description="POOH: Parallelized Optimized Outlier Hunter",
        )
        help_parser.add_argument("ms_path", help="Path to Measurement Set file")
        help_parser.add_argument("--combinations", default="1,2,4,8,16,32,64")
        help_parser.add_argument("--sigma", type=float, default=6.0)
        help_parser.add_argument("--rho", type=float, default=1.5)
        help_parser.add_argument("--poly-degree", type=int, default=5)
        help_parser.add_argument("--deviation-threshold", type=float, default=3.0)
        help_parser.add_argument("--polarizations", default=None)
        help_parser.add_argument("--apply-flags", action="store_true")
        help_parser.add_argument("--diagnostic-plots", action="store_true")
        help_parser.add_argument("--output-dir", default="outputs")
        help_parser.add_argument("--max-threads", type=int, default=16)
        help_parser.add_argument("--max-memory-usage", type=float, default=0.8)
        help_parser.add_argument("--chunk-size", type=int, default=200000)
        help_parser.add_argument("--verbose", action="store_true")
        help_parser.print_help()
        return 0

    # Main parser (add_help=False to let sub-parsers handle help)
    parser = argparse.ArgumentParser(
        description="CATBOSS - Radio Astronomy RFI Flagging Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # Let sub-parsers handle help
        epilog="""
Examples:
  catboss -h                          (show this help)
  catboss --cat pooh data.ms --apply-flags
  catboss --cat pooh data.ms --config config.json
  catboss --cat pooh -h               (show POOH-specific help)
        """,
    )

    # Add manual help option
    parser.add_argument("-h", "--help", action="store_true", help="Show help message")

    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )
    parser.add_argument(
        "--cat", type=str, choices=["pooh"], help="Select RFI flagging algorithm"
    )

    # Parse to check if --cat is specified
    args, remaining = parser.parse_known_args()

    # Handle help for main parser
    if args.help and not args.cat:
        parser.print_help()
        return 0

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
        pooh_parser.add_argument(
            "ms_path", nargs="?", help="Path to Measurement Set file"
        )

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

        # If -h or --help in remaining, show POOH help and exit
        if "-h" in remaining or "--help" in remaining:
            pooh_parser.print_help()
            return 0

        # Parse POOH arguments
        pooh_args = pooh_parser.parse_args(remaining)

        # Check if ms_path was provided
        if not pooh_args.ms_path:
            pooh_parser.error("the following arguments are required: ms_path")

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
        from catboss.pooh.pooh import hunt_ms
        from catboss.logger import setup_logger, print_banner, print_cat_on_hunt

        # Print banner
        print_banner()

        # Setup logger
        logger = setup_logger("catboss", verbose=pooh_args.verbose)

        # Show which cat is hunting
        print_cat_on_hunt("POOH")

        # Log configuration
        logger.info(f"Processing Measurement Set: {pooh_args.ms_path}")
        logger.info("=" * 70)
        logger.info("Configuration:")
        logger.info(f"  Window sizes: {config_options['combinations']}")
        logger.info(f"  Sigma factor: {config_options['sigma_factor']}")
        logger.info(f"  Rho: {config_options['rho']}")
        logger.info(f"  Polynomial degree: {config_options['poly_degree']}")
        logger.info(f"  Deviation threshold: {config_options['deviation_threshold']}")
        logger.info(f"  Correlations: {config_options['corr_to_process']}")
        logger.info(f"  Apply flags: {config_options['apply_flags']}")
        logger.info(f"  Diagnostic plots: {config_options['diagnostic_plots']}")
        logger.info(f"  Chunk size: {config_options['chunk_size']} rows")
        logger.info("=" * 70)

        # Pass logger to options
        config_options["logger"] = logger

        # Process
        results = hunt_ms(pooh_args.ms_path, config_options)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("🎯 POOH Flagging Summary")
        logger.info("=" * 70)
        logger.info(f"Measurement Set: {pooh_args.ms_path}")
        logger.info(
            f"⏱️  Processing time: {results['total_processing_time']:.2f} seconds"
        )
        logger.info(f"📊 Total flagged: {results['overall_percent_flagged']:.2f}%")
        logger.info(f"🆕 New flags: {results['new_percent_flagged']:.2f}%")
        logger.info(f"✅ Baselines processed: {results['baselines_processed']}")
        logger.info(f"⏭️  Baselines skipped: {results['baselines_skipped']}")
        logger.info("=" * 70)

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
