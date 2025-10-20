#!/usr/bin/env python3
"""
CLI entry point for catboss
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="CATBOSS: A suite of radio astronomy flagging tools"
    )
    
    # Add general arguments
    parser.add_argument("--version", action="store_true", help="Show version information")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--cat", type=str, choices=["pooh"], 
                        help="Directly select which cat tool to use (alternative to subcommands)")
    
    # Create subparsers for each cat
    subparsers = parser.add_subparsers(dest="subcommand", help="Select which cat to use")
    
    # POOH subparser
    pooh_parser = subparsers.add_parser("pooh", help="Parallelized Optimized Outlier Hunter")
    pooh_parser.add_argument('ms_path', help='Path to the MS file')
    pooh_parser.add_argument('--combinations', type=str, default='1,2,4,8,16,32,64',
                           help='Comma-separated list of window sizes')
    pooh_parser.add_argument('--sigma', type=float, default=6.0,
                           help='Sigma factor for threshold calculation')
    pooh_parser.add_argument('--rho', type=float, default=1.5,
                           help='Factor to reduce threshold for larger window sizes')
    pooh_parser.add_argument('--polarizations', type=str, default=None,
                           help='Comma-separated list of polarization indices to process (default: all)')
    pooh_parser.add_argument('--apply-flags', action='store_true',
                           help='Apply flags to the MS file')
    pooh_parser.add_argument('--diagnostic-plots', action='store_true',
                           help='Generate diagnostic plots')
    pooh_parser.add_argument('--output-dir', type=str, default='outputs',
                           help='Directory to save diagnostic plots (default: outputs)')
    pooh_parser.add_argument('--max-memory-usage', type=float, default=0.8,
                           help='Maximum fraction of available memory to use (default: 0.8)')
    pooh_parser.add_argument('--verbose', action='store_true',
                           help='Enable verbose output')
    
    args, unknown_args = parser.parse_known_args()
    
    # Handle version request
    if args.version:
        from catboss import __version__
        print(f"catboss version {__version__}")
        return 0
    
    # Handle direct cat selection via --cat option
    selected_cat = None
    
    if args.cat:
        selected_cat = args.cat
        # If cat is selected directly via --cat, parse remaining args according to that cat's parser
        if selected_cat == "pooh":
            pooh_only_parser = argparse.ArgumentParser(description="POOH: Parallelized Optimized Outlier Hunter")
            pooh_only_parser.add_argument('ms_path', help='Path to the MS file')
            pooh_only_parser.add_argument('--combinations', type=str, default='1,2,4,8,16',
                                help='Comma-separated list of window sizes')
            pooh_only_parser.add_argument('--sigma', type=float, default=10.0,
                                help='Sigma factor for threshold calculation')
            pooh_only_parser.add_argument('--rho', type=float, default=1.5,
                                help='Factor to reduce threshold for larger window sizes')
            pooh_only_parser.add_argument('--polarizations', type=str, default=None,
                                help='Comma-separated list of polarization indices to process (default: all)')
            pooh_only_parser.add_argument('--apply-flags', action='store_true',
                                help='Apply flags to the MS file')
            pooh_only_parser.add_argument('--diagnostic-plots', action='store_true',
                                help='Generate diagnostic plots')
            pooh_only_parser.add_argument('--output-dir', type=str, default='outputs',
                                help='Directory to save diagnostic plots (default: outputs)')
            pooh_only_parser.add_argument('--max-memory-usage', type=float, default=0.8,
                                help='Maximum fraction of available memory to use (default: 0.8)')
            pooh_only_parser.add_argument('--verbose', action='store_true',
                                help='Enable verbose output')
            
            pooh_args = pooh_only_parser.parse_args(unknown_args)
            args.ms_path = pooh_args.ms_path
            args.combinations = pooh_args.combinations
            args.sigma = pooh_args.sigma
            args.rho = pooh_args.rho
            args.polarizations = pooh_args.polarizations
            args.apply_flags = pooh_args.apply_flags
            args.diagnostic_plots = pooh_args.diagnostic_plots
            args.output_dir = pooh_args.output_dir
            args.max_memory_usage = pooh_args.max_memory_usage
            args.verbose = pooh_args.verbose
    else:
        # Check if a subcommand was used
        selected_cat = args.subcommand
    
    # If no cat selected, show help
    if not selected_cat:
        parser.print_help()
        return 0
    
    
    # Dispatch to the selected cat
    if selected_cat == "pooh":
        from catboss.pooh.pooh import print_gpu_info, hunt_ms
        
        # Print GPU information
        print_gpu_info()
        
        # Convert args to options dict expected by hunt_ms
        options = {
            'combinations': [int(x) for x in args.combinations.split(',')],
            'sigma_factor': args.sigma,
            'rho': args.rho,
            'corr_to_process': [int(x) for x in args.polarizations.split(',')] if args.polarizations else None,
            'apply_flags': args.apply_flags,
            'diagnostic_plots': args.diagnostic_plots,
            'output_dir': args.output_dir,
            'max_memory_usage': args.max_memory_usage,
            'verbose': args.verbose
        }
        
        print(" Parallelized Optimized Outlier Hunter: POOH, Developed by Arpan Pal at NCRA-TIFR in 2025")
        print("Pooh is my first cat. He specializes in collecting snakes and birds, which he thoughtfully deposits on my bed. I write this code in return of those gifts.")
        
        # Process the MS
        results = hunt_ms(args.ms_path, options)
        
        print("\nPOOH Flagging Summary:")
        print(f"Measurement Set: {args.ms_path}")
        print(f"Total processing time: {results['total_processing_time']:.2f} seconds")
        print(f"Overall flagging percentage: {results['overall_percent_flagged']:.2f}%")
        print("Meow! Flagging complete.")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
