"""
NAMI - Fast RFI flagging using C++ accelerated functions
Fits real and imaginary parts separately
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from casacore.tables import table
import argparse
import time
from multiprocessing import Pool, cpu_count

from nami.core_functions import (
    calculate_uv_distances_fast,
    collect_data_points_fast,
    flag_outliers_fast,
    fit_spline_fast
)


def parse_args():
    parser = argparse.ArgumentParser(description='NAMI - Fast RFI flagging')
    parser.add_argument('ms_file', help='Input MS')
    parser.add_argument('--datacolumn', default='DATA', choices=['DATA', 'CORRECTED_DATA', 'MODEL'])
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--timebin', type=float, default=30.0, help='Minutes')
    parser.add_argument('--nknots', type=int, default=-1)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_dir', default='nami_plots')
    parser.add_argument('--ncpu', type=int, default=0)
    parser.add_argument('--field', default='')
    parser.add_argument('--spw', default='')
    parser.add_argument('--corr', default='')
    parser.add_argument('--no-write', action='store_true', dest='no_write')
    parser.add_argument('--flag-all-corr', action='store_true', default=True, dest='flag_all_corr')
    parser.add_argument('--flag-selected-only', action='store_false', dest='flag_all_corr')
    return parser.parse_args()


def get_ms_info(ms_file):
    """Get MS info"""
    with table(ms_file) as tb:
        n_rows = tb.nrows()
        shape = tb.getcol('DATA', nrow=1).shape
    
    with table(f"{ms_file}::FIELD") as ft:
        fields = list(ft.getcol('NAME'))
    
    with table(f"{ms_file}::SPECTRAL_WINDOW") as st:
        spw_chans = [st.getcell('NUM_CHAN', i) for i in range(st.nrows())]
    
    return {'n_rows': n_rows, 'n_chan': shape[1], 'n_corr': shape[2], 
            'fields': fields, 'spw_chans': spw_chans}


def get_frequencies(ms_file, spws):
    """Get frequencies for SPWs"""
    with table(f"{ms_file}::SPECTRAL_WINDOW") as st:
        freqs = {}
        for spw in spws:
            if spw < st.nrows():
                freqs[spw] = st.getcol('CHAN_FREQ', startrow=spw, nrow=1)[0]
    return freqs


def read_chunk(ms_file, field_id, tstart, tend, datacolumn, spws):
    """Read one time chunk"""
    with table(ms_file) as tb:
        q = f"FIELD_ID=={field_id} AND TIME>={tstart} AND TIME<{tend}"
        if spws:
            q += f" AND DATA_DESC_ID IN [{','.join(map(str,spws))}]"
        
        with tb.query(q) as sub:
            if sub.nrows() == 0:
                return None
            return {
                'data': sub.getcol(datacolumn),
                'flags': sub.getcol('FLAG'),
                'uvw': sub.getcol('UVW'),
                'weight': sub.getcol('WEIGHT'),
                'ddids': sub.getcol('DATA_DESC_ID'),
                'rows': sub.rownumbers()
            }


def process_chunk(args):
    """Process one time chunk"""
    (ms_file, field_id, tstart, tend, datacolumn, spws, freqs,
     corrs, sigma, nknots, flag_all_corr, do_plot, plot_dir, chunk_idx) = args
    
    chunk = read_chunk(ms_file, field_id, tstart, tend, datacolumn, spws)
    if chunk is None:
        return {'empty': True, 'chunk_idx': chunk_idx, 'flags': []}
    
    data = chunk['data']
    flags = chunk['flags']
    uvw = chunk['uvw']
    weight = chunk['weight']
    ddids = chunk['ddids']
    rows = chunk['rows']
    
    n_rows, n_chan, n_corr = data.shape
    
    # Calculate UV distances
    c = 299792458.0
    uv_per_spw = {}
    for spw, f in freqs.items():
        uv_per_spw[spw] = calculate_uv_distances_fast(uvw, c / f)
    
    # Collect flag positions
    flag_list = []  # (ms_row, chan, corr or -1)
    plot_data = []
    
    for spw in np.unique(ddids):
        if spw not in uv_per_spw:
            continue
        
        mask = ddids == spw
        spw_rows = np.where(mask)[0]
        uv_dist = uv_per_spw[spw]
        
        outlier_map = {}  # (row, chan) -> set of corrs
        
        for corr in corrs:
            # Collect data (C++)
            collected = collect_data_points_fast(
                data, flags, uv_dist, weight,
                np.array(spw_rows, dtype=np.int32),
                np.array([corr], dtype=np.int32)
            )
            
            uv = collected['uv_dists']
            real = collected['real_values']
            imag = collected['imag_values']
            wts = collected['weights']
            pts = collected['point_indices']
            
            if len(uv) < 100:
                continue
            
            # Fit real (C++ spline)
            pred_real = fit_spline_fast(uv, real, wts, nknots)
            
            # Fit imag (C++ spline)
            pred_imag = fit_spline_fast(uv, imag, wts, nknots)
            
            # Outliers (C++)
            out_r, res_r, mad_r = flag_outliers_fast(uv, real, pred_real, sigma, wts)
            out_i, res_i, mad_i = flag_outliers_fast(uv, imag, pred_imag, sigma, wts)
            
            outliers = out_r | out_i
            n_out = np.sum(outliers)
            
            print(f"  F{field_id} C{chunk_idx} S{spw} corr{corr}: {n_out} "
                  f"(R:{np.sum(out_r)} I:{np.sum(out_i)})", flush=True)
            
            # Track positions
            for i, out in enumerate(outliers):
                if out:
                    r, ch = pts[i]
                    k = (int(r), int(ch))
                    if k not in outlier_map:
                        outlier_map[k] = set()
                    outlier_map[k].add(corr)
            
            # Plot data
            if do_plot:
                plot_data.append({
                    'corr': corr, 'spw': spw, 'uv': uv.copy(),
                    'real': real.copy(), 'imag': imag.copy(),
                    'pred_r': pred_real.copy(), 'pred_i': pred_imag.copy(),
                    'out': outliers.copy(), 'res_r': res_r.copy(), 'res_i': res_i.copy(),
                    'mad_r': mad_r, 'mad_i': mad_i
                })
        
        # Convert to flag list
        for (r, ch), corr_set in outlier_map.items():
            ms_row = rows[r]
            if flag_all_corr:
                flag_list.append((int(ms_row), int(ch), -1))
            else:
                for c in corr_set:
                    flag_list.append((int(ms_row), int(ch), int(c)))
    
    # Generate plots
    if do_plot and plot_data:
        os.makedirs(plot_dir, exist_ok=True)
        for p in plot_data:
            fname = os.path.join(plot_dir, f"f{field_id}_c{chunk_idx}_s{p['spw']}_corr{p['corr']}.png")
            make_plot(p, sigma, fname)
    
    return {'empty': False, 'chunk_idx': chunk_idx, 'flags': flag_list, 'n_out': len(flag_list)}


def make_plot(p, sigma, fname):
    """2x2 plot: real, imag, residuals"""
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    
    uv, out = p['uv'], p['out']
    idx = np.argsort(uv)
    
    # Real
    ax[0,0].scatter(uv[~out], p['real'][~out], s=1, alpha=0.3, c='b')
    ax[0,0].scatter(uv[out], p['real'][out], s=2, alpha=0.5, c='r')
    ax[0,0].plot(uv[idx], p['pred_r'][idx], 'k-', lw=1)
    ax[0,0].set_title(f"Corr {p['corr']} Real")
    ax[0,0].set_xlabel('UV'); ax[0,0].set_ylabel('Real')
    
    # Imag
    ax[0,1].scatter(uv[~out], p['imag'][~out], s=1, alpha=0.3, c='b')
    ax[0,1].scatter(uv[out], p['imag'][out], s=2, alpha=0.5, c='r')
    ax[0,1].plot(uv[idx], p['pred_i'][idx], 'k-', lw=1)
    ax[0,1].set_title(f"Corr {p['corr']} Imag")
    ax[0,1].set_xlabel('UV'); ax[0,1].set_ylabel('Imag')
    
    # Real residuals
    ax[1,0].scatter(uv[~out], p['res_r'][~out], s=1, alpha=0.3, c='b')
    ax[1,0].scatter(uv[out], p['res_r'][out], s=2, alpha=0.5, c='r')
    ax[1,0].axhline(0, c='k', lw=0.5)
    ax[1,0].axhline(sigma*p['mad_r'], c='orange', ls='--')
    ax[1,0].axhline(-sigma*p['mad_r'], c='orange', ls='--')
    ax[1,0].set_title(f"Real Res (MAD={p['mad_r']:.4f})")
    ax[1,0].set_xlabel('UV'); ax[1,0].set_ylabel('Residual')
    
    # Imag residuals
    ax[1,1].scatter(uv[~out], p['res_i'][~out], s=1, alpha=0.3, c='b')
    ax[1,1].scatter(uv[out], p['res_i'][out], s=2, alpha=0.5, c='r')
    ax[1,1].axhline(0, c='k', lw=0.5)
    ax[1,1].axhline(sigma*p['mad_i'], c='orange', ls='--')
    ax[1,1].axhline(-sigma*p['mad_i'], c='orange', ls='--')
    ax[1,1].set_title(f"Imag Res (MAD={p['mad_i']:.4f})")
    ax[1,1].set_xlabel('UV'); ax[1,1].set_ylabel('Residual')
    
    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()


def write_flags(ms_file, flags, n_corr):
    """Write flags"""
    if not flags:
        return 0
    
    lock = f"{ms_file}/table.lock"
    if os.path.exists(lock):
        try: os.remove(lock)
        except: pass
    
    # Group by row
    by_row = {}
    for r, ch, c in flags:
        if r not in by_row:
            by_row[r] = []
        by_row[r].append((ch, c))
    
    n = 0
    with table(ms_file, readonly=False) as tb:
        for row, items in by_row.items():
            cur = tb.getcol('FLAG', startrow=row, nrow=1)[0]
            for ch, c in items:
                if c == -1:
                    cur[ch, :] = True
                    n += n_corr
                else:
                    cur[ch, c] = True
                    n += 1
            tb.putcol('FLAG', cur.reshape(1, -1, n_corr), startrow=row, nrow=1)
    return n


def main():
    args = parse_args()
    
    field_sel = [int(x) for x in args.field.split(',')] if args.field else None
    spw_sel = [int(x) for x in args.spw.split(',')] if args.spw else None
    corr_sel = [int(x) for x in args.corr.split(',')] if args.corr else None
    
    ncpu = args.ncpu if args.ncpu > 0 else cpu_count()
    
    try:
        info = get_ms_info(args.ms_file)
        print(f"MS: {args.ms_file}")
        print(f"Rows: {info['n_rows']}, Chan: {info['n_chan']}, Corr: {info['n_corr']}")
        print(f"Fields: {list(enumerate(info['fields']))}")
        print(f"SPWs: {info['spw_chans']}")
        
        fields = field_sel if field_sel else list(range(len(info['fields'])))
        spws = spw_sel if spw_sel else list(range(len(info['spw_chans'])))
        corrs = corr_sel if corr_sel else list(range(info['n_corr']))
        
        print(f"\nFields: {fields}, SPWs: {spws}, Corrs: {corrs}")
        print(f"Sigma: {args.sigma}, CPUs: {ncpu}, Plot: {args.plot}")
        
        freqs = get_frequencies(args.ms_file, spws)
        total_flags = 0
        t0 = time.time()
        
        for fid in fields:
            print(f"\n{'='*50}\nField {fid}: {info['fields'][fid]}\n{'='*50}")
            
            # Get time range
            with table(args.ms_file) as tb:
                with tb.query(f"FIELD_ID=={fid}") as sub:
                    if sub.nrows() == 0:
                        print("  No data"); continue
                    times = sub.getcol('TIME')
            
            t_min, t_max = times.min(), times.max()
            chunk_sec = args.timebin * 60
            bounds = np.arange(t_min, t_max + chunk_sec, chunk_sec)
            n_chunks = len(bounds) - 1
            print(f"  {n_chunks} chunks, {t_max-t_min:.0f}s total")
            
            # Build tasks
            tasks = [(args.ms_file, fid, bounds[i], bounds[i+1], args.datacolumn,
                     spws, freqs, corrs, args.sigma, args.nknots, args.flag_all_corr,
                     args.plot, args.plot_dir, i) for i in range(n_chunks)]
            
            # Process
            if ncpu == 1:
                results = [process_chunk(t) for t in tasks]
            else:
                with Pool(ncpu) as pool:
                    results = pool.map(process_chunk, tasks)
            
            # Collect flags
            all_flags = []
            for r in results:
                if not r.get('empty', True):
                    all_flags.extend(r['flags'])
            
            print(f"\n  Field {fid}: {len(all_flags)} flags")
            
            # Write
            if not args.no_write and all_flags:
                print("  Writing...")
                n = write_flags(args.ms_file, all_flags, info['n_corr'])
                total_flags += n
                print(f"  Wrote {n}")
        
        print(f"\n{'='*50}\nDone in {time.time()-t0:.1f}s, {total_flags} flags\n{'='*50}")
        
    except KeyboardInterrupt:
        print("\nInterrupted")
        lock = f"{args.ms_file}/table.lock"
        if os.path.exists(lock): os.remove(lock)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()