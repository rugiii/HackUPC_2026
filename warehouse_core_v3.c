/*
 * warehouse_core_v3.c  –  Hot-path geometry compiled with -O2 + OpenMP
 *
 * CAMBIOS vs v2:
 *   - full_sweep_topk   → paralelo por combos (omp parallel for + merge thread-local)
 *   - full_sweep        → paralelo con reducción de best_f
 *   - precompute_bitmap → omp parallel for collapse(2)
 *   - batch_validate    → omp parallel for (cada índice es independiente)
 *   - Todos los helpers de geometría son read-only → thread-safe sin cambios
 *
 * COMPILAR:
 *   gcc -O2 -march=native -ffast-math -fopenmp \
 *       -shared -fPIC -o warehouse_core_v3.so warehouse_core_v3.c -lm
 *
 * THREADS: controla con OMP_NUM_THREADS=N antes de lanzar Python.
 *          Por defecto usa todos los cores lógicos disponibles.
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/* ── point_in_polygon (ray casting) ─────────────────────────────────────── */
/* Completamente read-only → thread-safe */
int point_in_polygon(double px, double py, const double *verts, int n) {
    int inside = 0;
    int j = n - 1;
    for (int i = 0; i < n; i++) {
        double xi = verts[2*i], yi = verts[2*i+1];
        double xj = verts[2*j], yj = verts[2*j+1];
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi))
            inside = !inside;
        j = i;
    }
    return inside;
}

static int point_on_boundary(double px, double py, const double *verts, int n) {
    const double EPS = 1e-6;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        double x1=verts[2*i], y1=verts[2*i+1];
        double x2=verts[2*j], y2=verts[2*j+1];
        double cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1);
        if (fabs(cross) < EPS) {
            double xlo = x1<x2?x1:x2, xhi = x1>x2?x1:x2;
            double ylo = y1<y2?y1:y2, yhi = y1>y2?y1:y2;
            if (px >= xlo-EPS && px <= xhi+EPS && py >= ylo-EPS && py <= yhi+EPS)
                return 1;
        }
    }
    return 0;
}

static int point_inside_or_boundary(double px, double py, const double *verts, int n) {
    return point_on_boundary(px,py,verts,n) || point_in_polygon(px,py,verts,n);
}

/* ── Precompute occupancy bitmap ─────────────────────────────────────────── */
/* PARALELO: cada celda (ix,iy) es independiente → collapse(2) trivial      */
void precompute_bitmap(
    const double *verts, int n_verts,
    int xmin, int ymin, int nx, int ny, int step,
    uint8_t *out_bitmap)
{
    double half = step * 0.5;
    #pragma omp parallel for collapse(2) schedule(dynamic, 32)
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            double cy = ymin + iy*step + half;
            double cx = xmin + ix*step + half;
            out_bitmap[iy*nx + ix] = (uint8_t)point_in_polygon(cx, cy, verts, n_verts);
        }
    }
}

/* ── rect_in_polygon via corners + concave-vertex check ─────────────────── */
/* Read-only → thread-safe sin cambios */
int rect_in_polygon(
    double rx0, double ry0, double rx1, double ry1,
    const double *verts, int n_verts)
{
    double corners[4][2] = {{rx0,ry0},{rx1,ry0},{rx1,ry1},{rx0,ry1}};
    for (int i = 0; i < 4; i++) {
        if (!point_inside_or_boundary(corners[i][0], corners[i][1], verts, n_verts))
            return 0;
    }
    for (int i = 0; i < n_verts; i++) {
        double vx = verts[2*i], vy = verts[2*i+1];
        if (vx > rx0 && vx < rx1 && vy > ry0 && vy < ry1) return 0;
    }
    return 1;
}

/* ── rect_overlaps_any ──────────────────────────────────────────────────── */
/* Read-only → thread-safe sin cambios */
int rect_overlaps_any(
    double ax0, double ay0, double ax1, double ay1,
    const double *rects, int n_rects)
{
    for (int i = 0; i < n_rects; i++) {
        double bx0=rects[4*i], by0=rects[4*i+1], bx1=rects[4*i+2], by1=rects[4*i+3];
        if (ax0 < bx1 && ax1 > bx0 && ay0 < by1 && ay1 > by0) return 1;
    }
    return 0;
}

/* ── ceiling_at ─────────────────────────────────────────────────────────── */
/* Read-only → thread-safe sin cambios */
double ceiling_at(double xmin, double xmax, const double *segs, int n_segs) {
    double min_h = 1e18;
    for (int i = 0; i < n_segs; i++) {
        double seg_start = segs[2*i];
        double seg_end   = (i+1 < n_segs) ? segs[2*(i+1)] : 1e18;
        double seg_h     = segs[2*i+1];
        if (seg_end <= xmin) continue;
        if (seg_start >= xmax) break;
        if (seg_h < min_h) min_h = seg_h;
    }
    return min_h;
}

/* ── batch_validate ──────────────────────────────────────────────────────── */
/* PARALELO: cada candidato es independiente → parallel for directo          */
int batch_validate(
    const double *bay_params,
    const double *candidates,
    int n_cands,
    const double *verts, int n_verts,
    const double *obstacles, int n_obs,
    const double *ceiling_segs, int n_ceil,
    const double *placed_rects, int n_placed,
    const double *placed_gaps,  int n_pgaps,
    int *out_valid)
{
    double W = bay_params[0], D = bay_params[1];
    double H = bay_params[2], G = bay_params[3];
    double needed_h = H + G;
    int count = 0;

    #pragma omp parallel for schedule(dynamic, 64) reduction(+:count)
    for (int ci = 0; ci < n_cands; ci++) {
        double px = candidates[3*ci];
        double py = candidates[3*ci+1];
        int rot   = (int)candidates[3*ci+2];

        double bx0,by0,bx1,by1, gx0,gy0,gx1,gy1;

        if (rot == 0) {
            bx0=px;    by0=py;    bx1=px+W;  by1=py+D;
            gx0=px;    gy0=py+D;  gx1=px+W;  gy1=py+D+G;
        } else if (rot == 90) {
            bx0=px;    by0=py;    bx1=px+D;  by1=py+W;
            gx0=px-G;  gy0=py;    gx1=px;    gy1=py+W;
        } else if (rot == 180) {
            bx0=px;    by0=py;    bx1=px+W;  by1=py+D;
            gx0=px;    gy0=py-G;  gx1=px+W;  gy1=py;
        } else {
            bx0=px;    by0=py;    bx1=px+D;  by1=py+W;
            gx0=px+D;  gy0=py;    gx1=px+D+G; gy1=py+W;
        }

        int valid = 1;
        if (!rect_in_polygon(bx0,by0,bx1,by1, verts,n_verts))           valid=0;
        else if (!rect_in_polygon(gx0,gy0,gx1,gy1, verts,n_verts))      valid=0;
        else if (ceiling_at(bx0,bx1,ceiling_segs,n_ceil) < needed_h)    valid=0;
        else if (rect_overlaps_any(bx0,by0,bx1,by1, obstacles,n_obs))   valid=0;
        else if (rect_overlaps_any(gx0,gy0,gx1,gy1, obstacles,n_obs))   valid=0;
        else if (rect_overlaps_any(bx0,by0,bx1,by1, placed_rects,n_placed)) valid=0;
        else if (rect_overlaps_any(gx0,gy0,gx1,gy1, placed_rects,n_placed)) valid=0;
        else if (rect_overlaps_any(bx0,by0,bx1,by1, placed_gaps,n_pgaps))   valid=0;

        out_valid[ci] = valid;
        if (valid) count++;
    }
    return count;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * full_sweep  –  PARALELO: cada combo calcula su f_val independientemente.
 *   Usamos reducción manual con critical section para el best global.
 * ──────────────────────────────────────────────────────────────────────────── */
int full_sweep(
    const double *combos,   int n_combos,
    const double *cands,    int n_cands,
    const double *verts,    int n_verts,
    const double *obstacles,int n_obs,
    const double *ceil_segs,int n_ceil,
    const double *p_rects,  int n_prects,
    const double *p_gaps,   int n_pgaps,
    double sum_p, double sum_l, double occ_area, double total_area,
    int *out_combo, int *out_cand
)
{
    double best_f = 1e18;
    int best_combo = -1, best_cand = -1;

    /*
     * Paralelizamos el loop exterior (combos).
     * Cada thread mantiene sus propias variables locales best_f/combo/cand
     * y al final hacemos una reducción con critical section.
     * El loop interior (cands) permanece secuencial dentro de cada thread
     * porque hace un early-break en cuanto encuentra el primer válido.
     */
    #pragma omp parallel
    {
        double   t_best_f     = 1e18;
        int      t_best_combo = -1;
        int      t_best_cand  = -1;

        #pragma omp for schedule(dynamic, 4) nowait
        for (int ci = 0; ci < n_combos; ci++) {
            double W  = combos[7*ci+0];
            double D  = combos[7*ci+1];
            double H  = combos[7*ci+2];
            double G  = combos[7*ci+3];
            double pr = combos[7*ci+4];
            double ld = combos[7*ci+5];
            int   rot = (int)combos[7*ci+6];

            double needed_h = H + G;
            double ew = (rot==0||rot==180) ? W : D;
            double ed = (rot==0||rot==180) ? D : W;
            double new_occ = occ_area + ew*ed;
            double new_sp  = sum_p + pr;
            double new_sl  = sum_l + ld;
            if (new_sl <= 0 || new_sp <= 0) continue;
            double f_val = pow(new_sp/new_sl, 2.0 - new_occ/total_area);

            if (f_val >= t_best_f) continue;

            for (int pi = 0; pi < n_cands; pi++) {
                double px = cands[2*pi];
                double py = cands[2*pi+1];

                double bx0,by0,bx1,by1, gx0,gy0,gx1,gy1;
                if (rot == 0) {
                    bx0=px; by0=py; bx1=px+W; by1=py+D;
                    gx0=px; gy0=py+D; gx1=px+W; gy1=py+D+G;
                } else if (rot == 90) {
                    bx0=px; by0=py; bx1=px+D; by1=py+W;
                    gx0=px-G; gy0=py; gx1=px; gy1=py+W;
                } else if (rot == 180) {
                    bx0=px; by0=py; bx1=px+W; by1=py+D;
                    gx0=px; gy0=py-G; gx1=px+W; gy1=py;
                } else {
                    bx0=px; by0=py; bx1=px+D; by1=py+W;
                    gx0=px+D; gy0=py; gx1=px+D+G; gy1=py+W;
                }

                if (!rect_in_polygon(bx0,by0,bx1,by1,verts,n_verts)) continue;
                if (!rect_in_polygon(gx0,gy0,gx1,gy1,verts,n_verts)) continue;
                if (ceiling_at(bx0,bx1,ceil_segs,n_ceil) < needed_h) continue;
                if (rect_overlaps_any(bx0,by0,bx1,by1,obstacles,n_obs)) continue;
                if (rect_overlaps_any(gx0,gy0,gx1,gy1,obstacles,n_obs)) continue;
                if (rect_overlaps_any(bx0,by0,bx1,by1,p_rects,n_prects)) continue;
                if (rect_overlaps_any(gx0,gy0,gx1,gy1,p_rects,n_prects)) continue;
                if (rect_overlaps_any(bx0,by0,bx1,by1,p_gaps,n_pgaps)) continue;

                t_best_f     = f_val;
                t_best_combo = ci;
                t_best_cand  = pi;
                break;
            }
        }

        /* Reducción: un thread a la vez actualiza el mejor global */
        #pragma omp critical
        {
            if (t_best_combo >= 0 && t_best_f < best_f) {
                best_f     = t_best_f;
                best_combo = t_best_combo;
                best_cand  = t_best_cand;
            }
        }
    }

    *out_combo = best_combo;
    *out_cand  = best_cand;
    return (best_combo >= 0) ? 1 : 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * full_sweep_topk  –  PARALELO: top-K distinct placements
 *
 * Estrategia:
 *   1. Cada thread trabaja sobre un subconjunto de combos.
 *   2. Cada thread construye su propio buffer local de resultados (_SweepResult[]).
 *   3. Al final, los buffers se fusionan y ordenan para extraer el top-K global.
 *
 * Esto evita cualquier lock durante el loop caliente.
 * ──────────────────────────────────────────────────────────────────────────── */

typedef struct { double f; int combo; int cand; } _SweepResult;

static int _cmp_sweep(const void *a, const void *b) {
    double fa = ((_SweepResult*)a)->f;
    double fb = ((_SweepResult*)b)->f;
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

int full_sweep_topk(
    const double *combos,   int n_combos,
    const double *cands,    int n_cands,
    const double *verts,    int n_verts,
    const double *obstacles,int n_obs,
    const double *ceil_segs,int n_ceil,
    const double *p_rects,  int n_prects,
    const double *p_gaps,   int n_pgaps,
    double sum_p, double sum_l, double occ_area, double total_area,
    int topk,
    int *out_results
)
{
    /*
     * Reservamos un buffer global donde cada thread escribe en su propia franja.
     * Franja del thread t: [t*n_combos .. (t+1)*n_combos)
     * Así evitamos completamente sincronización en el loop caliente.
     */
    int n_threads;
    #pragma omp parallel
    #pragma omp single
    n_threads = omp_get_num_threads();

    /* Buffer: worst-case cada thread encuentra n_combos resultados */
    _SweepResult *buf = (_SweepResult*)malloc(
        (size_t)n_threads * n_combos * sizeof(_SweepResult));
    if (!buf) return 0;

    /* found[t] = cuántos resultados encontró el thread t */
    int *found = (int*)calloc(n_threads, sizeof(int));
    if (!found) { free(buf); return 0; }

    #pragma omp parallel
    {
        int tid      = omp_get_thread_num();
        int t_found  = 0;
        /* Puntero al inicio de la franja de este thread */
        _SweepResult *my_buf = buf + tid * n_combos;

        #pragma omp for schedule(dynamic, 4) nowait
        for (int ci = 0; ci < n_combos; ci++) {
            double W  = combos[7*ci+0];
            double D  = combos[7*ci+1];
            double H  = combos[7*ci+2];
            double G  = combos[7*ci+3];
            double pr = combos[7*ci+4];
            double ld = combos[7*ci+5];
            int   rot = (int)combos[7*ci+6];

            double needed_h = H + G;
            double ew = (rot==0||rot==180) ? W : D;
            double ed = (rot==0||rot==180) ? D : W;
            double new_occ = occ_area + ew*ed;
            double new_sp  = sum_p + pr;
            double new_sl  = sum_l + ld;
            if (new_sl <= 0 || new_sp <= 0) continue;
            double f_val = pow(new_sp/new_sl, 2.0 - new_occ/total_area);

            /* scan candidates para la primera posición válida de este combo */
            for (int pi = 0; pi < n_cands; pi++) {
                double px = cands[2*pi];
                double py = cands[2*pi+1];

                double bx0,by0,bx1,by1, gx0,gy0,gx1,gy1;
                if (rot == 0) {
                    bx0=px; by0=py; bx1=px+W; by1=py+D;
                    gx0=px; gy0=py+D; gx1=px+W; gy1=py+D+G;
                } else if (rot == 90) {
                    bx0=px; by0=py; bx1=px+D; by1=py+W;
                    gx0=px-G; gy0=py; gx1=px; gy1=py+W;
                } else if (rot == 180) {
                    bx0=px; by0=py; bx1=px+W; by1=py+D;
                    gx0=px; gy0=py-G; gx1=px+W; gy1=py;
                } else {
                    bx0=px; by0=py; bx1=px+D; by1=py+W;
                    gx0=px+D; gy0=py; gx1=px+D+G; gy1=py+W;
                }

                if (!rect_in_polygon(bx0,by0,bx1,by1,verts,n_verts)) continue;
                if (!rect_in_polygon(gx0,gy0,gx1,gy1,verts,n_verts)) continue;
                if (ceiling_at(bx0,bx1,ceil_segs,n_ceil) < needed_h) continue;
                if (rect_overlaps_any(bx0,by0,bx1,by1,obstacles,n_obs)) continue;
                if (rect_overlaps_any(gx0,gy0,gx1,gy1,obstacles,n_obs)) continue;
                if (rect_overlaps_any(bx0,by0,bx1,by1,p_rects,n_prects)) continue;
                if (rect_overlaps_any(gx0,gy0,gx1,gy1,p_rects,n_prects)) continue;
                if (rect_overlaps_any(bx0,by0,bx1,by1,p_gaps,n_pgaps)) continue;

                my_buf[t_found].f     = f_val;
                my_buf[t_found].combo = ci;
                my_buf[t_found].cand  = pi;
                t_found++;
                break;  /* mejor posición para este combo */
            }
        }
        found[tid] = t_found;
    }

    /* ── Merge de buffers de todos los threads ── */
    int total_found = 0;
    for (int t = 0; t < n_threads; t++) total_found += found[t];

    /* Compactar en buf[0..total_found) */
    int write = 0;
    for (int t = 0; t < n_threads; t++) {
        _SweepResult *src = buf + t * n_combos;
        if (write != t * n_combos) {
            memmove(buf + write, src, found[t] * sizeof(_SweepResult));
        }
        write += found[t];
    }

    /* Ordenar por f ascendente */
    qsort(buf, total_found, sizeof(_SweepResult), _cmp_sweep);

    int ret = total_found < topk ? total_found : topk;
    for (int i = 0; i < ret; i++) {
        out_results[2*i]   = buf[i].combo;
        out_results[2*i+1] = buf[i].cand;
    }

    free(found);
    free(buf);
    return ret;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * full_sweep_topk_multi  –  like full_sweep_topk but returns up to max_per_combo
 * valid positions per combo. Gives beam search more diverse candidates when
 * candidates are ordered btb/edge-first (so multiple positions per combo are
 * from different spatial locations).
 * ──────────────────────────────────────────────────────────────────────────── */
int full_sweep_topk_multi(
    const double *combos,   int n_combos,
    const double *cands,    int n_cands,
    const double *verts,    int n_verts,
    const double *obstacles,int n_obs,
    const double *ceil_segs,int n_ceil,
    const double *p_rects,  int n_prects,
    const double *p_gaps,   int n_pgaps,
    double sum_p, double sum_l, double occ_area, double total_area,
    int topk, int max_per_combo,
    int *out_results
)
{
    int n_threads;
    #pragma omp parallel
    #pragma omp single
    n_threads = omp_get_num_threads();

    int slots_per_thread = n_combos * max_per_combo;
    _SweepResult *buf = (_SweepResult*)malloc(
        (size_t)n_threads * slots_per_thread * sizeof(_SweepResult));
    if (!buf) return 0;
    int *found = (int*)calloc(n_threads, sizeof(int));
    if (!found) { free(buf); return 0; }

    #pragma omp parallel
    {
        int tid     = omp_get_thread_num();
        int t_found = 0;
        _SweepResult *my_buf = buf + tid * slots_per_thread;

        #pragma omp for schedule(dynamic, 4) nowait
        for (int ci = 0; ci < n_combos; ci++) {
            double W  = combos[7*ci+0];
            double D  = combos[7*ci+1];
            double H  = combos[7*ci+2];
            double G  = combos[7*ci+3];
            double pr = combos[7*ci+4];
            double ld = combos[7*ci+5];
            int   rot = (int)combos[7*ci+6];

            double needed_h = H + G;
            double ew = (rot==0||rot==180) ? W : D;
            double ed = (rot==0||rot==180) ? D : W;
            double new_occ = occ_area + ew*ed;
            double new_sp  = sum_p + pr;
            double new_sl  = sum_l + ld;
            if (new_sl <= 0 || new_sp <= 0) continue;
            double f_val = pow(new_sp/new_sl, 2.0 - new_occ/total_area);

            int n_for_combo = 0;
            for (int pi = 0; pi < n_cands && n_for_combo < max_per_combo; pi++) {
                double px = cands[2*pi];
                double py = cands[2*pi+1];

                double bx0,by0,bx1,by1, gx0,gy0,gx1,gy1;
                if (rot == 0) {
                    bx0=px; by0=py; bx1=px+W; by1=py+D;
                    gx0=px; gy0=py+D; gx1=px+W; gy1=py+D+G;
                } else if (rot == 90) {
                    bx0=px; by0=py; bx1=px+D; by1=py+W;
                    gx0=px-G; gy0=py; gx1=px; gy1=py+W;
                } else if (rot == 180) {
                    bx0=px; by0=py; bx1=px+W; by1=py+D;
                    gx0=px; gy0=py-G; gx1=px+W; gy1=py;
                } else {
                    bx0=px; by0=py; bx1=px+D; by1=py+W;
                    gx0=px+D; gy0=py; gx1=px+D+G; gy1=py+W;
                }

                if (!rect_in_polygon(bx0,by0,bx1,by1,verts,n_verts)) continue;
                if (!rect_in_polygon(gx0,gy0,gx1,gy1,verts,n_verts)) continue;
                if (ceiling_at(bx0,bx1,ceil_segs,n_ceil) < needed_h) continue;
                if (rect_overlaps_any(bx0,by0,bx1,by1,obstacles,n_obs)) continue;
                if (rect_overlaps_any(gx0,gy0,gx1,gy1,obstacles,n_obs)) continue;
                if (rect_overlaps_any(bx0,by0,bx1,by1,p_rects,n_prects)) continue;
                if (rect_overlaps_any(gx0,gy0,gx1,gy1,p_rects,n_prects)) continue;
                if (rect_overlaps_any(bx0,by0,bx1,by1,p_gaps,n_pgaps)) continue;

                if (t_found < slots_per_thread) {
                    my_buf[t_found].f     = f_val;
                    my_buf[t_found].combo = ci;
                    my_buf[t_found].cand  = pi;
                    t_found++;
                }
                n_for_combo++;
            }
        }
        found[tid] = t_found;
    }

    int total_found = 0;
    for (int t = 0; t < n_threads; t++) total_found += found[t];

    int write = 0;
    for (int t = 0; t < n_threads; t++) {
        _SweepResult *src = buf + t * slots_per_thread;
        if (write != t * slots_per_thread)
            memmove(buf + write, src, found[t] * sizeof(_SweepResult));
        write += found[t];
    }

    qsort(buf, total_found, sizeof(_SweepResult), _cmp_sweep);

    int ret = total_found < topk ? total_found : topk;
    for (int i = 0; i < ret; i++) {
        out_results[2*i]   = buf[i].combo;
        out_results[2*i+1] = buf[i].cand;
    }

    free(found);
    free(buf);
    return ret;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * validate_single  –  sin cambios (solo se llama una vez)
 * ──────────────────────────────────────────────────────────────────────────── */
int validate_single(
    double W, double D, double H, double G,
    double px, double py, int rot,
    const double *verts,    int n_verts,
    const double *obstacles,int n_obs,
    const double *ceil_segs,int n_ceil,
    const double *p_rects,  int n_prects,
    const double *p_gaps,   int n_pgaps)
{
    double bx0,by0,bx1,by1, gx0,gy0,gx1,gy1;
    double needed_h = H + G;

    if (rot == 0) {
        bx0=px; by0=py; bx1=px+W; by1=py+D;
        gx0=px; gy0=py+D; gx1=px+W; gy1=py+D+G;
    } else if (rot == 90) {
        bx0=px; by0=py; bx1=px+D; by1=py+W;
        gx0=px-G; gy0=py; gx1=px; gy1=py+W;
    } else if (rot == 180) {
        bx0=px; by0=py; bx1=px+W; by1=py+D;
        gx0=px; gy0=py-G; gx1=px+W; gy1=py;
    } else {
        bx0=px; by0=py; bx1=px+D; by1=py+W;
        gx0=px+D; gy0=py; gx1=px+D+G; gy1=py+W;
    }

    if (!rect_in_polygon(bx0,by0,bx1,by1,verts,n_verts)) return 0;
    if (!rect_in_polygon(gx0,gy0,gx1,gy1,verts,n_verts)) return 0;
    if (ceiling_at(bx0,bx1,ceil_segs,n_ceil) < needed_h) return 0;
    if (rect_overlaps_any(bx0,by0,bx1,by1,obstacles,n_obs)) return 0;
    if (rect_overlaps_any(gx0,gy0,gx1,gy1,obstacles,n_obs)) return 0;
    if (rect_overlaps_any(bx0,by0,bx1,by1,p_rects,n_prects)) return 0;
    if (rect_overlaps_any(gx0,gy0,gx1,gy1,p_rects,n_prects)) return 0;
    if (rect_overlaps_any(bx0,by0,bx1,by1,p_gaps,n_pgaps)) return 0;
    return 1;
}
