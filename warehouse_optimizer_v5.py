"""
=============================================================================
WAREHOUSE OPTIMIZER  ·  HackUPC 2026 / Mecalux  ·  v5 (Beam + LS + OpenMP)
=============================================================================

ARQUITECTURA:
  · BEAM SEARCH: mantiene top-K estados parciales en paralelo.
    En cada paso, cada estado genera hasta BEAM_TOPK candidatos vía C.
    Los mejores BEAM_WIDTH estados se propagan.
  · MULTIPROCESSING: los estados del beam se evalúan en paralelo con
    ProcessPoolExecutor — cada estado del beam en su propio proceso.
  · OpenMP en C: full_sweep_topk, full_sweep y batch_validate corren
    en paralelo a nivel de threads dentro del proceso (sin GIL).
    Controla con OMP_NUM_THREADS=N.
  · LOCAL SEARCH post-greedy: swap/remove para mejorar F.
  · Multi-start: arranca desde varias estrategias (ratio, área, mixto).
  · El mejor resultado global se devuelve.

FUNCIÓN OBJETIVO (minimizar):
  F = (sum_prices / sum_loads) ^ (2 - AreaUsed%)

REGLAS DE GAP:
  - gap DENTRO del almacén.
  - footprint nuevo NO puede invadir gap de otra bay.
  - gaps pueden solaparse entre sí.
=============================================================================
"""

import csv
import ctypes
import heapq
import math
import os
import sys
import time
import copy
import random
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

GRID_STEP    = 100          # mm grid resolution
MAX_PER_TYPE = 200          # max bays of each type
ROTATIONS    = [0, 90, 180, 270]

BEAM_WIDTH   = 6            # number of beam states to keep per step
BEAM_TOPK    = 12           # candidates to generate per beam state per step
LOCAL_SEARCH_ITERS = 80     # local search iterations after beam
TIME_LIMIT   = 25.0         # hard stop (seconds)

# ──────────────────────────────────────────────────────────────────────────────
# C EXTENSION
# ──────────────────────────────────────────────────────────────────────────────

def _load_lib():
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'warehouse_core_v3.so'),
        os.path.join(os.getcwd(), 'warehouse_core_v3.so'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'warehouse_core_v2.so'),
        os.path.join(os.getcwd(), 'warehouse_core_v2.so'),
    ]
    for so in candidates:
        if not os.path.exists(so):
            continue
        try:
            lib = ctypes.CDLL(so)
            D   = ctypes.c_double
            I   = ctypes.c_int
            PD  = ctypes.POINTER(D)
            PI  = ctypes.POINTER(I)

            # full_sweep (original)
            lib.full_sweep.restype  = I
            lib.full_sweep.argtypes = [
                PD, I, PD, I, PD, I, PD, I, PD, I, PD, I, PD, I,
                D, D, D, D, PI, PI,
            ]

            # full_sweep_topk (new)
            if hasattr(lib, 'full_sweep_topk'):
                lib.full_sweep_topk.restype  = I
                lib.full_sweep_topk.argtypes = [
                    PD, I, PD, I, PD, I, PD, I, PD, I, PD, I, PD, I,
                    D, D, D, D, I, PI,
                ]
                lib._has_topk = True
            else:
                lib._has_topk = False

            # validate_single (new)
            if hasattr(lib, 'validate_single'):
                lib.validate_single.restype  = I
                lib.validate_single.argtypes = [
                    D, D, D, D, D, D, I,
                    PD, I, PD, I, PD, I, PD, I, PD, I,
                ]
                lib._has_vs = True
            else:
                lib._has_vs = False

            print(f"[C ext] Loaded: {so}  topk={lib._has_topk}  vs={lib._has_vs}")
            return lib
        except Exception as e:
            print(f"[C ext] Error loading {so}: {e}")
    print("[C ext] No .so found — pure Python fallback.")
    return None


_lib = _load_lib()


def _darr(*values):
    n = len(values)
    if n == 0:
        arr = (ctypes.c_double * 1)(0.0)
    else:
        arr = (ctypes.c_double * n)(*values)
    return arr

def _iarr(*values):
    n = len(values)
    arr = (ctypes.c_int * max(n,1))(*values)
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# PURE PYTHON GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────

def polygon_area(v):
    n, a = len(v), 0.0
    for i in range(n):
        j = (i+1) % n
        a += v[i][0]*v[j][1] - v[j][0]*v[i][1]
    return abs(a)/2.0


def _pip(px, py, v):
    inside, j, n = False, len(v)-1, len(v)
    for i in range(n):
        xi,yi=v[i]; xj,yj=v[j]
        if ((yi>py)!=(yj>py)) and (px<(xj-xi)*(py-yi)/(yj-yi)+xi):
            inside=not inside
        j=i
    return inside


def _on_boundary(px, py, v, eps=1e-6):
    n=len(v)
    for i in range(n):
        j=(i+1)%n
        x1,y1=v[i]; x2,y2=v[j]
        if abs((x2-x1)*(py-y1)-(y2-y1)*(px-x1))<eps:
            if min(x1,x2)-eps<=px<=max(x1,x2)+eps and min(y1,y2)-eps<=py<=max(y1,y2)+eps:
                return True
    return False


def _rect_in_poly(rx0,ry0,rx1,ry1,v):
    for cx,cy in [(rx0,ry0),(rx1,ry0),(rx1,ry1),(rx0,ry1)]:
        if not (_on_boundary(cx,cy,v) or _pip(cx,cy,v)):
            return False
    for vx,vy in v:
        if rx0<vx<rx1 and ry0<vy<ry1:
            return False
    return True


def _overlap(ax0,ay0,ax1,ay1,bx0,by0,bx1,by1):
    return ax0<bx1 and ax1>bx0 and ay0<by1 and ay1>by0


def _ceil_h(x0,x1,segs):
    mh=float('inf')
    for i,(ss,sh) in enumerate(segs):
        se=segs[i+1][0] if i+1<len(segs) else float('inf')
        if se<=x0: continue
        if ss>=x1: break
        mh=min(mh,sh)
    return mh


# ──────────────────────────────────────────────────────────────────────────────
# PARSEO
# ──────────────────────────────────────────────────────────────────────────────

def parse_warehouse(fp):
    v=[]
    with open(fp,newline='') as f:
        for row in csv.reader(f):
            row=[r.strip() for r in row if r.strip()]
            if len(row)>=2: v.append((float(row[0]),float(row[1])))
    if len(v)<3: raise ValueError("≥3 vértices requeridos")
    area=polygon_area(v)
    print(f"[Warehouse] {len(v)} vértices, área={area/1e6:.2f} m²")
    return v, area


def parse_obstacles(fp):
    obs=[]
    try:
        with open(fp) as f:
            for line in f.read().strip().replace('=','\n').split('\n'):
                p=[x.strip() for x in line.split(',') if x.strip()]
                if len(p)>=4:
                    x,y,w,d=map(float,p[:4])
                    obs.append((x,y,x+w,y+d))
    except FileNotFoundError: pass
    print(f"[Obstacles] {len(obs)}")
    return obs


def parse_ceiling(fp):
    s=[]
    with open(fp,newline='') as f:
        for row in csv.reader(f):
            row=[r.strip() for r in row if r.strip()]
            if len(row)>=2: s.append((float(row[0]),float(row[1])))
    s.sort()
    print(f"[Ceiling] {s}")
    return s


def parse_bays(fp):
    bays=[]
    with open(fp,newline='') as f:
        for row in csv.reader(f):
            row=[r.strip() for r in row if r.strip()]
            if len(row)>=7:
                nl,pr=float(row[5]),float(row[6])
                bays.append({
                    'id':int(row[0]),'width':float(row[1]),'depth':float(row[2]),
                    'height':float(row[3]),'gap':float(row[4]),
                    'nloads':nl,'price':pr,
                    'ratio':pr/nl if nl>0 else float('inf'),
                    'area':float(row[1])*float(row[2]),
                })
    print(f"[Bays] {len(bays)} tipos")
    return bays


# ──────────────────────────────────────────────────────────────────────────────
# BAY GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────

def bay_rects(bay, px, py, rot):
    w,d,g = bay['width'],bay['depth'],bay['gap']
    if rot==0:   return (px,py,px+w,py+d),(px,py+d,px+w,py+d+g),'N'
    if rot==90:  return (px,py,px+d,py+w),(px-g,py,px,py+w),     'W'
    if rot==180: return (px,py,px+w,py+d),(px,py-g,px+w,py),     'S'
    return              (px,py,px+d,py+w),(px+d,py,px+d+g,py+w), 'E'


def footprint_dims(bay, rot):
    return (bay['width'],bay['depth']) if rot%180==0 else (bay['depth'],bay['width'])


# ──────────────────────────────────────────────────────────────────────────────
# GRID + CANDIDATE GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def build_grid(vertices, step):
    xs=[v[0] for v in vertices]; ys=[v[1] for v in vertices]
    xmin=int(math.floor(min(xs)/step)*step)
    xmax=int(math.ceil(max(xs)/step)*step)
    ymin=int(math.floor(min(ys)/step)*step)
    ymax=int(math.ceil(max(ys)/step)*step)
    half=step/2.0
    pts=[]
    for x in range(xmin,xmax,step):
        for y in range(ymin,ymax,step):
            if _pip(x+half,y+half,vertices):
                pts.append((float(x),float(y)))
    # Sort BL first
    pts.sort(key=lambda p:(p[1],p[0]))
    print(f"[Grid] {len(pts)} puntos (step={step}mm)")
    return pts


def btb_candidates(placed_list, snap):
    """Back-to-back pattern candidates."""
    extra = []
    for p in placed_list:
        br=p['rect']; gr=p['gap_rect']; rot=p['rotation']
        bx0,by0,bx1,by1=br
        gx0,gy0,gx1,gy1=gr
        if rot==0:
            for dx in (-snap, 0, snap):
                extra.append((round((bx0+dx)/snap)*snap, float(gy1)))
        elif rot==180:
            for dx in (-snap, 0, snap):
                for delta_d in (800,1000,1200,1600,2400):
                    extra.append((round((bx0+dx)/snap)*snap,
                                  round((gy0-delta_d)/snap)*snap))
        elif rot==90:
            for dy in (-snap, 0, snap):
                for delta_d in (800,1000,1200,1600,2400):
                    extra.append((round((gx0-delta_d)/snap)*snap,
                                  round((by0+dy)/snap)*snap))
        elif rot==270:
            for dy in (-snap, 0, snap):
                extra.append((float(gx1), round((by0+dy)/snap)*snap))
    return list(dict.fromkeys(extra))


def edge_candidates(placed_list, snap):
    """
    Generate candidates snapped right next to each placed bay
    (left/right/top/bottom touching edges). Helps dense packing.
    """
    extra = []
    for p in placed_list:
        bx0,by0,bx1,by1 = p['rect']
        for dx in (-snap, 0, snap):
            extra.append((round((bx1+dx)/snap)*snap, round(by0/snap)*snap))
            extra.append((round((bx0-snap+dx)/snap)*snap, round(by0/snap)*snap))
        for dy in (-snap, 0, snap):
            extra.append((round(bx0/snap)*snap, round((by1+dy)/snap)*snap))
            extra.append((round(bx0/snap)*snap, round((by0-snap+dy)/snap)*snap))
    return list(dict.fromkeys(extra))


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE
# ──────────────────────────────────────────────────────────────────────────────

def objective(sp, sl, oa, ta):
    if sl<=0 or sp<=0: return float('inf')
    return (sp/sl)**(2.0-oa/ta)


# ──────────────────────────────────────────────────────────────────────────────
# STATE — encapsulates a partial solution
# ──────────────────────────────────────────────────────────────────────────────

class State:
    __slots__ = ('placed_list','placed_rects','placed_gaps',
                 'sum_p','sum_l','occ_area','counts','f')

    def __init__(self, bay_types, total_area):
        self.placed_list  = []
        self.placed_rects = []
        self.placed_gaps  = []
        self.sum_p = 0.0
        self.sum_l = 0.0
        self.occ_area = 0.0
        self.counts = {b['id']:0 for b in bay_types}
        self.f = float('inf')

    def clone(self):
        s = object.__new__(State)
        s.placed_list  = list(self.placed_list)
        s.placed_rects = list(self.placed_rects)
        s.placed_gaps  = list(self.placed_gaps)
        s.sum_p    = self.sum_p
        s.sum_l    = self.sum_l
        s.occ_area = self.occ_area
        s.counts   = dict(self.counts)
        s.f        = self.f
        return s

    def commit(self, bay, px, py, rot, total_area):
        br, gr, fr = bay_rects(bay, px, py, rot)
        ew, ed = footprint_dims(bay, rot)
        self.placed_list.append({
            'bay':bay,'x':px,'y':py,'rotation':rot,'front':fr,
            'rect':br,'gap_rect':gr,
        })
        self.placed_rects.append(br)
        self.placed_gaps.append(gr)
        self.sum_p    += bay['price']
        self.sum_l    += bay['nloads']
        self.occ_area += ew*ed
        self.counts[bay['id']] += 1
        self.f = objective(self.sum_p, self.sum_l, self.occ_area, total_area)


# ──────────────────────────────────────────────────────────────────────────────
# C CALL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _make_combos(bay_types, counts):
    """Build sorted combos flat array. Returns (combos_flat, combo_meta)."""
    pq = []
    for b in bay_types:
        if counts[b['id']] < MAX_PER_TYPE:
            heapq.heappush(pq, (b['ratio'], b['id'], b))
    combos_flat = []
    combo_meta  = []
    while pq:
        _, _, b = heapq.heappop(pq)
        for rot in ROTATIONS:
            combos_flat += [b['width'], b['depth'], b['height'],
                            b['gap'], b['price'], b['nloads'], float(rot)]
            combo_meta.append((b, rot))
    return combos_flat, combo_meta


def sweep_topk(state, all_cands, bay_types, verts_flat, obs_flat, ceil_flat,
               total_area, topk):
    """
    Call full_sweep_topk (or fallback to full_sweep).
    Returns list of (bay, rot, px, py, f_val) — up to topk results.
    """
    combos_flat, combo_meta = _make_combos(bay_types, state.counts)
    if not combos_flat:
        return []

    n_combos = len(combo_meta)
    cands_flat = [c for pt in all_cands for c in pt]
    n_cands = len(all_cands)
    pr_flat = [c for r in state.placed_rects for c in r]
    pg_flat = [c for g in state.placed_gaps  for c in g]

    combos_a = _darr(*combos_flat)
    cands_a  = _darr(*cands_flat)
    verts_a  = _darr(*verts_flat)
    obs_a    = _darr(*obs_flat)   if obs_flat  else _darr(0.0)
    ceil_a   = _darr(*ceil_flat)
    pr_a     = _darr(*pr_flat)    if pr_flat   else _darr(0.0)
    pg_a     = _darr(*pg_flat)    if pg_flat   else _darr(0.0)

    results = []

    if _lib is not None and _lib._has_topk:
        out_arr = (ctypes.c_int * (2 * topk))()
        found = _lib.full_sweep_topk(
            combos_a, n_combos,
            cands_a,  n_cands,
            verts_a,  len(verts_flat)//2,
            obs_a,    len(obs_flat)//4  if obs_flat  else 0,
            ceil_a,   len(ceil_flat)//2,
            pr_a,     len(pr_flat)//4   if pr_flat   else 0,
            pg_a,     len(pg_flat)//4   if pg_flat   else 0,
            ctypes.c_double(state.sum_p),
            ctypes.c_double(state.sum_l),
            ctypes.c_double(state.occ_area),
            ctypes.c_double(total_area),
            ctypes.c_int(topk),
            out_arr,
        )
        for i in range(found):
            ci = out_arr[2*i]
            pi = out_arr[2*i+1]
            bay, rot = combo_meta[ci]
            px, py   = all_cands[pi]
            ew, ed   = footprint_dims(bay, rot)
            f_val    = objective(state.sum_p+bay['price'],
                                 state.sum_l+bay['nloads'],
                                 state.occ_area+ew*ed, total_area)
            results.append((f_val, bay, rot, px, py))

    elif _lib is not None:
        # Fallback: call full_sweep once
        out_c = ctypes.c_int(-1)
        out_p = ctypes.c_int(-1)
        _lib.full_sweep(
            combos_a, n_combos, cands_a, n_cands,
            verts_a, len(verts_flat)//2,
            obs_a,   len(obs_flat)//4  if obs_flat  else 0,
            ceil_a,  len(ceil_flat)//2,
            pr_a,    len(pr_flat)//4   if pr_flat   else 0,
            pg_a,    len(pg_flat)//4   if pg_flat   else 0,
            ctypes.c_double(state.sum_p),
            ctypes.c_double(state.sum_l),
            ctypes.c_double(state.occ_area),
            ctypes.c_double(total_area),
            ctypes.byref(out_c), ctypes.byref(out_p),
        )
        if out_c.value >= 0:
            bay, rot = combo_meta[out_c.value]
            px, py   = all_cands[out_p.value]
            ew, ed   = footprint_dims(bay, rot)
            f_val    = objective(state.sum_p+bay['price'],
                                 state.sum_l+bay['nloads'],
                                 state.occ_area+ew*ed, total_area)
            results.append((f_val, bay, rot, px, py))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def _rebuild_state(placed_list, bay_types, total_area):
    """Rebuild a State from a list of placements."""
    s = State(bay_types, total_area)
    for p in placed_list:
        s.commit(p['bay'], p['x'], p['y'], p['rotation'], total_area)
    return s


def _validate_placement(bay, px, py, rot,
                         verts, obs, ceil_segs,
                         prects, pgaps):
    """Pure-Python validation of a single placement."""
    br,gr,fr = bay_rects(bay,px,py,rot)
    bx0,by0,bx1,by1=br; gx0,gy0,gx1,gy1=gr
    if not _rect_in_poly(bx0,by0,bx1,by1,verts): return False
    if not _rect_in_poly(gx0,gy0,gx1,gy1,verts): return False
    if _ceil_h(bx0,bx1,ceil_segs)<bay['height']+bay['gap']: return False
    for ox0,oy0,ox1,oy1 in obs:
        if _overlap(bx0,by0,bx1,by1,ox0,oy0,ox1,oy1): return False
        if _overlap(gx0,gy0,gx1,gy1,ox0,oy0,ox1,oy1): return False
    for rx0,ry0,rx1,ry1 in prects:
        if _overlap(bx0,by0,bx1,by1,rx0,ry0,rx1,ry1): return False
        if _overlap(gx0,gy0,gx1,gy1,rx0,ry0,rx1,ry1): return False
    for gx0_,gy0_,gx1_,gy1_ in pgaps:
        if _overlap(bx0,by0,bx1,by1,gx0_,gy0_,gx1_,gy1_): return False
    return True


def local_search(state, bay_types, grid_pts,
                 warehouse_vertices, obstacle_list, ceiling_segments,
                 total_area, t_start, time_limit):
    """
    Improve state via:
      1. Remove-then-fill: remove one bad bay, try to fill freed space.
      2. Swap: replace bay type at position.
      3. Add-more: try to squeeze in more bays after current placement.
    """
    verts = warehouse_vertices
    obs   = obstacle_list
    ceil  = ceiling_segments

    best_f = state.f
    best_placed = list(state.placed_list)
    n_bays = len(best_placed)

    iters = 0
    while iters < LOCAL_SEARCH_ITERS and (time.time()-t_start) < time_limit:
        iters += 1
        if not best_placed:
            break

        move = random.randint(0, 2)

        if move == 0:
            # ── Remove worst-ratio bay ──────────────────────────────────────
            # Bay whose removal most improves current ratio
            if len(best_placed) < 2:
                continue
            # Score each bay: if removed, what's the new ratio?
            sp = sum(p['bay']['price']  for p in best_placed)
            sl = sum(p['bay']['nloads'] for p in best_placed)
            oa = sum((p['rect'][2]-p['rect'][0])*(p['rect'][3]-p['rect'][1])
                     for p in best_placed)
            best_remove_gain = 0.0
            best_remove_idx  = -1
            for i, p in enumerate(best_placed):
                bp = p['bay']['price']; bl = p['bay']['nloads']
                bw = p['rect'][2]-p['rect'][0]; bd2 = p['rect'][3]-p['rect'][1]
                new_sp = sp - bp; new_sl = sl - bl; new_oa = oa - bw*bd2
                if new_sl <= 0 or new_sp <= 0: continue
                new_f = objective(new_sp, new_sl, new_oa, total_area)
                gain = best_f - new_f
                if gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove_idx  = i

            if best_remove_idx < 0:
                continue

            new_placed = [p for i,p in enumerate(best_placed) if i != best_remove_idx]
            new_state = _rebuild_state(new_placed, bay_types, total_area)
            if new_state.f < best_f:
                best_f = new_state.f
                best_placed = new_placed
                print(f"  [LS remove] F={best_f:.5f}  bays={len(best_placed)}")

        elif move == 1:
            # ── Swap bay type at random position ───────────────────────────
            if not best_placed:
                continue
            idx = random.randint(0, len(best_placed)-1)
            p   = best_placed[idx]
            px, py, rot = p['x'], p['y'], p['rotation']

            # Build state without this bay
            other = [pp for i,pp in enumerate(best_placed) if i != idx]
            s_tmp = _rebuild_state(other, bay_types, total_area)
            prects = list(s_tmp.placed_rects)
            pgaps  = list(s_tmp.placed_gaps)

            improved = False
            for b in sorted(bay_types, key=lambda x:x['ratio']):
                if s_tmp.counts[b['id']] >= MAX_PER_TYPE:
                    continue
                for r in ROTATIONS:
                    if not _validate_placement(b, px, py, r, verts, obs, ceil,
                                               prects, pgaps):
                        continue
                    s2 = s_tmp.clone()
                    s2.commit(b, px, py, r, total_area)
                    if s2.f < best_f:
                        best_f = s2.f
                        best_placed = list(s2.placed_list)
                        improved = True
                        print(f"  [LS swap] F={best_f:.5f}  bays={len(best_placed)}")
                        break
                if improved:
                    break

        else:
            # ── Try adding one more bay greedily ────────────────────────────
            s_tmp = _rebuild_state(best_placed, bay_types, total_area)
            btb   = btb_candidates(best_placed, GRID_STEP)
            edge  = edge_candidates(best_placed, GRID_STEP)
            grid_set = set(grid_pts)
            extra = [p for p in btb+edge if p not in grid_set]
            cands = grid_pts + extra

            verts_flat = [c for pt in verts for c in pt]
            obs_flat   = [c for o  in obs   for c in o ]
            ceil_flat  = [c for s  in ceil  for c in s ]

            top = sweep_topk(s_tmp, cands, bay_types,
                             verts_flat, obs_flat, ceil_flat,
                             total_area, 1)
            if top:
                f_val, bay, rot, ppx, ppy = top[0]
                if f_val < best_f:
                    s_tmp.commit(bay, ppx, ppy, rot, total_area)
                    best_f = s_tmp.f
                    best_placed = list(s_tmp.placed_list)
                    print(f"  [LS add] F={best_f:.5f}  bays={len(best_placed)}")

    return _rebuild_state(best_placed, bay_types, total_area)


# ──────────────────────────────────────────────────────────────────────────────
# BEAM SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def _expand_beam_state(args):
    """
    Worker: expande UN estado del beam y devuelve la lista de (f, placed_list).
    Se ejecuta en un proceso separado → sin GIL, OpenMP activo dentro del .so.
    """
    (placed_list, bay_types_list, total_area, grid_pts,
     verts_flat, obs_flat, ceil_flat, sorted_bays, topk) = args

    # Reconstruir el state en el proceso worker
    state = _rebuild_state(placed_list, bay_types_list, total_area)

    btb      = btb_candidates(state.placed_list, GRID_STEP)
    edge     = edge_candidates(state.placed_list, GRID_STEP)
    grid_set = set(grid_pts)
    extra    = [p for p in btb + edge if p not in grid_set]
    all_cands = grid_pts + extra

    results = sweep_topk(state, all_cands, sorted_bays,
                         verts_flat, obs_flat, ceil_flat, total_area, topk)

    if not results:
        # Estado agotado — devolver tal cual
        return [(state.f, state.placed_list)]

    expansions = []
    for f_val, bay, rot, px, py in results:
        ns = state.clone()
        ns.commit(bay, px, py, rot, total_area)
        expansions.append((ns.f, ns.placed_list))

    return expansions


def beam_search(warehouse_vertices, total_area, obstacle_list,
                ceiling_segments, bay_types, grid_pts,
                t_start, time_limit,
                beam_width=BEAM_WIDTH, topk=BEAM_TOPK,
                sort_strategy='ratio'):
    """
    Beam search: maintain beam_width states.
    PARALELO: cada estado del beam se expande en su propio proceso
    (ProcessPoolExecutor). Dentro de cada proceso, OpenMP paraleliza
    el C-sweep a nivel de threads. Doble paralelismo: procesos × threads.
    """
    verts_flat = [c for pt in warehouse_vertices for c in pt]
    obs_flat   = [c for o  in obstacle_list      for c in o ]
    ceil_flat  = [c for s  in ceiling_segments   for c in s ]

    # Sort bay types according to strategy
    if sort_strategy == 'ratio':
        sorted_bays = sorted(bay_types, key=lambda b: b['ratio'])
    elif sort_strategy == 'area':
        sorted_bays = sorted(bay_types, key=lambda b: -b['area'])
    elif sort_strategy == 'mixed':
        sorted_bays = sorted(bay_types, key=lambda b: b['ratio'] / b['area'])
    else:
        sorted_bays = list(bay_types)

    # Cuántos procesos en el pool del beam (independiente del pool multi-strategy)
    # Usamos min(beam_width, cpus//2) para no saturar si hay varias estrategias
    beam_workers = max(1, min(beam_width, cpu_count() // 2))

    # Initial beam: one empty state
    beam = [State(bay_types, total_area)]

    iteration = 0
    while (time.time() - t_start) < time_limit:
        iteration += 1
        next_states = []

        if beam_workers > 1 and len(beam) > 1:
            # ── Expansión paralela de estados del beam ───────────────────────
            task_args = [
                (s.placed_list, bay_types, total_area, grid_pts,
                 verts_flat, obs_flat, ceil_flat, sorted_bays, topk)
                for s in beam
            ]
            try:
                with ProcessPoolExecutor(max_workers=beam_workers) as ex:
                    for expansions in ex.map(_expand_beam_state, task_args,
                                             chunksize=1):
                        for f_val, placed in expansions:
                            s = _rebuild_state(placed, bay_types, total_area)
                            next_states.append((s.f, s))
            except Exception as e:
                # Fallback secuencial si ProcessPoolExecutor falla
                print(f"  [Beam parallel] fallback: {e}", flush=True)
                for state in beam:
                    for expansions in [_expand_beam_state((
                        state.placed_list, bay_types, total_area, grid_pts,
                        verts_flat, obs_flat, ceil_flat, sorted_bays, topk))]:
                        for f_val, placed in expansions:
                            s = _rebuild_state(placed, bay_types, total_area)
                            next_states.append((s.f, s))
        else:
            # ── Expansión secuencial (beam pequeño o un solo core) ───────────
            for state in beam:
                btb   = btb_candidates(state.placed_list, GRID_STEP)
                edge  = edge_candidates(state.placed_list, GRID_STEP)
                grid_set = set(grid_pts)
                extra = [p for p in btb + edge if p not in grid_set]
                all_cands = grid_pts + extra

                results = sweep_topk(state, all_cands, sorted_bays,
                                     verts_flat, obs_flat, ceil_flat,
                                     total_area, topk)

                if not results:
                    next_states.append((state.f, state))
                    continue

                for f_val, bay, rot, px, py in results:
                    ns = state.clone()
                    ns.commit(bay, px, py, rot, total_area)
                    next_states.append((ns.f, ns))

        if not next_states:
            break

        # Keep best beam_width states (by F)
        next_states.sort(key=lambda x: x[0])
        beam = [s for f, s in next_states[:beam_width]]

        best_f   = beam[0].f
        n_placed = len(beam[0].placed_list)
        pct      = 100 * beam[0].occ_area / total_area if total_area else 0
        print(f"  [Beam iter {iteration}] best_F={best_f:.4f}  "
              f"bays={n_placed}  area={pct:.1f}%  "
              f"beam_size={len(beam)}  workers={beam_workers}", flush=True)

        # Early exit if no state improved
        all_exhausted = all(
            not sweep_topk(s, grid_pts, sorted_bays,
                           verts_flat, obs_flat, ceil_flat, total_area, 1)
            for s in beam[:1]
        )
        if all_exhausted:
            break

    return beam[0]  # best state


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-START RUNNER (for parallel execution)
# ──────────────────────────────────────────────────────────────────────────────

def _run_strategy(args):
    """Worker function for multiprocessing."""
    (warehouse_vertices, total_area, obstacle_list,
     ceiling_segments, bay_types, grid_pts,
     t_start, time_limit, strategy, beam_width, topk) = args

    print(f"\n[Strategy: {strategy}] starting...", flush=True)
    state = beam_search(
        warehouse_vertices, total_area, obstacle_list,
        ceiling_segments, bay_types, grid_pts,
        t_start, time_limit * 0.6,  # leave time for local search
        beam_width=beam_width, topk=topk,
        sort_strategy=strategy,
    )

    remaining = time_limit - (time.time() - t_start)
    if remaining > 1.0:
        print(f"  [{strategy}] beam done F={state.f:.4f}, running local search...",
              flush=True)
        state = local_search(
            state, bay_types, grid_pts,
            warehouse_vertices, obstacle_list, ceiling_segments,
            total_area, t_start, time_limit * 0.9,
        )

    print(f"  [{strategy}] final F={state.f:.4f}  bays={len(state.placed_list)}",
          flush=True)
    return state.placed_list, state.f


# ──────────────────────────────────────────────────────────────────────────────
# MAIN SOLVER
# ──────────────────────────────────────────────────────────────────────────────

def place_bays(warehouse_vertices, total_area, obstacle_list,
               ceiling_segments, bay_types):

    t_start = time.time()
    grid_pts = build_grid(warehouse_vertices, GRID_STEP)

    # Strategies to try in parallel
    strategies = ['ratio', 'area', 'mixed', 'ratio']
    beam_configs = [
        (BEAM_WIDTH, BEAM_TOPK),
        (4, 8),
        (3, 6),
        (BEAM_WIDTH*2, BEAM_TOPK//2),
    ]

    # Cap to available CPUs
    n_workers = min(len(strategies), max(1, cpu_count()))
    print(f"\n[Parallel] {n_workers} workers / {len(strategies)} strategies")

    time_per_worker = TIME_LIMIT * 0.85  # give workers 85% of limit

    task_args = [
        (warehouse_vertices, total_area, obstacle_list,
         ceiling_segments, bay_types, grid_pts,
         t_start, time_per_worker,
         strategies[i], beam_configs[i][0], beam_configs[i][1])
        for i in range(len(strategies))
    ]

    best_placed = []
    best_f = float('inf')

    if n_workers > 1:
        try:
            with Pool(processes=n_workers) as pool:
                results = pool.map(_run_strategy, task_args,
                                   chunksize=1)
            for placed, f in results:
                if f < best_f:
                    best_f = f
                    best_placed = placed
        except Exception as e:
            print(f"[Parallel] error: {e}, falling back to sequential")
            for args in task_args:
                if (time.time()-t_start) > TIME_LIMIT*0.9:
                    break
                placed, f = _run_strategy(args)
                if f < best_f:
                    best_f = f
                    best_placed = placed
    else:
        for args in task_args:
            if (time.time()-t_start) > TIME_LIMIT*0.9:
                break
            placed, f = _run_strategy(args)
            if f < best_f:
                best_f = f
                best_placed = placed

    # Final local search on best result with remaining time
    remaining = TIME_LIMIT - (time.time() - t_start)
    if remaining > 2.0 and best_placed:
        print(f"\n[Final LS] F={best_f:.5f}  remaining={remaining:.1f}s")
        best_state = _rebuild_state(best_placed, bay_types, total_area)
        best_state = local_search(
            best_state, bay_types, grid_pts,
            warehouse_vertices, obstacle_list, ceiling_segments,
            total_area, t_start, TIME_LIMIT - 0.5,
        )
        best_placed = best_state.placed_list
        best_f = best_state.f

    print(f"\n[BEST] F={best_f:.6f}  bays={len(best_placed)}")
    return best_placed


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def write_output(placed, fp='output.csv'):
    with open(fp,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['Id','X','Y','Rotation'])
        for p in placed:
            w.writerow([p['bay']['id'],int(p['x']),int(p['y']),p['rotation']])
    print(f"[Output] {fp}  ({len(placed)} bays)")


def print_report(placed, total_area):
    if not placed: print("Sin bays."); return
    sp=sum(p['bay']['price']  for p in placed)
    sl=sum(p['bay']['nloads'] for p in placed)
    oa=sum((p['rect'][2]-p['rect'][0])*(p['rect'][3]-p['rect'][1]) for p in placed)
    pct=oa/total_area
    f=objective(sp,sl,oa,total_area)
    print("\n"+"="*66)
    print(f"  Bays:{len(placed)}  Price:{sp:.0f}  Loads:{sl:.0f}  "
          f"Ratio:{sp/sl:.3f}")
    print(f"  Area:{oa:.0f}/{total_area:.0f} mm² ({pct*100:.2f}%)  "
          f"Exp:{2-pct:.4f}")
    print(f"  ★ F = {f:.6f}")
    print("="*66)
    print(f"  {'#':>3} {'id':>3} {'X':>7} {'Y':>7} {'rot':>5} {'fr':>3}  gap_rect")
    for i,p in enumerate(placed):
        g=p['gap_rect']
        print(f"  {i:3d} {p['bay']['id']:3d} {p['x']:7.0f} {p['y']:7.0f} "
              f"{p['rotation']:4d}° {p['front']:>3}  "
              f"({g[0]:.0f},{g[1]:.0f})→({g[2]:.0f},{g[3]:.0f})")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args=sys.argv[1:]
    wf = args[0] if len(args)>0 else 'warehouse.csv'
    of = args[1] if len(args)>1 else 'obstacles.csv'
    cf = args[2] if len(args)>2 else 'ceiling.csv'
    bf = args[3] if len(args)>3 else 'types_of_bays.csv'
    out= args[4] if len(args)>4 else 'output.csv'

    print("="*66)
    print(f"WAREHOUSE OPTIMIZER  ·  HackUPC 2026  ·  v5 (Beam+LS+OpenMP)")
    print(f"  grid={GRID_STEP}mm  beam={BEAM_WIDTH}  topk={BEAM_TOPK}  "
          f"ls_iters={LOCAL_SEARCH_ITERS}  time_limit={TIME_LIMIT}s")
    print("="*66)

    verts, area = parse_warehouse(wf)
    obs         = parse_obstacles(of)
    ceil        = parse_ceiling(cf)
    bays        = parse_bays(bf)

    t0 = time.time()
    placed = place_bays(verts, area, obs, ceil, bays)
    elapsed = time.time()-t0

    print_report(placed, area)
    write_output(placed, out)
    print(f"\n[Tiempo] {elapsed:.2f}s")


if __name__ == '__main__':
    main()
