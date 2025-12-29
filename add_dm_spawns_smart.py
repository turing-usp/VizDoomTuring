#!/usr/bin/env python3
"""
add_dm_spawns_smart.py

Adiciona (ou estende) spawnpoints de Deathmatch (Thing type=11) em um mapa dentro de um WAD.

Suporta:
- UDMF (TEXTMAP)  ✅  (seu multi_duel.wad é UDMF)
- DOOM/HEXEN (THINGS binário)  (fallback simples)

Para UDMF:
- Amostra pontos dentro de setores (via geometria de linedefs/sidedefs/vertices)
- Filtra por distância mínima de paredes (wall_clear)
- Seleciona pontos maximizando separação (farthest-point sampling) com relax automático do min-dist se necessário
- Insere novos "thing { ... }" no TEXTMAP com flags dm=true

Uso:
  python add_dm_spawns_smart.py --wad framework/maps/multi_duel.wad --map MAP01 --players 15 --min-dist 256 --wall-clear 48 --candidates 20000 --seed 0

Saída padrão:
  framework/maps/multi_duel_dm15_smart.wad
"""

from __future__ import annotations

import argparse
import math
import random
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ------------------------- WAD IO -------------------------

@dataclass(frozen=True)
class WadLump:
    name: str
    data: bytes


@dataclass(frozen=True)
class WadFile:
    ident: str  # "IWAD" or "PWAD"
    lumps: List[WadLump]


def read_wad(path: Path) -> WadFile:
    raw = path.read_bytes()
    if len(raw) < 12:
        raise ValueError("Arquivo WAD inválido (muito pequeno).")

    ident = raw[:4].decode("ascii", errors="strict")
    if ident not in ("IWAD", "PWAD"):
        raise ValueError(f"Ident inválido: {ident}")

    num_lumps, dir_ofs = struct.unpack_from("<ii", raw, 4)
    if dir_ofs < 12 or dir_ofs + num_lumps * 16 > len(raw):
        raise ValueError("Diretório WAD inválido (offset/tamanho fora do arquivo).")

    lumps: List[WadLump] = []
    for i in range(num_lumps):
        filepos, size = struct.unpack_from("<ii", raw, dir_ofs + i * 16)
        name = (
            raw[dir_ofs + i * 16 + 8 : dir_ofs + i * 16 + 16]
            .rstrip(b"\0")
            .decode("ascii", errors="replace")
        )
        if filepos < 0 or size < 0 or filepos + size > len(raw):
            raise ValueError(f"Lump #{i} inválido: {name} (pos/size fora do arquivo)")
        data = raw[filepos : filepos + size]
        lumps.append(WadLump(name=name, data=data))

    return WadFile(ident=ident, lumps=lumps)


def write_wad(path: Path, wad: WadFile) -> None:
    out = bytearray()
    out += wad.ident.encode("ascii")
    out += struct.pack("<ii", len(wad.lumps), 0)  # dir offset placeholder

    filepos = 12
    dir_entries = bytearray()

    for lump in wad.lumps:
        data = lump.data
        out += data
        name8 = lump.name.encode("ascii", errors="replace")[:8]
        name8 = name8 + b"\0" * (8 - len(name8))
        dir_entries += struct.pack("<ii", filepos, len(data))
        dir_entries += name8
        filepos += len(data)

    dir_ofs = filepos
    struct.pack_into("<i", out, 8, dir_ofs)  # patch dir offset
    out += dir_entries

    path.write_bytes(bytes(out))


# ------------------------- Map detection -------------------------

_MAP_MARKER_RE = re.compile(r"^(MAP\d\d|E\dM\d)$", re.IGNORECASE)


def is_map_marker(name: str) -> bool:
    return _MAP_MARKER_RE.match(name.strip()) is not None


@dataclass(frozen=True)
class MapRange:
    start_idx: int
    end_idx_excl: int


def find_map_range(lumps: Sequence[WadLump], map_name: str) -> MapRange:
    map_upper = map_name.upper()
    start = None
    for i, l in enumerate(lumps):
        if l.name.upper() == map_upper:
            start = i
            break
    if start is None:
        raise ValueError(f"Mapa {map_name} não encontrado no WAD.")

    end = len(lumps)
    for j in range(start + 1, len(lumps)):
        if is_map_marker(lumps[j].name):
            end = j
            break
    return MapRange(start_idx=start, end_idx_excl=end)


def detect_map_format(lumps: Sequence[WadLump], mr: MapRange) -> str:
    names = [lumps[i].name.upper() for i in range(mr.start_idx, mr.end_idx_excl)]
    # UDMF se tiver TEXTMAP em qualquer lugar dentro do range do mapa
    if "TEXTMAP" in names:
        return "UDMF"
    if "THINGS" in names:
        return "DOOM"
    raise ValueError("Formato do mapa não reconhecido: não achei TEXTMAP nem THINGS no range do mapa.")


# ------------------------- UDMF parsing helpers -------------------------

_BLOCK_RE_CACHE: Dict[str, re.Pattern] = {}


def iter_udmf_blocks(text: str, block_name: str) -> Iterable[str]:
    """
    Itera blocos tipo:
      block_name
      {
         ...
      }
    Retorna apenas o conteúdo dentro das chaves.
    """
    if block_name not in _BLOCK_RE_CACHE:
        _BLOCK_RE_CACHE[block_name] = re.compile(
            rf"(?ms)^\s*{re.escape(block_name)}\b[^\n]*\n\{{(.*?)^\s*\}}\s*"
        )
    pat = _BLOCK_RE_CACHE[block_name]
    for m in pat.finditer(text):
        yield m.group(1)


def parse_kv(body: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if "//" in s:
            s = s.split("//", 1)[0].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().rstrip(";").strip()
        d[k] = v
    return d


def parse_float(s: str) -> float:
    return float(s.strip())


def parse_int(s: str) -> int:
    return int(s.strip(), 0)


# ------------------------- Geometry & sampling (UDMF) -------------------------

def point_in_poly_evenodd(px: float, py: float, poly: Sequence[Tuple[float, float]]) -> bool:
    """
    Even-odd point-in-polygon. poly pode estar fechado (último=primeiro) ou não.
    """
    if len(poly) < 3:
        return False
    x0, y0 = poly[0]
    if poly[-1] != (x0, y0):
        pts = list(poly) + [(x0, y0)]
    else:
        pts = list(poly)

    inside = False
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if (y1 > py) != (y2 > py):
            x_int = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x_int > px:
                inside = not inside
    return inside


def point_segment_dist2(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    c1 = wx * vx + wy * vy
    if c1 <= 0:
        return (px - x1) ** 2 + (py - y1) ** 2
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return (px - x2) ** 2 + (py - y2) ** 2
    t = c1 / c2
    bx = x1 + t * vx
    by = y1 + t * vy
    return (px - bx) ** 2 + (py - by) ** 2


def min_dist_to_segments2(px: float, py: float, segs: Sequence[Tuple[float, float, float, float]]) -> float:
    return min(point_segment_dist2(px, py, *s) for s in segs)


@dataclass(frozen=True)
class SectorGeom:
    bbox: Tuple[float, float, float, float]  # minx,maxx,miny,maxy
    loops: Optional[List[List[Tuple[float, float]]]]  # lista de polígonos (loops) ou None
    segs: List[Tuple[float, float, float, float]]  # fallback p/ inside-test


def _build_loops_from_edges(edges: Sequence[Tuple[int, int]], vertices: Sequence[Tuple[float, float]]) -> Optional[List[List[Tuple[float, float]]]]:
    """
    Tenta reconstruir loops ordenados se o grafo tiver grau 2 (caso simples).
    Se tiver bifurcação (grau != 2), retorna None.
    """
    from collections import defaultdict

    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    if not adj:
        return None

    if any(len(nbrs) != 2 for nbrs in adj.values()):
        return None  # geometria complexa, cai no fallback

    visited_v: set[int] = set()
    loops: List[List[Tuple[float, float]]] = []

    for start in list(adj.keys()):
        if start in visited_v:
            continue

        loop_vidx: List[int] = [start]
        prev = None
        cur = start

        # caminhada até fechar
        for _ in range(len(adj) + 5):
            visited_v.add(cur)
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
            if nxt == start:
                loop_vidx.append(start)
                break
            loop_vidx.append(nxt)
            prev, cur = cur, nxt
        else:
            return None  # não fechou

        poly = [vertices[i] for i in loop_vidx]
        loops.append(poly)

    return loops if loops else None


def build_udmf_geometry(textmap: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float, float]], Dict[int, SectorGeom]]:
    # vertices
    vertices: List[Tuple[float, float]] = []
    for body in iter_udmf_blocks(textmap, "vertex"):
        kv = parse_kv(body)
        vertices.append((parse_float(kv["x"]), parse_float(kv["y"])))
    if not vertices:
        raise ValueError("UDMF: não achei vertices.")

    # sidedefs: idx -> sector
    sidedef_sector: List[int] = []
    for body in iter_udmf_blocks(textmap, "sidedef"):
        kv = parse_kv(body)
        sidedef_sector.append(parse_int(kv.get("sector", "0")))

    # linedefs
    linedefs: List[Tuple[int, int, int, int]] = []
    for body in iter_udmf_blocks(textmap, "linedef"):
        kv = parse_kv(body)
        v1 = parse_int(kv["v1"])
        v2 = parse_int(kv["v2"])
        sf = parse_int(kv.get("sidefront", "-1"))
        sb = parse_int(kv.get("sideback", "-1"))
        linedefs.append((v1, v2, sf, sb))
    if not linedefs:
        raise ValueError("UDMF: não achei linedefs.")

    # segments globais (paredes)
    all_segs: List[Tuple[float, float, float, float]] = []
    for v1, v2, _, _ in linedefs:
        x1, y1 = vertices[v1]
        x2, y2 = vertices[v2]
        all_segs.append((x1, y1, x2, y2))

    # edges por setor (usando índices de vértices)
    from collections import defaultdict

    sector_edges: Dict[int, set] = defaultdict(set)
    for v1, v2, sf, sb in linedefs:
        e = (v1, v2) if v1 <= v2 else (v2, v1)
        if sf >= 0 and sf < len(sidedef_sector):
            sector_edges[sidedef_sector[sf]].add(e)
        if sb >= 0 and sb < len(sidedef_sector):
            sector_edges[sidedef_sector[sb]].add(e)

    sector_geom: Dict[int, SectorGeom] = {}
    for sec, edges in sector_edges.items():
        # fallback segs coords
        segs_coords: List[Tuple[float, float, float, float]] = []
        xs: List[float] = []
        ys: List[float] = []
        for a, b in edges:
            x1, y1 = vertices[a]
            x2, y2 = vertices[b]
            segs_coords.append((x1, y1, x2, y2))
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        if not xs:
            continue

        loops = _build_loops_from_edges(list(edges), vertices)
        bbox = (min(xs), max(xs), min(ys), max(ys))
        sector_geom[sec] = SectorGeom(bbox=bbox, loops=loops, segs=segs_coords)

    return vertices, all_segs, sector_geom


def point_in_sector(px: float, py: float, g: SectorGeom) -> bool:
    minx, maxx, miny, maxy = g.bbox
    if not (minx <= px <= maxx and miny <= py <= maxy):
        return False

    # Se temos loops ordenados, usa XOR (even-odd) entre loops para lidar com buracos
    if g.loops:
        inside = False
        for poly in g.loops:
            if point_in_poly_evenodd(px, py, poly):
                inside = not inside
        return inside

    # fallback: even-odd em segmentos “soltos”
    inside = False
    for x1, y1, x2, y2 in g.segs:
        if (y1 > py) != (y2 > py):
            x_int = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x_int > px:
                inside = not inside
    return inside


def find_sector_for_point(px: float, py: float, sectors: Dict[int, SectorGeom]) -> Optional[int]:
    for sec, g in sectors.items():
        if point_in_sector(px, py, g):
            return sec
    return None


def sample_candidates(
    rng: random.Random,
    bbox: Tuple[float, float, float, float],
    sectors: Dict[int, SectorGeom],
    all_segs: Sequence[Tuple[float, float, float, float]],
    n_points: int,
    wall_clear: float,
    max_attempts: int,
) -> List[Tuple[float, float]]:
    minx, maxx, miny, maxy = bbox
    # evita grudar na borda do bbox
    pad = max(0.0, wall_clear)
    minx2, maxx2 = minx + pad, maxx - pad
    miny2, maxy2 = miny + pad, maxy - pad
    if minx2 >= maxx2 or miny2 >= maxy2:
        minx2, maxx2, miny2, maxy2 = minx, maxx, miny, maxy

    wall2 = wall_clear * wall_clear
    pts: List[Tuple[float, float]] = []
    attempts = 0

    while len(pts) < n_points and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(minx2, maxx2)
        y = rng.uniform(miny2, maxy2)

        sec = find_sector_for_point(x, y, sectors)
        if sec is None:
            continue

        # longe das paredes
        if min_dist_to_segments2(x, y, all_segs) < wall2:
            continue

        # “robustez”: exige que pequenos offsets ainda caiam em algum setor
        eps = 4.0
        if (
            find_sector_for_point(x + eps, y, sectors) is None
            or find_sector_for_point(x - eps, y, sectors) is None
            or find_sector_for_point(x, y + eps, sectors) is None
            or find_sector_for_point(x, y - eps, sectors) is None
        ):
            continue

        pts.append((x, y))

    if len(pts) < n_points:
        raise RuntimeError(
            f"Não consegui amostrar {n_points} pontos válidos (consegui {len(pts)}). "
            f"Tente reduzir --wall-clear ou reduzir --min-dist, ou aumentar --candidates."
        )
    return pts


def select_spawns_max_separation(
    existing: List[Tuple[float, float]],
    candidates: List[Tuple[float, float]],
    target_total: int,
    min_dist: float,
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Seleção greedy “farthest point” com relax automático do min_dist.
    Retorna (selected, min_dist_usado).
    """
    if len(existing) >= target_total:
        return existing[:target_total], float(min_dist)

    selected = list(existing)
    remaining = list(candidates)

    threshold = float(min_dist)
    thr2 = threshold * threshold

    # se não tem existing (raro), começa com um candidato qualquer (o mais central)
    if not selected and remaining:
        sx = sum(x for x, _ in remaining) / len(remaining)
        sy = sum(y for _, y in remaining) / len(remaining)
        best = max(remaining, key=lambda p: (p[0] - sx) ** 2 + (p[1] - sy) ** 2)
        selected.append(best)

    while len(selected) < target_total and remaining:
        best = None
        best_score = -1.0

        for x, y in remaining:
            d2 = min((x - sx) ** 2 + (y - sy) ** 2 for sx, sy in selected)
            if d2 >= thr2 and d2 > best_score:
                best_score = d2
                best = (x, y)

        if best is None:
            # relaxa
            if threshold <= 32.0:
                # pega o mais distante sem constraint
                best = max(
                    remaining,
                    key=lambda p: min((p[0] - sx) ** 2 + (p[1] - sy) ** 2 for sx, sy in selected),
                )
                selected.append(best)
                remaining.remove(best)
                continue

            threshold *= 0.9
            thr2 = threshold * threshold
            continue

        selected.append(best)
        remaining.remove(best)

    return selected, threshold


# ------------------------- UDMF spawn editing -------------------------

_THING_BLOCK_RE = re.compile(r"(?ms)^\s*thing\b[^\n]*\n\{.*?^\s*\}\s*")


def parse_udmf_things(textmap: str) -> List[Tuple[Tuple[int, int], Dict[str, str]]]:
    out: List[Tuple[Tuple[int, int], Dict[str, str]]] = []
    for m in _THING_BLOCK_RE.finditer(textmap):
        body = m.group(0)
        inner = re.search(r"(?ms)\{(.*)\}", body)
        if not inner:
            continue
        kv = parse_kv(inner.group(1))
        out.append((m.span(), kv))
    return out


def format_udmf_dm_start(x: float, y: float, angle: int = 0) -> str:
    # ZDoom UDMF: "height" é o campo usual; se ignorado, default é ok.
    return (
        "\nthing\n{\n"
        f"    x = {x:.3f};\n"
        f"    y = {y:.3f};\n"
        "    height = 0;\n"
        f"    angle = {angle};\n"
        "    type = 11;\n"
        "    skill1 = true;\n"
        "    skill2 = true;\n"
        "    skill3 = true;\n"
        "    skill4 = true;\n"
        "    skill5 = true;\n"
        "    skill6 = true;\n"
        "    skill7 = true;\n"
        "    skill8 = true;\n"
        "    single = true;\n"
        "    coop = true;\n"
        "    dm = true;\n"
        "}\n"
    )


def add_dm_spawns_udmf(
    textmap: str,
    players: int,
    min_dist: float,
    wall_clear: float,
    candidates_n: int,
    seed: int,
) -> Tuple[str, int, int, float]:
    things = parse_udmf_things(textmap)

    existing_dm: List[Tuple[float, float]] = []
    for _, kv in things:
        try:
            t = parse_int(kv.get("type", "-1"))
        except Exception:
            continue
        if t == 11 and "x" in kv and "y" in kv:
            try:
                existing_dm.append((parse_float(kv["x"]), parse_float(kv["y"])))
            except Exception:
                pass

    existing_count = len(existing_dm)
    if existing_count >= players:
        return textmap, existing_count, 0, float(min_dist)

    vertices, all_segs, sector_geom = build_udmf_geometry(textmap)

    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    bbox = (min(xs), max(xs), min(ys), max(ys))

    rng = random.Random(seed)
    candidates = sample_candidates(
        rng=rng,
        bbox=bbox,
        sectors=sector_geom,
        all_segs=all_segs,
        n_points=candidates_n,
        wall_clear=float(wall_clear),
        max_attempts=max(100_000, candidates_n * 50),
    )

    selected, used_min_dist = select_spawns_max_separation(
        existing=existing_dm,
        candidates=candidates,
        target_total=int(players),
        min_dist=float(min_dist),
    )

    to_add = selected[existing_count:]
    added_count = len(to_add)

    # inserir após o último bloco thing (mais seguro)
    insert_pos = 0
    last = None
    for m in _THING_BLOCK_RE.finditer(textmap):
        last = m
    if last is not None:
        insert_pos = last.end()
    else:
        m = re.search(r'(?m)^\s*namespace\s*=\s*".*?";\s*$', textmap)
        insert_pos = m.end() if m else 0

    add_blob = "".join(format_udmf_dm_start(x, y, angle=0) for x, y in to_add)
    new_textmap = textmap[:insert_pos] + add_blob + textmap[insert_pos:]

    return new_textmap, existing_count, added_count, used_min_dist


# ------------------------- DOOM/HEXEN (binary THINGS) fallback -------------------------

def _is_hexen_things(data: bytes) -> bool:
    # Hexen format = 20 bytes/thing
    return len(data) % 20 == 0 and len(data) % 10 != 0


def parse_doom_things(things_data: bytes) -> List[Tuple[int, int, int, int, int]]:
    if len(things_data) % 10 != 0:
        raise ValueError("THINGS lump (DOOM) inválido (tamanho não múltiplo de 10).")
    out = []
    for i in range(0, len(things_data), 10):
        x, y, angle, ttype, flags = struct.unpack_from("<hhhhh", things_data, i)
        out.append((x, y, angle, ttype, flags))
    return out


def build_doom_things(things: Sequence[Tuple[int, int, int, int, int]]) -> bytes:
    buf = bytearray()
    for x, y, angle, ttype, flags in things:
        buf += struct.pack("<hhhhh", int(x), int(y), int(angle), int(ttype), int(flags))
    return bytes(buf)


def add_dm_spawns_doom(things_data: bytes, players: int, seed: int) -> Tuple[bytes, int, int]:
    # Fallback simples: espalha em círculo crescente ao redor do primeiro DM start.
    things = parse_doom_things(things_data)
    existing_dm = [(x, y) for x, y, _, t, _ in things if t == 11]
    if len(existing_dm) >= players:
        return things_data, len(existing_dm), 0
    if not existing_dm:
        raise ValueError("Não há DM starts (type=11) no THINGS pra usar como base.")

    rng = random.Random(seed)
    base_x, base_y = existing_dm[0]
    existing_flags = next((flags for _, _, _, t, flags in things if t == 11), 23)
    need = players - len(existing_dm)

    radius = 192
    for i in range(need):
        ang = rng.uniform(0, 2 * math.pi)
        x = int(round(base_x + radius * math.cos(ang)))
        y = int(round(base_y + radius * math.sin(ang)))
        things.append((x, y, 0, 11, existing_flags))
        radius += 24

    return build_doom_things(things), len(existing_dm), need


# ------------------------- Main -------------------------

def default_out_path(in_path: Path, players: int) -> Path:
    stem = in_path.stem
    return in_path.with_name(f"{stem}_dm{players}_smart.wad")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", required=True, help="Input WAD path")
    ap.add_argument("--map", required=True, help="Map marker (e.g., MAP01, E1M1)")
    ap.add_argument("--players", type=int, required=True, help="Total desejado de DM spawn points (type=11)")
    ap.add_argument("--min-dist", type=float, default=192.0, help="Distância mínima alvo entre spawns (relaxa automaticamente se impossível)")
    ap.add_argument("--wall-clear", type=float, default=48.0, help="Distância mínima de paredes/linedefs para amostragem (UDMF)")
    ap.add_argument("--candidates", type=int, default=20000, help="Quantos candidatos amostrar (UDMF)")
    ap.add_argument("--seed", type=int, default=0, help="Seed RNG (determinístico)")
    ap.add_argument("--out", default=None, help="Output WAD path (default: ao lado do input)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.wad)
    wad = read_wad(in_path)

    mr = find_map_range(wad.lumps, args.map)
    fmt = detect_map_format(wad.lumps, mr)

    lumps = list(wad.lumps)

    if fmt == "UDMF":
        text_idx = None
        for i in range(mr.start_idx, mr.end_idx_excl):
            if lumps[i].name.upper() == "TEXTMAP":
                text_idx = i
                break
        if text_idx is None:
            raise RuntimeError("UDMF detectado mas TEXTMAP não encontrado no range do mapa.")

        textmap = lumps[text_idx].data.decode("utf-8", errors="replace")
        new_textmap, existing, added, used_min_dist = add_dm_spawns_udmf(
            textmap=textmap,
            players=int(args.players),
            min_dist=float(args.min_dist),
            wall_clear=float(args.wall_clear),
            candidates_n=int(args.candidates),
            seed=int(args.seed),
        )
        lumps[text_idx] = WadLump(name=lumps[text_idx].name, data=new_textmap.encode("utf-8"))
        print(f"UDMF: DM starts existentes={existing}, adicionados={added}, min_dist_usado≈{used_min_dist:.1f}")

    else:  # DOOM/HEXEN
        things_idx = None
        for i in range(mr.start_idx, mr.end_idx_excl):
            if lumps[i].name.upper() == "THINGS":
                things_idx = i
                break
        if things_idx is None:
            raise RuntimeError("Formato DOOM detectado mas THINGS não encontrado no range do mapa.")

        if _is_hexen_things(lumps[things_idx].data):
            raise ValueError("THINGS parece ser Hexen-format (20 bytes). Este script não edita Hexen-format ainda.")

        new_things, existing, added = add_dm_spawns_doom(lumps[things_idx].data, int(args.players), int(args.seed))
        lumps[things_idx] = WadLump(name=lumps[things_idx].name, data=new_things)
        print(f"DOOM: DM starts existentes={existing}, adicionados={added}")

    out_path = Path(args.out) if args.out else default_out_path(in_path, int(args.players))
    write_wad(out_path, WadFile(ident=wad.ident, lumps=lumps))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
