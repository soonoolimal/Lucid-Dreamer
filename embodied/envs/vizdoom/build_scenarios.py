import argparse
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import vizdoom

from .parse_dyn_info import DYNAMICS, SCN_DIR

ACC_PATH = shutil.which('acc') or '/usr/local/bin/acc'
ACC_INCLUDE_PATH = str(Path.home() / 'acc')  # /home/username/acc

_CHEAT_SCRIPTS = {
    'deadly_corridor': [
        (
            'GiveInventory("ClipBox", 6);',
            'GiveInventory("ClipBox", 6);\n'
            '\n'
            '    // Cheat\n'
            '    SetActorProperty(0, APROP_Health, 100000);\n'
            '    GiveInventory("ClipBox", 10000);',
        ),
    ],
    'defend_the_center': [
        (
            'GiveInventory("ClipBox", 6);',
            'GiveInventory("ClipBox", 6);\n'
            '\n'
            '    // Cheat\n'
            '    SetActorProperty(0, APROP_Health, 100000);\n'
            '    GiveInventory("ClipBox", 10000);',
        ),
    ],
    'defend_the_line': [
        (
            '    GiveInventory("Clip",20);\n'
            '    /* Infinite ammo */\n'
            '    while(1)\n'
            '    {\n'
            '        delay(1);\n'
            '        GiveInventory("Clip", 1 );\n'
            '    }',
            '    // Cheat\n'
            '    SetActorProperty(0, APROP_Health, 100000);\n'
            '    GiveInventory("Clip", 1000);\n'
            '    GiveInventory("ClipBox", 100);',
        ),
    ],
    'health_gathering': [
        (
            'ClearInventory();',
            'ClearInventory();\n'
            '\n'
            '    // Cheat: raise MaxHealth above HP so medkits remain collectible\n'
            '    SetActorProperty(0, APROP_SpawnHealth, 200000);\n'
            '    SetActorProperty(0, APROP_Health, 100000);',
        ),
    ],
    'health_gathering_supreme': [
        (
            'ClearInventory();',
            'ClearInventory();\n'
            '\n'
            '    // Cheat: raise MaxHealth above HP so medkits remain collectible\n'
            '    SetActorProperty(0, APROP_SpawnHealth, 200000);\n'
            '    SetActorProperty(0, APROP_Health, 100000);',
        ),
    ],
}


def parse_wad(path):
    data = path.read_bytes()
    magic = data[:4].decode('ascii')
    n_lumps, dir_offset = struct.unpack('<II', data[4:12])
    lumps = []
    for i in range(n_lumps):
        base = dir_offset + i * 16
        offset, size = struct.unpack('<II', data[base:base + 8])
        name = data[base + 8:base + 16].rstrip(b'\x00').decode('ascii', errors='replace')
        lumps.append([name, data[offset:offset + size]])
    return magic, lumps


def assemble_wad(magic, lumps):
    lump_blob = b''
    offsets = []
    for _, data in lumps:
        offsets.append(12 + len(lump_blob))
        lump_blob += data
    dir_offset = 12 + len(lump_blob)
    header = magic.encode('ascii') + struct.pack('<II', len(lumps), dir_offset)
    directory = b''.join(
        struct.pack('<II', off, len(data)) + name.encode('ascii').ljust(8, b'\x00')[:8]
        for (name, data), off in zip(lumps, offsets)
    )
    return header + lump_blob + directory


def compile_acs(scripts_text):
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / 'script.acs'
        out = Path(tmpdir) / 'script.o'
        src.write_text(scripts_text)
        result = subprocess.run(
            [ACC_PATH, '-i', ACC_INCLUDE_PATH, str(src), str(out)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f'ACC compilation failed:\n{result.stderr}')
        return out.read_bytes()


def build_wad_tex(src_path, tex_src, tex_dst):
    """Texture replacement only; no SCRIPTS/BEHAVIOR modification."""
    magic, lumps = parse_wad(src_path)
    textmap_idx = next(i for i, (n, _) in enumerate(lumps) if n == 'TEXTMAP')
    textmap_text = lumps[textmap_idx][1].decode('utf-8')
    if tex_src not in textmap_text:
        raise ValueError(f"tex_src '{tex_src}' not found in TEXTMAP of {src_path.name}")
    lumps[textmap_idx][1] = textmap_text.replace(tex_src, tex_dst).encode('utf-8')
    return assemble_wad(magic, lumps)


def build_wad_cheat(src_path, scn_id, tex_src=None, tex_dst=None):
    """Cheat scripts with optional texture replacement; requires ACC."""
    magic, lumps = parse_wad(src_path)

    scripts_idx = next(i for i, (n, _) in enumerate(lumps) if n == 'SCRIPTS')
    scripts_text = lumps[scripts_idx][1].decode('utf-8').replace('\r\n', '\n')
    for old, new in _CHEAT_SCRIPTS[scn_id]:
        if old not in scripts_text:
            raise ValueError(f'anchor not found in SCRIPTS of {src_path.name}:\n{old!r}')
        scripts_text = scripts_text.replace(old, new, 1)
    lumps[scripts_idx][1] = scripts_text.encode('utf-8')

    if tex_src is not None:
        textmap_idx = next(i for i, (n, _) in enumerate(lumps) if n == 'TEXTMAP')
        textmap_text = lumps[textmap_idx][1].decode('utf-8')
        if tex_src not in textmap_text:
            raise ValueError(f"tex_src '{tex_src}' not found in TEXTMAP of {src_path.name}")
        lumps[textmap_idx][1] = textmap_text.replace(tex_src, tex_dst).encode('utf-8')

    behavior_data = compile_acs(scripts_text)
    lumps = [[n, d] for n, d in lumps if n != 'BEHAVIOR']
    scripts_idx = next(i for i, (n, _) in enumerate(lumps) if n == 'SCRIPTS')
    lumps.insert(scripts_idx, ['BEHAVIOR', behavior_data])

    return assemble_wad(magic, lumps)


def copy_defaults(scn_ids, force=False):
    """Copy original WAD/CFG files from vizdoom package to scenarios/default/."""
    src_dir = Path(vizdoom.__file__).parent / 'scenarios'
    out_dir = SCN_DIR / 'default'
    out_dir.mkdir(parents=True, exist_ok=True)
    for scn_id in scn_ids:
        for ext in ('wad', 'cfg'):
            src = src_dir / f'{scn_id}.{ext}'
            dst = out_dir / f'{scn_id}.{ext}'
            if not src.exists():
                print(f'[WARN] {src.name} not found in vizdoom package')
                continue
            if not dst.exists() or force:
                print(f'[COPY] {src.name} -> scenarios/default/')
                dst.write_bytes(src.read_bytes())
            else:
                print(f'[SKIP] {dst.name} already exists')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cheat', action='store_true', help='also build cheat variants (requires ACC)')
    parser.add_argument('--force', action='store_true', help='overwrite existing files')
    args = parser.parse_args()

    scn_ids = list({meta['id'] for meta in DYNAMICS.values()})
    copy_defaults(scn_ids, force=args.force)

    for _scn_name, meta in DYNAMICS.items():
        scn_id = meta['id']
        obs_shift = next(
            (meta[k]['obs_shift'] for k in range(meta['n_dynamics']) if meta[k]['obs_shift'] is not None),
            None,
        )

        src_wad = SCN_DIR / 'default' / f'{scn_id}.wad'
        src_cfg = SCN_DIR / 'default' / f'{scn_id}.cfg'

        if not src_wad.exists():
            print(f'[SKIP] {src_wad.name} not found')
            continue

        # default/{obs_shift}/
        if obs_shift is not None:
            tex_src, tex_dst = obs_shift.split('-to-')
            out_dir = SCN_DIR / 'default' / obs_shift
            out_dir.mkdir(parents=True, exist_ok=True)
            out_wad = out_dir / f'{scn_id}.wad'
            if not out_wad.exists() or args.force:
                print(f'[BUILD] {scn_id} (default/{obs_shift}) ...', end=' ', flush=True)
                out_wad.write_bytes(build_wad_tex(src_wad, tex_src, tex_dst))
                (out_dir / f'{scn_id}.cfg').write_bytes(src_cfg.read_bytes())
                print('Done')
            else:
                print(f'[SKIP] {scn_id} (default/{obs_shift}) already exists')

        if not args.cheat:
            continue

        # cheat/
        out_dir = SCN_DIR / 'cheat'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wad = out_dir / f'{scn_id}.wad'
        if not out_wad.exists() or args.force:
            print(f'[BUILD] {scn_id} (cheat) ...', end=' ', flush=True)
            out_wad.write_bytes(build_wad_cheat(src_wad, scn_id))
            (out_dir / f'{scn_id}.cfg').write_bytes(src_cfg.read_bytes())
            print('Done')
        else:
            print(f'[SKIP] {scn_id} (cheat) already exists')

        # cheat/{obs_shift}/
        if obs_shift is not None:
            tex_src, tex_dst = obs_shift.split('-to-')
            out_dir = SCN_DIR / 'cheat' / obs_shift
            out_dir.mkdir(parents=True, exist_ok=True)
            out_wad = out_dir / f'{scn_id}.wad'
            if not out_wad.exists() or args.force:
                print(f'[BUILD] {scn_id} (cheat/{obs_shift}) ...', end=' ', flush=True)
                out_wad.write_bytes(build_wad_cheat(src_wad, scn_id, tex_src, tex_dst))
                (out_dir / f'{scn_id}.cfg').write_bytes(src_cfg.read_bytes())
                print('Done')
            else:
                print(f'[SKIP] {scn_id} (cheat/{obs_shift}) already exists')


if __name__ == '__main__':
    main()
