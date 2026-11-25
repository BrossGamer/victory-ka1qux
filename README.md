# victory-ka1qux

Runs an Xbox Series title natively on Windows, from files the user supplies.
A fork of [victory-fork](https://github.com/fxceless/victory-fork) by fxceless,
itself a fork of [VIctory](https://github.com/Marvisak/VIctory) by Marvisak,
developed and measured on an AMD Radeon card.

This fork collects what it took to get the title's world drawn on one AMD
machine: fixes of its own, open pull requests of victory-fork that were tested
there one by one, and ports from other forks. Every change that affects the
picture was checked in the running title on that machine, most of them against
a counter rather than against a screenshot.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's developer
or publisher.

## State on the test machine (2026-11-25)

Windows 11, Intel Core i5-13600KF, AMD Radeon RX 7800 XT 16 GB (RDNA3, driver
32.0.32029.2010), 32 GB RAM. Series X profile, diagnostics build.

| | |
|---|---|
| Boot, menus, saves | work (as upstream) |
| Getting into the world | about 3 minutes from launch, with the caches warm and the movies skipped |
| The world | drawn from a save: rooms with walls, ceilings and textures, lit by their lamps and by the sun; characters with hair, and with skin and clothes in some launches (see the known problems); exteriors with buildings; HUD and prompts |
| Play | a 25-minute play session, from the first room on to the next objective, on an earlier build of this branch |
| Frame rate | about 2 frames a second with the switches below: 0.40 to 0.60 s a frame, median 0.50 s; about 3.5 with victory-fork's speed defaults (median 0.29 s, one run) |
| Stability | no crash and no device removal in the last twelve runs (5 to 25 minutes each) |
| Memory | about 11 GB of the card's 16 and 6 GB of the PC's memory shared with it (13.7 and 14.9 GB before the texture fix below) |

Known problems:

- Whatever the sun or a lamp does not reach directly is dark. The title fills
  shadow with ray-traced light, and its ray-tracing structures are not built
  here (the builder is off).
- The shell of the first room (walls and ceiling) is missing in the opening
  shot of some launches and is there later in the same run.
- The characters' baked skin and clothes are right in some launches and black
  in others (one launch of three with this build had both right, one had black
  uniforms, one a black face too). The texture queue is empty at the bake in
  all three, so the cause is something else and is not found yet.
- Two of the title's compute programs do not translate, and the programs that
  only shade ray-tracing hits are left out on purpose.
- About 2 frames a second. Roughly a third of a frame is the card; the rest is
  preparing the frame and copying title memory to and from the card.
- Tested on one machine only. The switches under "Running it" are the ones the
  state above was measured with.

## What this fork adds to victory-fork

Each change is explained in its commit message and in a comment where it is.

**Copies of title memory that went stale** (walls, ceilings and materials
missing in some launches and not in others):

- A diagnostic that measures it, `VICTORY_STALE_CHECK=1`: at every update,
  each page of a copy the write tracking calls unchanged is compared with the
  title's. Before the fixes 5.4 % of those pages differed, 300,000 of them all
  zeros in the copy; with them 0.1 %.
- The fixes are open pull requests of victory-fork applied here: #438 by
  sennecools (write guards see the other views of the same physical pages, and
  the runtime's own writes) and #571 by arminkhaleghirod (DirectStorage page
  writes are told where the pages are mapped).

**Views and textures**

- A raw or structured buffer view that does not begin on 16 bytes of the host
  copy of its memory is made on a copy of its own range. The host's raw views
  begin on 16 bytes and the console's anywhere, so such views read wrong or
  were left null: more than 100,000 view creations a run.
- A command list's reset lets go of the textures its prelude uploaded. Two
  pull requests that were each right alone left every list holding every
  texture it had ever uploaded: no idle texture was freed, the card filled, and
  some 10,000 textures picked by index were never made.
- The queue of textures picked by index: slots never uploaded go ahead of
  re-looks at uploaded ones; re-looks are bounded per frame and back off for
  textures that keep changing.
- Texture views are told apart by the memory their mips really take, uploads
  start at the first mip that is there, and views the queue lost are put back
  (ported from [myokex/victory-amd](https://github.com/myokex/victory-amd)).
- The title bakes its characters' skin and clothes once, while it loads, from
  textures it picks by index. Before each bake the texture pass now runs until
  nothing more is due (it took at most 4,096 slots, once). That removes one
  cause of black skin and stained clothes; another remains (known problems).
- The texture ring of the intro movie no longer shows its frames out of order.
- A descriptor read only for its texture's size is answered by the host.

**Dispatches and pipelines**

- A dispatch of more than 65,535 groups in one dimension is run in parts,
  direct or indirect, with the clamp of indirect arguments on or off. The
  title's final gather of indirect light asks for 115,200. (After
  myokex/victory-amd, extended to any number of parts and to count buffers.)
- Compute programs that only shade ray-tracing hits get no host pipeline while
  the ray-tracing structures are not built: they cost AMD's compiler seconds
  each. Compute pipelines are waited for longer while a world loads.
- Offline warm-up of the shader cache and of the pipelines earlier runs used
  (victory-fork #282 by t4l03 and #237 by fffluffy-dev). The shader cache is
  keyed by the translator's sources, so a rebuild that leaves the translator
  alone keeps it, and a run after a warm-up compiles next to nothing while it
  plays (2 pipelines made, 2,770 loaded, in the run that measured it).

**Loading**

- A DirectStorage queue made for manual submission waits for `Submit`
  (victory-fork #575 by Cuebert), and a page list still being filled is read
  again for a moment before its request fails.
- `VICTORY_SKIP_MOVIES=1` ends the title's non-looping movies on their first
  playback tick (`all`: looping ones too).

**Translator**

- Mesh primitive exports take 9-bit vertex indices (ported from upstream).
- The valid-mask bit of a pixel export decides which pixels survive.
- A heap buffer asked only for its size gets a non-uniform handle
  (victory-fork #452 by MalfoyDraco99, for a crash of AMD's compiler).
- Sixteen-bit float outputs of a mesh shader reach the pixel shader as zeros on
  this card, and the 32 KiB limit on mesh outputs is the validator's alone:
  both are measured in tests (`mesh_size_tests`).

**Diagnostics**

- `VICTORY_STALE_CHECK=1` (above; `fix` also copies what it finds) and
  `VICTORY_FULL_REFRESH=1` (every copy copied whole at each update).
- The frame statistics say what became of buffer views that did not fit their
  copy, how the texture queue stands, and what each bake found due.
- `VICTORY_RENDERDOC=1` loads RenderDoc before the device and captures a
  frame on request.

### Pull requests of victory-fork merged or applied here

Thanks to their authors. Each was read for what it adds, built, and run
through the test suites on the Radeon before it went in; #123 was taken back
out because its tests fail on this card.

#9, #25, #71 and #185 (Noname0202), #24 (NuraadGit), #32 (armin), #38
(spacede), #40 and #52 (jaschinski97-ops), #49 (fever308), #51 (pgm1207), #61
(srcamarda), #67 and #83 (ryotdev), #77 (m335671), #78 and #129
(vicennpacheco), #80, #119 and #506 (Adrito), #87 and #374 (davutsimsekk), #90
(António Mendes), #98 (sillynocodecat), #104 and #112 (avt0m4t), #106
(PastiiJay), #114 (Grégory Bonzon), #121 and #175 (Kamiloj9), #122 (Davi713),
#135 (alexdeliu), #154 (BitWuehler), #169 (HailLucifer666), #186 (Vyacheslav
Venglovsky), #191 (gdawid94), #196 (JaumeMarRib), #237 (fffluffy-dev), #282
(t4l03), #362 (renzowit35), #438 (sennecools), #452 (MalfoyDraco99), #571
(arminkhaleghirod), #575 (Cuebert).

## Running it

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with shader
model 6.6 and mesh shaders, Rust (pinned in `rust-toolchain.toml`) with the
MSVC build tools, and several GB of free Windows commit.

```powershell
cargo build --profile diag -p victory --features diagnostics

$g = "<your game folder>"
$env:VICTORY_GAME_DIR  = $g
$env:VICTORY_CACHE_DIR = "<a folder for caches, outside the repository>"
$env:VICTORY_DATA_DIR  = "<a folder for saves, outside the repository>"

# Once after every change to the translator (5 to 15 minutes): translate the
# title's shaders and prebuild the pipelines earlier runs used. Use the same
# executable and the same switches as the run.
target\diag\victory.exe translate --cache --pipelines --used-pipelines `
    "$g\shaders.rpf" "$g\<the title's exe>" "$g\XboxBvhBuilder.dll" "$g\amd_fidelityfx_upscaler_xbox.dll"

# The switches the state above was measured with.
$env:VICTORY_INDIRECT_CLAMP = "0"   # the clamp of indirect arguments has dropped geometry here
$env:VICTORY_SKIP_MOVIES    = "1"
$env:VICTORY_RELAY = "0"; $env:VICTORY_WIDE_BACK = "0"
$env:VICTORY_CARD_BUFFERS_MIN_MB = "16"; $env:VICTORY_CARD_BUFFERS_MB = "512"
$env:VICTORY_LATE_WAIT_LISTS = "128"

target\diag\victory.exe run "$g\<the title's exe>" --execute
```

The last three lines of switches turn off victory-fork's newer speed defaults
(sending submissions ahead of the card). Measured on this card with this build,
they bring a frame from a median of 0.50 s down to 0.29 s (0.17 s at best),
with no failure in the one run made with them. An earlier version of them cost
characters their skin here, so they are off in the recommended settings until
more runs say they are safe.

Input: XInput pads; with none connected the keyboard is a pad while the
title's window has focus. Commands, the full list of switches and debugging
notes are in [docs/victory-fork-readme.md](docs/victory-fork-readme.md);
[docs/architecture.md](docs/architecture.md) describes the layers and
[docs/asset-policy.md](docs/asset-policy.md) what may never enter the
repository.

### Intel Arc (A770 / B580)

See [docs/intel-arc.md](docs/intel-arc.md) for the Intel Arc hardware
measurements, Agility SDK setup (required on Windows 10 for shader model 6.6+
and mesh shaders), wave-size pinning (`8..32` / `16..32` → `WaveSize(32)`),
integer `f64` wave reductions (`doubles = false`), and ready-to-use profiles:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch-agility.ps1
cargo build --profile diag -p victory --features diagnostics
.\target\diag\victory.exe gpu-info

. .\scripts\arc-a770.ps1          # or: . .\scripts\arc-b580.ps1
```

## Credits

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors: the loader, the runtime services, the Direct3D 12 back end and
  the shader translator.
- **[victory-fork](https://github.com/fxceless/victory-fork)** by fxceless and
  the authors of its pull requests: the work that first drew the title's world,
  and most of what is merged here.
- **[myokex/victory-amd](https://github.com/myokex/victory-amd)**: fixes found
  on the same graphics card, some ported here.
- **[XCL](https://github.com/xcl-dev/xcl)**: a different project with the same
  aim, whose public notes on the console's formats and on driver behaviour
  pointed at several problems. No code of it is used here.
- This fork's own changes were developed with
  [Claude Code](https://claude.com/claude-code).

## License

GPL-3.0-or-later, as VIctory and victory-fork (see `LICENSE`). The history of
both is kept below this fork's commits.
