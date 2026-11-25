# Intel Arc (A770 / B580) — Configuração, Medições e Arquitetura

Este documento reúne o que foi medido e adaptado para placas **Intel Arc** neste fork (`victory-ka1qux`), cobrindo **Alchemist (Arc A770 8 GB / 16 GB, Xe-HPG)** e **Battlemage (Arc B580 12 GB, Xe2-HPG)**.

---

## 1. Pré-requisitos obrigatórios para placas Intel Arc

1. **Resizable BAR (ReBAR) ativado na BIOS/UEFI**
   - Essencial tanto na A770 quanto na B580: a camada copia e sincroniza centenas de MB por frame entre memória do título e buffers da placa. Sem ReBAR o tráfego PCIe degrada severamente.
2. **DirectX 12 Agility SDK (`1.616.1`)**
   - No **Windows 10** (como LTSC `19044`), o `D3D12Core.dll` nativo do sistema operacional limita o dispositivo a **Shader Model 6.5** e oculta `MeshShaderTier` e `ID3D12Device13`.
   - Mesmo no Windows 11, carregar o Agility SDK padroniza o runtime D3D12 (SM 6.8, Enhanced Barriers e `OpenExistingHeapFromAddress1`).
   - Como a política do repositório (`docs/asset-policy.md`) proíbe binários no Git, baixe o runtime para `target\...\D3D12\` sempre após clonar ou rodar `cargo clean`:
     ```powershell
     powershell -ExecutionPolicy Bypass -File scripts\fetch-agility.ps1
     ```
3. **Arquivo de paginação do Windows (Pagefile) em PCs com 16 GB de RAM**
   - O título reserva vários GB de *Windows commit* na inicialização (~24 GB somando processo + caches em jogo).
   - Em máquinas com **16 GB de RAM física**, configure o arquivo de paginação do Windows (em SSD NVMe) com pelo menos **24 GB a 32 GB** para evitar erro `1455` (`ERROR_COMMITMENT_LIMIT`) em `crashes.log`.

---

## 2. O que foi medido na Intel Arc A770 16 GB

Medido em **Windows 10 Enterprise LTSC (19044)**, `Intel(R) Arc(TM) A770 Graphics` (`8086:56a0`), driver `32.0.101.9033`, Agility SDK `1.616.1`, via `victory gpu-info` e pelas suítes completas de hardware (`victory-shader` e `victory-gpu`):

| Propriedade / Subsistema | Arc A770 16 GB (medido) | Arc B580 12 GB (Xe2 esperado) | Adaptação no código |
|---|---|---|---|
| VRAM dedicada | `16258 MB` (ou `8192 MB`) | `~12288 MB` | Orçamento dinâmico via DXGI `QueryVideoMemoryInfo` (`VICTORY_TEXTURE_ROOM_PERCENT=96` na A770 16 GB, `92` na B580 12 GB) |
| Shader Model máximo | `6.8` (via Agility SDK; `6.5` sem ele no Win10) | `6.8` | `victory_gpu::agility` exporta `D3D12SDKVersion = 616` e `D3D12SDKPath` |
| `WaveLaneCountMin..Max` | `8..32` (SIMD8 / SIMD16 / SIMD32) | `16..32` (SIMD16 / SIMD32) | Compute shaders com wave ops recebem `WaveSize(32)` automaticamente (`pins_wave`) e `wave64` do console divide em `2x wave32` |
| `DoublePrecisionFloatShaderOps` | `false` (sem `f64` nativo em hardware) | `false` / emulado | `WaveActiveSum` de `f64` (`active_f64_sum`) baixa para soma IEEE-754 exata em inteiros de 32/64 bits sem exigir `DOUBLES` |
| `MeshShaderTier` | Tier `1.0` (`10`) | Tier `1.0` (`10`) | Pipelines de mesh compilam, desenham (`edge_flagged_triangles`) e recarregam do `ID3D12PipelineLibrary1` |
| `DepthBoundsTestSupported` | `true` | `true` | Pipeline stream `DEPTH_STENCIL1` + `OMSetDepthBounds` validado no hardware |
| Limite do heap de views | `1,000,000` (`PORTABLE_VIEWS`) | `1,000,000` | `probe_view_limit` escolhe o slot nulo `999,999` desde o primeiro shader |
| `AtomicInt64OnDescriptorHeapResource` | `true` | `true` | Buffers 64-bit em `ResourceDescriptorHeap[]` funcionam nativamente |
| `AtomicInt64OnTypedResource` | `false` | `true` (novo no Xe2) | Na A770, atomics 64-bit em texturas tipadas (`R32G32_UINT`) baixam automaticamente para leitura/ALU `i64`/escrita `uint2`; na B580 rodam nativamente |
| `BarycentricsSupported` (`SV_Barycentrics`) | `false` (neste driver) | verificar via `gpu-info` | Quando `false`, `SV_Barycentrics` e `attributeAtVertex` baixam automaticamente para interpolante comum `BARYC` e interpolação de hardware |
| Arredondamento `f32tof16` (`v_add_f16`) | Para zero (`0x4201`, igual NVIDIA) | Para zero / par | `VICTORY_F16_NEAREST=1` força arredondamento exato do console (`0x4202`) se necessário |
| Memória *in-place* (`OpenExistingHeapFromAddress1`) | `true` (leitura e escrita, `0.7 ms` / 4 MB) | `true` | Suportado via `VICTORY_INPLACE=1` quando Agility SDK está ativo |

### Resultado das suítes de hardware na Arc A770 16 GB
- `cargo test -p victory-shader --lib`: **252 aprovados, 0 falhas** (`valu_tests`, `permlane_tests` em waves `8..32`, `wave64_tests`, `waterfall_tests`, `intersect_tests` 32/64-bit, `valid_mask_tests`, `size_tests`, mesh shaders e soma `soft-f64`).
- `cargo test -p victory-gpu --lib`: **174 aprovados, 0 falhas** (serial e paralelo).

---

## 3. Passo a passo de uso

### Passo 1 — Baixar o Agility SDK e conferir a GPU (não precisa do jogo)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch-agility.ps1
cargo build --profile diag -p victory --features diagnostics
.\target\diag\victory.exe gpu-info
```
Verifique se `d3d12 runtime` mostra `Agility SDK` e `world verdict` mostra `OK`.
Em PCs com vídeo integrado + placa dedicada, o `victory` já escolhe a GPU de alto desempenho automaticamente; para forçar manualmente, defina `$env:VICTORY_ADAPTER = "A770"` ou `"B580"` (ou o índice listado em `gpu-info`).

### Passo 2 — Quando tiver o dump do jogo
1. Copie `victory.example.toml` para `victory.toml` (ignorado pelo Git) e aponte `game_dir`, `cache_dir` e `data_dir` para pastas **fora** do repositório:
   ```powershell
   Copy-Item victory.example.toml victory.toml
   .\target\diag\victory.exe config
   ```
2. Carregue o perfil da sua placa na sessão do PowerShell:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

   # Para Intel Arc A770 16 GB:
   . .\scripts\arc-a770.ps1

   # Para Intel Arc A770 8 GB:
   . .\scripts\arc-a770.ps1 -VramGB 8

   # Para Intel Arc B580 12 GB:
   . .\scripts\arc-b580.ps1
   ```
   *(O próprio `victory.exe` agora detecta GPUs Intel (`0x8086`) e aplica automaticamente `VICTORY_INDIRECT_CLAMP=compute`, `VICTORY_RELAY=0`, `VICTORY_WIDE_BACK=0`, `VICTORY_RAYS=0`, `VICTORY_TEXTURE_ROOM_PERCENT=95`/`92` e pula o compute shader `f133e35efed2a48a` que causava `GEN12_DX12COMPUTE_BBHANG` pós-loading. Use `-Rays` após pré-aquecer `XboxBvhBuilder.dll` para testar o Ray Tracing, ou `-Fast` para testar o modo experimental `VICTORY_RELAY=2`.)*
3. Pré-aqueça o cache de shaders e de pipelines **na mesma sessão** (para usar exatamente as mesmas capacidades `wave_min`, `wave_max`, `doubles` e `view_limit`):
   ```powershell
   $g = "<pasta do seu jogo>"
   .\target\diag\victory.exe translate --cache --pipelines --used-pipelines `
       "$g\shaders.rpf" "$g\<exe do titulo>" "$g\XboxBvhBuilder.dll" "$g\amd_fidelityfx_upscaler_xbox.dll"
   ```
4. Execute o título:
   ```powershell
   .\target\diag\victory.exe run "$g\<exe do titulo>" --execute
   ```

---

## 4. Variáveis úteis para depuração em Intel Arc

| Variável | Quando testar na Arc |
|---|---|
| `VICTORY_ADAPTER=A770` (ou `B580`, `0`, `1`) | Escolhe explicitamente o adaptador DXGI |
| `VICTORY_INDIRECT_CLAMP=compute` (`1` / `0`) | Clampa todos os `DispatchIndirect` de compute (evita `GEN12_DX12COMPUTE_BBHANG` / `PAGEFAULT` por passes Wave64 dropados) sem limitar draws de geometria |
| `VICTORY_SOFT_F64=1` | Força a soma `f64` em inteiros mesmo em GPUs que declaram `doubles = true` |
| `VICTORY_SOFT_TYPED_ATOMIC64=1` (`=0` desliga) | Controla o fallback de atomics 64-bit em texturas tipadas (`RWTexture*<uint2>`) |
| `VICTORY_WAVE_WIDE=1` | Reduz recusas de compute shaders `wave64` com branches por lane |
| `VICTORY_BAKE_PROBE=1` | Diagnostica estado das texturas lidas e hash do resultado dos *bakes* de pele e roupas |
| `VICTORY_F16_NEAREST=1` | Arredondamento `f16` idêntico ao console (troca truncamento do `f32tof16` da Arc por round-to-nearest-even) |
| `VICTORY_MESH_OUTPUT_STRICT=1` | Recusa mesh shaders acima de 32 KiB caso o compilador IGC trave em algum deles |
| `VICTORY_D3D12_DEBUG=1` / `VICTORY_D3D12_DRED=1` | Em build `diag`, identifica o comando exato caso ocorra *device removal* |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | Pula draws/dispatches de um shader específico se algum shader travar no compilador da Intel |

---

## 5. Limitações conhecidas em Alchemist (Arc A770)

- **Pixel/Mesh shaders com operações de wave:** em SM 6.6 o atributo `WaveSize(32)` só existe para *compute shaders*. Se algum *pixel shader* do título depender de largura fixa de 32 lanes (além das garantias padrão do HLSL), o compilador Intel IGC poderá rodá-lo em SIMD8/SIMD16.
- **Atomics 64-bit em texturas tipadas na A770:** a arquitetura Alchemist não suporta `AtomicInt64OnTypedResource`; o tradutor baixa essas operações para leitura/ALU `i64`/escrita em `RWTexture*<uint2>` (não atômico entre threads concorrentes no mesmo texel). Na **Arc B580 (Xe2)** o recurso é nativo.
- **Baricêntricas em meshlets compartilhados na A770:** quando `BarycentricsSupported = false`, `SV_Barycentrics` usa o interpolante `BARYC` por canto (`vértice mod 3`), exato em triângulos desindexados e aproximado quando vértices são compartilhados no mesmo meshlet, enquanto pesos calculados pelo shader usam a interpolação perspectiva do hardware.
- **Orçamento de texturas (`VICTORY_TEXTURE_BUDGET_MB`):** mantenha essa variável **desativada** na A770 16 GB e B580 12 GB (como já fazem `scripts\arc-a770.ps1` e `scripts\arc-b580.ps1`) para que `resource::room()` use o orçamento vivo do DXGI (`94%` / `92%`) e não interrompa os uploads antes dos *bakes* de pele e roupas dos personagens.
