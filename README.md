# AceFlow

A workflow-focused web UI built on top of **ACE-Step v1.5**.

AceFlow does **not** replace ACE-Step and does **not** reimplement the generation engine.  
It is a structured browser layer on top of the ACE-Step runtime: the UI collects inputs, the backend normalizes and validates them, the in-process queue serializes jobs, and ACE-Step performs the actual generation.

AceFlow exists for users who want something more guided and production-friendly than the raw Gradio page, without losing access to advanced features such as:

- model-aware task routing
- LoRA selection and A/B compare
- LM-assisted prompting and transcription helpers
- source-audio workflows beyond basic generation
- chord progression tools with real conditioning
- JSON import/export for round-trip reuse
- queue polling and per-job logs
- optional authentication and admin user management

In plain English:

- **ACE-Step is the engine**
- **AceFlow is the workflow dashboard**

---

## 🚀 Installation into an existing ACE-Step root

AceFlow is meant to be installed **from inside the root of your ACE-Step repository**.

That means before running the installer command, your shell should already be inside the folder that contains:

```text
acestep/
```

The installer downloads the AceFlow repository archive, extracts only the needed integration payload, and copies it into the ACE-Step tree.

### What the installer places in the ACE-Step root

AceFlow installation is expected to place:

```text
acestep/
  ui/
    aceflow/
start_aceflow_ui.bat
start_aceflow_ui.sh
```

So the installer copies **both** the AceFlow package folder **and** the two launchers into the ACE-Step root.

### What the installer does not need to copy

This README is intended as a **standalone release document**.  
It does **not** need to be installed automatically into the ACE-Step root unless you explicitly decide to ship it there yourself.

### Windows

On Windows the installer is a PowerShell script, but you can launch the same command either from:

- **PowerShell**
- regular **Command Prompt (`cmd.exe`)**

In both cases, run it only when your current working directory is already the **ACE-Step root**.

#### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/robustini/AceFlow/main/installer/install_aceflow.ps1 | iex"
```

#### Windows (Command Prompt)

```bat
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/robustini/AceFlow/main/installer/install_aceflow.ps1 | iex"
```

### Linux

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/robustini/AceFlow/main/installer/install_aceflow.sh)"
```

### Important note

These commands are **not** meant to be launched from an arbitrary folder.

Run them only when your current working directory is already the **ACE-Step root**, so the installer can correctly place:

- `acestep/ui/aceflow/`
- `start_aceflow_ui.bat`
- `start_aceflow_ui.sh`

### Installer repository source

The installer scripts are aligned to the AceFlow GitHub archive:

```text
https://github.com/robustini/AceFlow/archive/refs/heads/main.zip
```

A correct installed layout looks like this:

```text
ACE-Step/
├─ acestep/
│  └─ ui/
│     └─ aceflow/
│        ├─ app.py
│        ├─ queue.py
│        ├─ run.py
│        ├─ chord_file.py
│        ├─ chord_parser.py
│        ├─ chord_reference.py
│        ├─ chord_soundfont.py
│        ├─ chord_synth.py
│        ├─ chord_voicing.py
│        ├─ lora_catalog.json
│        ├─ songs.json
│        ├─ soundfonts/
│        ├─ static/
│        └─ vendor/
├─ start_aceflow_ui.bat
├─ start_aceflow_ui.sh
└─ ...
```

---

## Launchers and runtime setup

AceFlow ships with dedicated root launchers:

- `start_aceflow_ui.bat` → Windows launcher
- `start_aceflow_ui.sh` → Linux shell launcher with VRAM presets and saved last configuration

The application itself runs through the FastAPI factory entrypoint:

```bash
python -m uvicorn acestep.ui.aceflow.app:create_app --factory --host 0.0.0.0 --port 7861
```

### Default launcher behavior

The launchers are not decorative wrappers. They define part of the runtime before the browser opens.

Out of the box, the launcher workflow includes:

- default port `7861`
- default bind address `0.0.0.0`
- default results root `aceflow_outputs`
- default auth disabled
- default cleanup TTL `3600` seconds
- default turbo clamp bypass enabled
- saved last configuration in `.aceflow_last`

### Linux preset wizard

The shell launcher exposes these profiles:

- **8 GB** → stronger offload, INT8, LM `0.6B`
- **12–16 GB** → balanced profile, LM `1.7B`
- **24 GB+** → least constrained profile, LM `4B`
- **Custom** → manual configuration

### Runtime values exposed by the launcher

The launchers set or preserve values such as:

- `ACESTEP_REMOTE_CONFIG_PATH`
- `ACESTEP_REMOTE_LM_MODEL_PATH`
- `ACESTEP_REMOTE_DEVICE`
- `ACESTEP_REMOTE_LM_BACKEND`
- `ACESTEP_REMOTE_RESULTS_DIR`
- `ACESTEP_REMOTE_INIT_LLM`
- `ACESTEP_REMOTE_OFFLOAD_TO_CPU`
- `ACESTEP_REMOTE_OFFLOAD_DIT_TO_CPU`
- `ACESTEP_REMOTE_INT8_QUANTIZATION`
- `ACESTEP_REMOTE_COMPILE_MODEL`
- `ACESTEP_REMOTE_USE_FLASH_ATTENTION`
- `ACESTEP_REMOTE_LM_OFFLOAD_TO_CPU`
- `ACESTEP_REMOTE_LORA_ROOT`
- `ACEFLOW_AUTH_ENABLED`
- `ACEFLOW_SESSION_SECURE`
- `ACEFLOW_BYPASS_CORE_TURBO_STEP_CLAMP`
- `ACEFLOW_CLEANUP_TTL_SECONDS`

So the launchers are part of the product behavior, not just a convenience shortcut.

---

## ✨ What AceFlow is actually for

AceFlow is designed to make ACE-Step easier to use for real generation workflows.

The codebase is not only a frontend skin. It adds a structured layer around ACE-Step that covers:

- request normalization
- alias compatibility across payload shapes
- model-aware task visibility
- source-audio routing
- queue submission and status polling
- JSON round-tripping
- per-job outputs and logs
- optional auth gating

This matters because AceFlow is not limited to classic text-to-music prompting. It also understands specialized workflows such as:

- Cover
- Remix
- Repaint
- Extract
- Lego
- Complete

That makes it closer to a compact production panel than to a minimal demo page.

---

## 🧭 Runtime flow

The end-to-end path is:

1. You configure a job in the UI.
2. AceFlow builds a payload from the current UI state.
3. The backend normalizes aliases and mode-specific fields.
4. The request is validated.
5. The job enters an **in-process single-worker queue**.
6. ACE-Step performs the generation task.
7. AceFlow writes metadata, logs, uploads, and downloadable results.
8. The frontend polls `/api/jobs/{job_id}` until the result is ready.

This means AceFlow behaves like a controlled frontend with queue discipline, not like a page that blindly fires infinite concurrent jobs.

---

## 🎛️ Main generation modes

AceFlow starts from four base modes visible in the page layout and then extends them dynamically depending on the selected model.

### Simple

A lighter classic generation path for straightforward text-to-music use.

Use it when you want the shortest path from prompt to song without all the extra conditioning blocks.

### Custom

The most flexible mode.

This is where AceFlow exposes the richest combination of:

- prompt and lyrics
- advanced sampling parameters
- LoRA selection
- LM-assisted helpers
- chord-derived conditioning
- optional audio-code injection

Custom is also the mode where full chord conditioning can convert the generated harmonic reference into **audio codes**.

### Cover

Reference-based conditioning mode.

In Cover, the uploaded song is treated as the source for cover-style conditioning. When full chord conditioning is used here, the generated chord reference is routed as **reference WAV audio**, not as audio codes.

### Remix

Source-audio remix workflow.

You upload audio, keep text guidance available, and can also limit the active source window through:

- `Source start`
- `Source end`

Remix is still text-guided, but centered on transforming an uploaded source.

---

## 🧩 Model-aware extended task modes

AceFlow does not always show the same extra modes for every checkpoint.  
The backend reads the selected model inventory and supported task types, then the frontend enables only the modes that make sense for that model.

These extended modes include:

### Repaint

Repaint uses uploaded source audio plus a selected time window and repaint settings.

Typical use:

- keep a source clip
- repaint only a region or contextually rebuild it
- adjust repaint mode and repaint strength

AceFlow routes Repaint through source-audio fields, disables the LM “thinking” toggle for that task, and sends repaint-specific parameters such as window range and repaint strength.

### Extract

Extract is stem-oriented.

You upload a source audio file and choose a **single target track** to extract.

Practical behavior:

- source audio is required
- `track_name` is required
- caption is not the main creative control here
- lyrics are cleared
- LM thinking is disabled

This is not a song-writing mode. It is an extraction task.

### Lego

Lego is context-plus-track assembly.

You upload source audio, choose a **single target track**, and AceFlow uses that context to add or rebuild the missing part.

Compared with Extract, Lego keeps more generation-style behavior around the chosen track but is still fundamentally source-audio-driven.

### Complete

Complete is the multi-track completion workflow.

You upload source audio and choose **one or more track classes to complete**.

AceFlow now exposes that selection as a multi-select chip picker instead of a raw ugly listbox. The actual payload still sends the selected track classes to the backend, but the UI presents them in a more controlled way.

Practical behavior:

- source audio is required
- at least one track class must be selected
- the selection is multi-value
- AceFlow supports `Select all` and `Clear`

---

## 🧱 Main UI sections

AceFlow is not just a flat wall of inputs. The page is organized into sections that change meaning depending on the active mode.

### 1. Generation mode block

This is the first routing decision.

It determines:

- which source-audio blocks appear
- which fields are hidden or disabled
- which task-specific payload fields are added
- which submit action label is shown

For example:

- Extract changes the submit button to an extraction action
- modes like Extract and Lego hide several classic song fields
- Complete shows multi-track-class selection instead of a single track selector

### 2. Source audio block

This block changes label and help text depending on the active task.

AceFlow uses different semantics for:

- Cover
- Remix
- Repaint
- Extract
- Lego
- Complete

So the same upload area is not treated as the same thing in every mode.

### 3. Style and lyrics block

This is the main writing area for text-driven generation.

Depending on the active mode, AceFlow may:

- use both style and lyrics normally
- suppress or ignore some of them
- auto-fill caption from task context in extraction-style workflows

### 4. Song meta block

This includes:

- duration
- BPM
- seed
- key / scale
- time signature
- vocal language
- auto toggles for the above
- instrumental toggle
- LM thinking toggle

These are still important even when advanced features are used, because they define the basic musical target unless a task-specific mode hides them.

### 5. LoRA block

This section is not just a dropdown.

It includes:

- LoRA catalog loading
- LoRA weight
- optional A/B compare mode

The compare mode is intended for direct comparison between:

- generation without LoRA
- generation with the selected LoRA

while keeping the same stable seed and forcing batch size behavior needed by the comparison bridge.

### 6. LM settings block

This section controls the prompt-language-model side rather than the core audio model.

Relevant controls include:

- LM temperature
- LM CFG scale
- top-k
- top-p
- negative prompt
- constrained decoding
- CoT metas
- CoT caption
- CoT language
- parallel thinking
- constrained decoding debug

This is the layer that influences how the LM-side guidance is prepared or interpreted.

### 7. Chord progression block

This is one of AceFlow’s most distinctive sections.

It is not cosmetic. It can act both as:

- a semantic harmony hint layer
- a real rendered conditioning source

This section is explained in depth below.

### 8. Advanced settings block

This block contains the lower-level technical controls for sampling and conditioning, including:

- batch size
- ADG
- inference steps
- inference method
- timesteps override
- guidance scale
- shift
- CFG interval start / end
- normalization enable and target dB
- score sensitivity and auto score
- latent shift
- latent rescale
- audio cover strength
- cover noise strength
- chord reference renderer
- cover conditioning balance where relevant
- output format and MP3-specific bitrate / sample rate choices

### 9. LM hints / audio codes block

This block is where AceFlow bridges auxiliary audio guidance into symbolic conditioning.

It includes:

- uploading audio for code conversion
- converting audio into codes
- transcribing audio codes back into lyrics or hints
- directly editing `audio_codes`

This area becomes especially important when combined with full chord conditioning in Custom mode.

### 10. Import JSON block

AceFlow supports reloading previous job state directly into the UI.

You can import via:

- pasted JSON
- uploaded `.json` file

The importer tries to reconstruct the form from several possible shapes, including:

- AceFlow merged exports
- backend metadata-style payloads
- request-centered JSON shapes

---

## 🎸 LoRA management

AceFlow reads LoRA options from `lora_catalog.json` and also expects the actual LoRA folders to exist in the ACE-Step LoRA root.

### Where LoRAs are expected on disk

By default, the real LoRA folders are expected under the ACE-Step root, typically:

```text
ACE-Step/
└─ lora/
```

The launcher can also expose a custom `ACESTEP_REMOTE_LORA_ROOT`.

### What the AceFlow catalog does

The local catalog provides the UI-facing mapping for each LoRA entry, typically with:

- `id`
- `trigger`
- `label`

### Practical behavior

AceFlow uses a **single-LoRA-per-job** workflow.

If a LoRA is selected, AceFlow can also carry the trigger into the backend request so the conditioning path stays aligned with the chosen style.

### A/B compare mode

A/B compare is tied to the LoRA section because the most practical use is comparing:

- baseline generation
- same seed with selected LoRA

AceFlow keeps the normal player cards, but also adds a dedicated paired control layer for comparing the two results together.

---

## 🎼 Chord progression system

This is the part that deserved a much better explanation.

AceFlow’s chord system is not just a text box for harmony notes. It has a full internal pipeline composed of:

- Roman progression handling
- chord symbol parsing
- voicing selection
- renderer selection
- WAV generation
- optional audio-code extraction
- final routing into generation conditioning

### Inputs you define in the UI

The chord section lets you specify:

- **Chord key**
- **Chord scale** (`major` or `minor`)
- **Roman progression**
- **Optional section map**
- whether the chord result should influence key/scale, BPM, and lyrics

Example Roman progression:

```text
I - vi - IV - V
```

Example section map:

```text
Verse=I - vi - IV - V
Chorus=vi - IV - I - V
Bridge=ii - IV - I - V
```

### What AceFlow resolves internally

Once generated, AceFlow turns those Roman choices into a resolved concrete harmonic plan and exposes previews such as:

- resolved chords
- caption-oriented harmonic hints
- key/scale preview
- section summary preview

### What “Generate” does

`Generate` resolves the harmonic plan and updates the preview state.

This is the first sanity-check action. It lets you inspect what the Roman input is becoming before you inject it deeper into the job.

### What “Auto Sections” does

`Auto Sections` builds section overrides from the current lyric structure when possible.

The goal is to reduce manual section mapping when the lyrics already imply section names or repeated structure.

### What “Apply” does

`Apply` is the light semantic path.

It applies the chord setup to related UI state without yet forcing the full rendered-conditioning workflow.

This can affect things like:

- harmonic hints for the caption
- optional key/scale population
- optional BPM alignment
- optional lyric-section harmonic hints

In other words, `Apply` keeps the chord system mostly in the world of **semantic guidance**.

### What “Apply Full” does

`Apply Full` is the heavy path and the important one.

This action triggers the real chord-reference conditioning chain:

1. resolve the chord plan
2. render a temporary reference WAV
3. store it in the upload area
4. if needed, extract audio codes from that generated reference
5. route the result into the currently selected generation mode

This is where the chord system stops being a suggestion layer and becomes a real audio-conditioning asset.

### What “Remove” does

`Remove` clears the active chord-derived conditioning state and resets the full-conditioning attachment.

That means it removes the generated chord conditioning state, not just the text preview.

---

## 🔊 Chord rendering: with and without `.sf2`

AceFlow supports **two** renderers for chord-reference audio:

- **SoundFont renderer**
- **Internal synth renderer**

### Preferred renderer selection

The advanced settings block exposes `Chord reference renderer` with two choices:

- `soundfont`
- `internal`

### Where AceFlow looks for `.sf2`

AceFlow searches relative to the AceFlow package directory, not to the general ACE-Step root.

Supported folders are:

```text
acestep/ui/aceflow/soundfonts/
acestep/ui/aceflow/soundfont/
```

The intended folder is `soundfonts/`.

### Behavior when a SoundFont exists

If the renderer preference is `soundfont` and a valid `.sf2` is present, AceFlow renders the chord reference using the SoundFont path.

The SoundFont renderer uses:

- piano layer
- bass layer
- chosen chord voicings
- a fixed sample rate suitable for the reference render

The goal is to make the chord reference feel more like a musical instrument sketch instead of a bare internal tone source.

### Behavior when no SoundFont exists

If no `.sf2` is found, AceFlow **falls back automatically** to the internal renderer.

So the chord workflow still works even without a SoundFont. The difference is the character of the generated reference.

### Multiple `.sf2` files

If multiple `.sf2` files are present, AceFlow uses the first alphabetical match.

That means you should not dump ten random SoundFonts into the folder unless you deliberately want alphabetical selection to decide the winner.

### What the internal renderer does

The internal renderer synthesizes the chord reference directly without relying on external `.sf2` content.

This path still includes:

- parsed chord symbols
- selected voicings
- duration handling
- output gain handling
- WAV generation

So the fallback is real and functional, not a dummy mode.

### Practical difference between the two

- **SoundFont** → more instrument-like reference, usually more natural to hear
- **Internal synth** → no external asset required, more self-contained and reliable as fallback

### Practical advice

Use a **small or medium General MIDI `.sf2`** with usable piano and bass presets if you want the SoundFont path.

If you do not care about the rendered reference being especially realistic, the internal renderer is perfectly valid and avoids depending on extra files.

---

## 🧪 Full chord conditioning path by mode

The same chord reference does **not** route the same way in every mode.

### In Custom mode

When full chord conditioning is used in **Custom**, the generated chord reference is converted into **audio codes**.

That means the chain becomes:

```text
Roman progression → rendered WAV → extracted audio codes → Custom conditioning
```

### In Cover mode

When full chord conditioning is used in **Cover**, the generated chord reference is used as **reference WAV audio**.

That means the chain becomes:

```text
Roman progression → rendered WAV → Cover reference audio conditioning
```

### Why this distinction matters

The chord system is not a single static trick.

AceFlow uses different conditioning routes depending on whether the active task expects:

- direct reference audio
- audio codes
- source audio
- no extra chord asset at all

So the same harmonic plan can be injected differently depending on the workflow.

---

## 🧠 LM hints and audio-code workflow

AceFlow includes a separate helper block for audio-code-related tasks.

This section lets you:

- upload audio
- convert audio into codes
- transcribe codes
- manually inspect or edit the resulting `audio_codes`

This is useful in at least three scenarios:

1. you already have audio guidance and want symbolic conditioning from it
2. you want to inspect or reuse codes manually
3. you want to combine chord-derived reference generation with code extraction in Custom mode

So this section is not isolated from the chord system. It is one of the bridge layers that makes full chord conditioning practical.

---

## 📥 JSON import and 📤 JSON export

AceFlow supports both round-trip directions.

### JSON export

Each generated result card exposes a JSON download action.

The export is not just a raw backend dump. AceFlow builds a merged structure that may include:

- backend metadata
- original request data
- UI-state snapshot

This is why the export is useful for reloading a previous job with good fidelity.

### JSON import

The import block accepts:

- pasted JSON
- uploaded file

The importer tries to reconstruct the current UI state from several shapes and restore fields such as:

- generation mode
- model
- caption and lyrics
- LoRA selection and weight
- advanced sampling fields
- conditioning fields
- chord settings
- imported reference data
- selected track classes for Complete

This makes AceFlow practical for iterative workflows instead of one-off manual re-entry.

---

## 🔐 Optional authentication

AceFlow supports optional login protection.

### Main auth variables

Relevant values include:

- `ACEFLOW_AUTH_ENABLED`
- `ACEFLOW_ADMIN_EMAIL`
- `ACEFLOW_ADMIN_PASSWORD`
- `ACEFLOW_SESSION_COOKIE`
- `ACEFLOW_SESSION_SECURE`

### What happens when auth is enabled

AceFlow creates auth storage under the results root in:

```text
<results_root>/_auth/
├─ users.json
└─ access_log.jsonl
```

### Built-in behavior

The auth system supports:

- bootstrap admin creation
- temporary password flow
- forced password change on first login when needed
- one active session per user
- IP mismatch invalidation
- admin creation of additional users
- admin deletion of users
- audit log recording

This is more than a cosmetic login form. It is a lightweight but real admin gate.

---

## 📦 Outputs, uploads, metadata, and logs

By default, AceFlow writes under:

```text
<ACE-Step root>/aceflow_outputs
```

unless overridden by:

```text
ACESTEP_REMOTE_RESULTS_DIR
```

### Main structure

A typical shape is:

```text
aceflow_outputs/
├─ <job-id-1>/
│  └─ metadata.json
├─ <job-id-2>/
│  └─ metadata.json
├─ _uploads/
├─ _logs/
└─ _songs_generated.json
```

### Per-job folder

Each job gets its own folder identified by job id.

This folder stores the backend-side factual record, especially `metadata.json`.

### Upload area

Uploaded source/reference files are stored under:

```text
aceflow_outputs/_uploads/
```

Chord full-conditioning also writes its generated temporary reference WAV there.

### Logs

AceFlow captures runtime output into log files under:

```text
aceflow_outputs/_logs/
```

This includes backend execution information useful for debugging conditioning, LoRA loading, model routing, and runtime errors.

### Generation counter

AceFlow also stores a persistent counter file:

```text
aceflow_outputs/_songs_generated.json
```

---

## 🧹 Cleanup TTL

AceFlow cleanup is controlled by:

```text
ACEFLOW_CLEANUP_TTL_SECONDS
```

Default launcher value:

```text
3600
```

That means one hour.

### What cleanup affects

Cleanup can remove old:

- per-job output folders
- uploads
- log files

### Important behavior

This is **not** a separate daemon.

Cleanup is triggered when new jobs are submitted, so old files become eligible and are cleaned on later activity rather than by a constant background service.

---

## ⚙️ Queue and practical limits

AceFlow uses an in-process **single-worker FIFO queue**.

### What that gives you

- predictable ordering
- safer polling state
- easy cancellation of queued jobs
- reduced chaos when multiple requests arrive

### Practical limits visible in code and UI

Relevant limits include:

- minimum duration in the UI: **10 seconds**
- maximum duration: **600 seconds**
- queue active cap: **30 jobs**
- batch size range: **1–4**

So yes, AceFlow is friendlier than raw Gradio, but it still enforces guard rails.

---

## 🎚️ DiT model defaults and clamp behavior

AceFlow applies model-aware defaults and limits for inference steps.

### UI-side defaults

The frontend can propose different defaults depending on whether the model behaves like:

- turbo
- SFT / quality-oriented
- other non-turbo configurations

### Backend-side safety clamp

The backend normalizes inference steps again so the UI is not the only safety layer.

### Optional AceFlow turbo clamp bypass

AceFlow also supports an optional runtime patch controlled by:

```text
ACEFLOW_BYPASS_CORE_TURBO_STEP_CLAMP
```

When enabled, AceFlow can bypass the core turbo step clamp inside the AceFlow process and inject explicit timesteps when needed, so requests above the usual turbo default can actually run as intended.

This is process-local behavior and does not modify ACE-Step core files on disk.

---

## 🧠 Examples and random prompt fill

AceFlow can load examples from:

```text
acestep/ui/aceflow/songs.json
```

This powers the random example workflow in the UI.

You can use it to maintain:

- demo prompts
- reusable templates
- internal presets
- preferred stylistic starting points

---

## 📝 Troubleshooting

If AceFlow opens correctly but generation does not behave properly, the problem is usually in one of these areas:

- missing or invalid ACE-Step model setup
- invalid LoRA root or missing LoRA folders
- unsupported task for the selected model
- invalid uploaded source/reference path
- no usable `.sf2` when you expected SoundFont rendering
- backend runtime or GPU memory issues
- auth/token restrictions in protected API routes

### Specific chord-progression troubleshooting

If the chord feature does not behave as expected, check these separately:

- Roman progression syntax
- key / scale choice
- section map formatting
- selected chord renderer
- presence or absence of `.sf2`
- whether you expected semantic `Apply` or real `Apply Full`
- whether the active mode should consume audio codes or reference WAV

That distinction alone explains many “why didn’t it do what I expected?” cases.

---

## License

AceFlow is distributed under the **GNU General Public License v3.0 or later**.

Follow the same license and usage terms that apply to the ACE-Step environment, models, LoRAs, and any other assets used behind this UI.
