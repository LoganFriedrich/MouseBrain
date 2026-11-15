# The analysis registry: an operator's guide

The analysis registry is where MouseBrain records every result it produces
(counts, measurements, figures) together with **how** it was made: the method
parameters, the source files, the time and the machine. You never have to call
it yourself -- the pipeline steps and batch tools register their results as
their last step. This page says what it keeps, what the "approved method" is,
what "stale" means, and what to do when something goes wrong.

Everything below is done with one command, `mousebrain-registry`, run in the
MouseBrain environment:

```
conda activate <your MouseBrain environment>
mousebrain-registry --name <analysis> [options]
```

`<analysis>` is the name of one analysis, e.g. `3D_Region_Counts`,
`3D_Orientation_Check`, or the name a slice batch tool printed. Each analysis
has its own folder under `Registry/exports/`; the folder names are the
analysis names.

## Where it lives

```
<pipeline root>/Registry/
+-- exports/<analysis>/registry.json        the manifest: one entry per result
+-- exports/<analysis>/registry.json.lock   present only while something is writing (see below)
+-- exports/<analysis>/...                  the data files (CSV)
+-- figures/<analysis>/...                  the figures
+-- logs/<analysis>.log                     audit trail, one line per event
+-- approved_method.json                    optional: the parameters your lab approved
```

Set `MOUSEBRAIN_REGISTRY_ROOT` to put it somewhere else. `--root <folder>` on
the command does the same for one command.

## Looking at a registry

```
mousebrain-registry --name <analysis> --stale --summary
```

prints how many entries there are, how many are current and how many were
invalidated, which approved method the manifest header holds and when it was
recorded, whether any entry is **stale** (made with parameters other than the
approved ones, and which parameters differ), and a table of entries.
Looking never changes anything.

## The approved method

"Approved" means: the parameters your lab has signed off on. A result made
with other parameters is **stale** -- it should be made again (or kept on
purpose). The approved method is never written in the code; it comes from, in
this order:

1. the file named by the environment variable `MOUSEBRAIN_APPROVED_METHOD`;
2. `Registry/approved_method.json`;
3. the tool's built-in slice-analysis parameters, when neither exists.

### Not every analysis has one

The built-in parameters (and a plain `approved_method.json`) describe the
**slice analysis** -- the parameters the slice batch tools run with. The 3D
region counts and the 3D orientation checks record their own parameters
(atlas, detection and classification settings, check version), which can never
match those, so they are **not** judged by them: their registries say
"No approved method is declared for this analysis" and none of their entries
is ever counted as stale. The code that registers an analysis declares this,
and the declaration is kept in the manifest (`follows_approved_method`).

**A 3D manifest made before this version** has no declaration yet, so it is
still judged by the slice parameters and shows its entries as stale. It picks
the declaration up on its next registration (the next region count or
orientation check of any brain). `--refresh-approved` alone does NOT change
it. To record the declaration now, without waiting:

```
mousebrain-registry --name 3D_Region_Counts --no-approved-method --stale
mousebrain-registry --name 3D_Orientation_Check --no-approved-method --stale
```

`--no-approved-method` records "this analysis has no approved method unless
`approved_method.json` names it" and refreshes the header;
`--follows-approved-method` records the opposite. Neither changes any entry.

### Giving each analysis its own approved method

`approved_method.json` can name analyses one by one:

```json
{
  "default":  { "detection": "threshold+log", "threshold_fraction": 0.2, "...": "..." },
  "analyses": {
    "<an analysis>":       { "<parameter>": "<value>", "...": "..." },
    "<another analysis>":  null
  }
}
```

- An analysis named under `"analyses"` is judged by its own entry. `null`
  means "this analysis has no approved method".
- An analysis not named there is judged by `"default"` if it follows the
  shared method, and has no approved method otherwise. Leaving `"default"`
  out means the built-in parameters.
- The slice batch tools (detection and ROI counting) record, as the
  parameters of each result they register, the approved method of THEIR
  analysis: its own entry under `"analyses"` if you gave it one, else
  `"default"`. (They record the approved method; the values a batch actually
  ran with come from its command line and the tuning tracker.) So naming a
  slice analysis under `"analyses"` takes effect from its next batch run.
- The old form -- the parameters alone, `{ "detection": ..., ... }` -- still
  works and means the same as `{"default": {...}}`.
- `"default"` and `"analyses"` are reserved words at the top of the file: a
  file with either one is read as the per-analysis form (a file holding only
  `{"default": {...}}` is fine), so neither can be the name of a parameter
  in the old form.
- A file that cannot be read, or an entry that is not a set of parameters,
  stops every registration with a message naming the file. Fix the file; the
  tool never guesses which parameters you meant.

### After changing the approved method

Every time anything is registered, the manifest's header is rewritten with
the approved method as it is NOW (`approved_method`, `approved_method_hash`,
and `approved_method_recorded_at`, the time it was read). Anything that pulls
the registry (a lab database, for example) counts "stale vs approved" from
that header; when `approved_method_hash` is `null` the analysis has no
approved method and nothing in it is stale. An analysis that has not
registered anything since you changed the file still carries the old header,
so refresh it by hand:

```
mousebrain-registry --name <analysis> --refresh-approved --stale
```

Do this once for each analysis after changing `approved_method.json`. It
registers nothing, changes no entry, and does nothing for an analysis that has
no manifest yet.

### Every machine that registers must see the same approved method

The header is rewritten by whichever machine wrote last, from the approved
method as THAT machine resolves it. `MOUSEBRAIN_APPROVED_METHOD` wins over
`approved_method.json`, and it is set per machine. If one machine has it set
and another does not, the header -- and with it the "stale vs approved" count
a lab database shows -- changes back and forth depending on which machine
registered last. So: either set `MOUSEBRAIN_APPROVED_METHOD` to the same file
on every machine that runs the pipeline, or (simpler) set it on none and keep
one `Registry/approved_method.json`.

To check, look at the header line `mousebrain-registry --name <analysis>`
prints: `recorded <time> on <machine>, from <source>`. The source is
`MOUSEBRAIN_APPROVED_METHOD=<file>`, the path of `approved_method.json`, or
`built-in parameters`. The same three fields are in `registry.json`
(`approved_method_recorded_at`, `approved_method_recorded_on`,
`approved_method_source`).

## Two machines writing at once

Every change to a manifest takes `registry.json.lock` first, so two
registrations at the same moment both land. You will normally never see the
lock: a registration holds it for about a second. An invalidation holds it for
the whole pass (it touches the lock after each entry, so it never looks
abandoned), and other writers wait for it to finish.

If a machine crashed while holding the lock, the next writer takes it over
once the lock has gone untouched for two minutes and prints
`[Registry] [!] Took over a lock left ...`. That includes an EMPTY lock file,
which a crash in the instant of creating it leaves behind. A writer that
cannot get the lock in five minutes stops with a message naming the holder;
if nothing is really writing, delete the `.lock` file it names.

## Invalidating results

`AnalysisRegistry(...).invalidate()` (all current entries) or
`invalidate(sample=...)` (one sample) moves each entry's files into an
`_archived/<time>/` folder beside where they were and marks the entry
invalidated. It works one entry at a time and writes the manifest after each:

- It first reads the approved method. If that file is broken, it stops with
  the file's error before moving anything.
- If ANY file of an entry cannot be moved (open in another program, for
  example), the files already moved are put back, the entry stays
  **current**, you see `[!] <entry> NOT invalidated, still current: ...`, and
  it goes on to the next entry.
- If the files moved but the manifest cannot be written (the network drive
  dropped out, or `registry.json` was held open by another program for more
  than ten seconds), the files are put back, the entry stays **current**, you
  see the same `[!]` line plus `[!] Invalidation stopped ...`, and the
  command stops with the error. Entries done before it stay invalidated.

In every case, fix the cause and run it again; entries already invalidated
are skipped. Each failure is also written to `logs/<analysis>.log` as an
`invalidate_failed` line.

The one case this cannot cover is the process being killed outright (power
cut, ended in Task Manager) in the instant between moving an entry's files
and writing the manifest. That entry's files are then in `_archived/<time>/`
while the manifest still calls it current, and the log has no `invalidate`
line for it. To repair it, move the files back from `_archived/<time>/` (or
run the invalidation again after moving them back).

### What an invalidated entry says, and what happens when its sample comes back

An invalidated entry keeps its record in the manifest with
`is_current: false`, `invalidated_at`, `invalidated_reason` (when the code
that invalidated it gave a reason -- the slice detection batch does) and
`archived_outputs`: for each of its files, where that file now is under
`_archived/<time>/`. Its `outputs` keep the old active paths.

When the same sample is registered again (by `register_output` or
`register_roi_counts`), the new entry takes its place and the invalidated
record is kept inside it, under `superseded` (newest first). In that kept
record `outputs` names the ARCHIVED files -- the active paths now hold the
new result's files -- and a file that was already missing when the entry was
invalidated is left out. So the manifest still shows what the earlier result
was, why it was retired, and where its files are. An entry that was still
current when it was registered over is not kept: its files were overwritten
in place, so nothing would hold its data. Entries invalidated before
`archived_outputs` existed are kept with empty `outputs`.

## When the manifest cannot be read

If `registry.json` is damaged (not valid JSON, or not a manifest), nothing
reads it as empty -- that used to erase the record of every earlier result.
Instead:

- `mousebrain-registry` prints `[FAIL]` with the reason and changes nothing.
- The next registration **renames** the damaged file to
  `registry.json.unreadable-<time>` (its contents untouched), fails that one
  registration with a message saying so, and the registration after it starts
  a new, empty manifest. The pipeline step prints the message as a warning
  line and carries on (its own results on disk are not affected).
- Between the setting-aside and the next registration, the analysis's folder
  has NO `registry.json`. Anything that pulls the registry (a lab database)
  sees an analysis folder without a manifest for that time; it should keep
  what it already copied rather than treat the analysis as withdrawn.

To recover the old entries: open the `.unreadable-<time>` file, repair it (or
take the manifest from a backup), and copy its `"entries"` into the new
`registry.json` (keep any entries the new file already has). Then run
`mousebrain-registry --name <analysis> --refresh-approved --summary` to check
it reads.

A manifest that cannot even be OPENED (a network drive dropping out) is left
exactly where it is; that registration fails and the next one tries again.
