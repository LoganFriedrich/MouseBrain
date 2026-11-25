# The analysis registry: an operator's guide

The analysis registry is where MouseBrain records the results of the 2D slice
tools, the 3D region counts (step 6) and the orientation check (counts,
measurements, figures) together with **how** they were made: the method
parameters, the source files, the time and the machine. You never have to call
it yourself -- those tools register their results as their last step. Steps 4
(detection) and 5 (classification) of the 3D pipeline do not register anything
yet; their record is the calibration tracker (`calibration_runs.csv`). This page says what it keeps, what the "approved method" is,
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
+-- exports/<analysis>/...                  the data files (CSV)
+-- figures/<analysis>/...                  the figures
+-- logs/<analysis>.log                     audit trail, one line per event
+-- logs/<analysis>.lock                    the write lock, only while something writes
+-- logs/<analysis>.registry.json.unreadable-<time>
|                                           a damaged manifest kept for repair (only
|                                           if one was ever damaged; see the end)
+-- approved_method.json                    optional: the parameters your lab approved
```

`exports/` and `figures/` hold only results and their manifests, nothing
temporary: anything that copies the registry (a lab database pulling it every
night, for example) can copy those two folders whole. That is why the lock,
the short-lived working files of a registration and any damaged manifest are
kept in `logs/` instead. The registry makes no folders of its own beyond
`exports/`, `figures/` and `logs/`.

What you may see in `logs/` besides the `.log` files:

- `<analysis>.lock` while something is writing (usually gone again within a
  second or so);
- for that same moment, files whose names start with a dot and end in `.tmp`
  or `.part` (a registration's working files), and files named
  `.clock-<8 letters/digits>` (the lock reading the file server's clock;
  each is deleted at once);
- `<analysis>.lock.break`: for a few thousandths of a second, while a writer
  takes over a lock left by a crashed writer (only one writer at a time may
  do that, and this file is how the others know to keep their hands off);
- only after a manifest was damaged, `<analysis>.registry.json.unreadable-<time>`
  (see "When the manifest cannot be read" -- do NOT delete this one before
  you have recovered its entries).

Leave the lock and the other files alone while anything might be registering.
Once nothing is registering, a `.lock`, a dot-file or a `.lock.break`
file that is still there was left by a crash (power cut, a process ended in
Task Manager) and is safe to delete.

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
2. `Registry/approved_method.json`.

When neither exists, **nothing is approved**: the tool has no approved method
of its own. Every analysis then has "no approved method"
(`approved_method_hash: null` in its manifest, nothing counted as stale), and
the 2D slice batch refuses to run until each of its settings is written into
the file or typed on its command line (see
`src/mousebrain/plugin_2d/ALGORITHM_REFERENCE.md`, section 6). WHY: which
nuclei count and how a cell is called positive are your lab's decisions for
its tissue and microscope, so the tool has no built-in parameter set to judge
results against: one would be reported as "the approved method" although nobody
had approved it.

Either file is plain JSON written in a text editor; a UTF-8 byte-order mark
(added by some editors and by Windows PowerShell) is accepted.

### Not every analysis has one

A plain `approved_method.json` (the parameters alone) describes the
**slice analysis** -- the parameters the slice batch tools run with. The 3D
region counts and the 3D orientation checks record their own parameters
(atlas, detection and classification settings, check version), which can never
match those, so they are **not** judged by them: their registries say
"No approved method is declared for this analysis" and `mousebrain-registry`
never counts any of their entries as stale. The code that registers an
analysis declares this, and the declaration is kept in the manifest
(`follows_approved_method`); the manifest header then says
`approved_method_hash: null`.

A lab database that pulls the registry counts "stale vs approved" on its own,
from that header. Whether it shows these analyses as "no approved method" or
still counts every entry as stale depends on that database reading
`approved_method_hash: null` as "no approved method"; check that yours does
before relying on its stale count for the 3D analyses.

**A 3D manifest written by an older version** has no declaration, so it is
judged by the slice parameters and shows its entries as stale. It picks
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
  out means there is no shared approved method: an analysis the file does
  not name then has none.
- The slice DETECTION batch (`<project>_Detection`, one analysis per
  project code) runs with the approved
  method of its analysis (its own entry under `"analyses"` if you gave it
  one, else `"default"`) for every number not typed on its command line or
  taken from a run marked best -- and stops, naming each one, when a number
  has none of these sources -- and records, as the parameters of each
  result, the values it ACTUALLY ran with. A result made with any other
  value is therefore stale, and its banner says so before it runs (see
  `src/mousebrain/plugin_2d/ALGORITHM_REFERENCE.md`, section 6).
- The ROI counter (`brainslice-roi batch`, analysis `<project>_ROI_Analysis`)
  records, for each sample's counts, the parameters of the DETECTION entry
  whose measurements it counted -- so ROI counts go stale exactly when the
  detection they rest on does. If the measurements cannot be tied to a
  current detection entry, the counts are recorded with a fixed "not tied"
  method and always show as stale (see [ROI_COUNTING.md](ROI_COUNTING.md)).
  If you give `<project>_Detection` its own entry under `"analyses"`, give
  `<project>_ROI_Analysis` the same entry, or every ROI count shows as stale;
  the counter prints a warning when the two differ. Naming a slice analysis
  under `"analyses"` takes effect from its next batch run.
- The old form -- the parameters alone, `{ "detection": ..., ... }` -- still
  works and means the same as `{"default": {...}}`.
- `"default"` and `"analyses"` are reserved words at the top of the file: a
  file with either one is read as the per-analysis form (a file holding only
  `{"default": {...}}` is fine), so neither can be the name of a parameter
  in the old form.
- A problem in the file stops, with a message naming the file, every
  registration whose approved method it could change -- and only those.
  A file that cannot be read at all, or whose `"analyses"` is not a table,
  stops every analysis. A bad entry for one analysis stops that analysis
  AND every analysis judged by the shared method that the file does not
  name (for example every slice batch registration), because the shared
  method is read from the same table. A bad `"default"` stops the analyses
  judged by the shared method. An analysis named with an entry of its own
  that is fine carries on, and so does an analysis with no approved method
  that the file does not name (the 3D counts and orientation checks,
  normally). A registration that
  is stopped changes nothing (see "When a registration is refused"). Fix the
  file; the tool never guesses which parameters you meant.

### After changing the approved method

Every time anything is registered, the manifest's header is rewritten with
the approved method as it is NOW (`approved_method`, `approved_method_hash`,
and `approved_method_recorded_at`, the time it was read). Anything that pulls
the registry (a lab database, for example) counts "stale vs approved" from
that header; `approved_method_hash: null` means the analysis has no
approved method and nothing in it is stale (a puller has to read it that way;
see "Not every analysis has one"). An analysis that has not
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
`none configured` (no file: nothing is approved). The same three fields are in `registry.json`
(`approved_method_recorded_at`, `approved_method_recorded_on`,
`approved_method_source`).

## Two machines writing at once

Every change to an analysis takes its lock, `logs/<analysis>.lock`,
first, so two registrations at the same moment both land, one after the
other. A registration holds the lock while it copies its files and writes the
manifest (about a second for a few small files; longer for large figures).
An invalidation holds it for the whole pass. For as long as a writer holds
the lock, it refreshes it by itself every 30 seconds -- even in the middle of
copying one large file over a slow network drive -- so a writer that is
still working never looks abandoned. Other writers print
`[Registry] Waiting for the lock ...` and wait for it.

The refresh is dated by the file server, never by the writer's own clock, so
a machine whose clock is wrong cannot make its own live lock look old to the
others.

When a writer finishes, it deletes the lock. On Windows that can fail for an
instant while another writer is reading the lock to see who holds it, so the
delete is retried for up to ten seconds. If it still fails, that writer
prints `[Registry] [!] Could not remove the lock <file> ...`.

The next writer takes over a lock that has gone untouched for two minutes
and prints `[Registry] [!] Took over a lock left <N> s ago by: <holder>`.
That happens when the holder crashed or was stopped while holding it (an
EMPTY lock file is a crash in the instant of creating it), or when the
holder's delete failed as above -- look on the holder's machine for the
`Could not remove the lock` line to tell which. A holder whose delete failed
had finished; one that crashed or was stopped while registering is covered
by the next section. A writer
that cannot get the lock in five minutes stops with a message naming the
holder; if nothing is really writing, delete the `.lock` file it names. A
machine whose account may not create files in `logs/` at all waits the same
five minutes and stops with a message saying "permission denied": give that
account write access to the registry folder.

**Every machine that registers results must run a version that takes the
lock.** An older version takes no lock, so a machine running it, registering
at the same moment as a newer one, is not held back and can still erase the
other's new entry. Update all such machines together.

## When a registration is refused

A registration first takes the lock, reads the manifest and works out the
approved method; only then does it put its files into the registry. If any of
those first steps refuses -- a damaged manifest, a broken
`approved_method.json`, the lock not coming free in five minutes -- nothing in
the registry changes: the files from the earlier registration of that sample
stay, and so does the entry describing them. If the manifest cannot be
written at the very end (a network drive dropping out), the files it was
replacing are put back. So a file in the registry is always the one its entry
describes.

The one case this cannot cover is the process being killed outright (power
cut, ended in Task Manager) in the middle of placing its files. Then the
sample's file may already be the new one while its entry still describes the
earlier run; the file it replaced is in `logs/`, named
`.<analysis>.<code>.replaced-<file name>.part`. To repair it, register that
sample again. Once nothing is registering, a file in `logs/` whose name
starts with `.<analysis>.` is such a leftover: a `replaced-` one holds an
earlier result, so move it into the `_archived/` folder beside the file it
replaced rather than deleting it; the others may be deleted.

The new results are still in the pipeline step's own output folder. Once the
cause is fixed, run the step (or its registration) again for those samples.

## Registering a sample again keeps the entry it replaces

When a sample that already has a current entry is registered again with a
different result (a re-run with other parameters, a re-export of its image,
new ROI counts), the earlier entry is kept, the same way invalidating keeps
it:

- every file the new registration writes over, and every file the earlier
  entry listed that the new one does not, is moved to an `_archived/<time>/`
  folder beside where it was. Nothing is deleted.
- `logs/<analysis>.log` gets a `replace_entry` line. Its last
  (tab-separated) column is JSON holding what the earlier entry said:
  `results`, `method_hash`, `method_params`, `source_files`,
  `registered_at`, `invalidated_at` (when it was replaced) and `archived_to`.
- the new entry in `registry.json` has a `replaced_entry` field: when the
  earlier entry was registered, its method hash, its results and the
  `_archived/<time>` folder its files went to.

WHY: overwriting the entry and deleting the files it replaced would keep the
earlier counts and the parameters that produced them nowhere; the log would
have only a `register` line with the new hash.

A registration that is identical to the current entry -- the same method
hash, the same results and files with exactly the same contents -- replaces
nothing: no archive folder, no `replace_entry` line. (The 2D window
registers an image's results each time they are saved; without this, every
save would archive another identical copy.)

The per-region ROI summary (`roi_summary_<region>.csv`) is not archived when
it is rewritten: it is built from the per-sample counts, and the earlier
per-sample counts file is archived.

If a replaced file cannot be moved to `_archived/` once the new entry has
been written (it is open in another program), it is left in `logs/` under a
name starting `.<analysis>.` and ending `.part`, and you see
`[!] <sample>: the earlier <file> could not be moved to ... it is kept as
...`; the same text is written to the log as a `replace_not_archived` line.
Move that file into the `_archived/<time>/` folder named in the message by
hand. If the `replace_entry` line cannot be written, the record is printed
as `[!] <key> was replaced, but its record could not be written ...`: copy
it into a text file beside the log, because it is then the only copy.

## A current entry whose newest run failed

The 2D detection batch (`batch_encr`) keeps a sample's earlier entry
**current** when the sample's re-run fails but the failure cannot be shown
to be the sample's: it was a data error (wrong channels, an image the
detector rejects) at a step that no other sample of the same run got past --
for example a run of that one image only. Such a failure may be the
software's, so the earlier result is not thrown away. (When another sample
DID get past that step, the earlier entry is retired; when the failure is a
share, a missing library or the like, the entry is kept and nothing is
recorded on it, because that says nothing about the sample.)

The entry is not left looking like an ordinary current result:

- it gets a `newest_attempt_failed` field in `registry.json`: `failed_at`
  (the step), `error`, `failure` ("sample"), `kept_because`, `method_hash`
  (the settings of the run that failed), `recorded_at` and `hostname`;
- `logs/<analysis>.log` gets an `attempt_failed` line with the same JSON;
- `mousebrain-registry --name <analysis>` lists it under
  `[!] Current, but the newest run failed: <n>`, one line per sample with
  the step, the time and the error;
- the batch's own summary says `[!] <sample>: earlier registry entry kept
  current ... the failure is recorded on the entry`.

What to do: fix the cause (look at the error) and run the batch again for
that sample, ideally together with samples that are known to work, so the
run can tell the sample's fault from the software's. The next successful
registration of the sample writes a new entry without the field.

## Invalidating results

`AnalysisRegistry(...).invalidate()` (all current entries) or
`invalidate(sample=...)` (one sample) moves each entry's files into an
`_archived/<time>/` folder beside where they were and marks the entry
invalidated. It works one entry at a time and writes the manifest after each:

- It first reads the manifest and the approved method. If the manifest is
  damaged, it stops with an error saying the manifest cannot be read before
  moving anything, and
  leaves the damaged file exactly where it is (only a registration sets a
  damaged manifest aside; see the end). If the approved-method file is
  broken, it stops with the file's error before moving anything.
- If ANY file of an entry cannot be moved (open in another program, for
  example -- it is retried for ten seconds first), the files already moved
  are put back, the entry stays **current**, you see
  `[!] <entry> NOT invalidated, still current: ...`, and it goes on to the
  next entry. Files are only ever renamed, never copied, so a file that
  could not be moved leaves no copy of itself in `_archived/`.
- If the files moved but the manifest cannot be written (the network drive
  dropped out, or `registry.json` was held open by another program for more
  than ten seconds), the files are put back, the entry stays **current**, you
  see the same `[!]` line plus `[!] Invalidation stopped ...`, and the
  command stops with the error. Entries done before it stay invalidated.
- If the pass is stopped some other way part-way through an entry's files
  (Ctrl+C, an unexpected error), that entry's moved files are put back
  before it stops; you see the same `[!] ... NOT invalidated` line. Entries
  done before it stay invalidated.

Invalidating an ROI-counts entry also takes its sample's row out of
`roi_summary_<region>.csv`, in the same all-or-nothing step: if the summary
cannot be rewritten, or the manifest cannot be written afterwards, the
summary and the files are put back and the entry stays current. WHY: the
summary is where a reader takes a region's counts from, and a retired count
left in it would still be reported.

`invalidate(sample=..., include_own_folder=True)` also moves every OTHER
file in the entry's own sample folder (a folder named after the sample alone,
such as `exports/<analysis>/<sample>/` for a detection entry; its `_archived/`
folder is left alone) into the same `_archived/<time>/`, under the same
rule: if any one of them will not move, everything already moved is put
back and the entry stays current. WHY: a folder can hold files of the same
run that the entry never listed, and archiving only the listed ones left
them in the active folder looking current. A folder shared by several
samples (an ROI analysis files by animal and region) is never swept. The
2D batch (`batch_encr`) always retires entries this way.

In every case, fix the cause and run it again; entries already invalidated
are skipped. Each failure is also written to `logs/<analysis>.log` as an
`invalidate_failed` line.

Each entry that IS invalidated gets its own `invalidate_entry` line in
`logs/<analysis>.log`, and the pass ends with one `invalidate` line giving
how many entries it retired. The `invalidate_entry` line carries, as JSON in
its last (tab-separated) column, what the entry said: `results`,
`method_hash`, `method_params`, `source_files`, `registered_at`,
`invalidated_at` and `archived_to` (the `_archived/<time>` folder). WHY: the
next registration of the same sample overwrites the entry in
`registry.json`, so without this line the retired results and the
parameters that produced them would be kept nowhere but in whatever files
the entry happened to list. Entries invalidated before this line existed
have only the `invalidate` count line.

Writing to the log never stops a pass (a log that cannot be written must not
leave files half moved). So when the `invalidate_entry` line itself cannot be
written (the log file is open elsewhere, the share drops out), the entry is
still invalidated, and `invalidate` prints `[!] <key> was invalidated, but its
record could not be written to logs/<analysis>.log (<error>). Its record is:
{...}` carrying the same JSON. That printed line is then the only copy of the
retired results and parameters once the sample is registered again: copy it
into a text file beside the log. Other log lines (`invalidate`,
`invalidate_failed`, `register`) that fail to write are not reported; they
describe events whose outcome `registry.json` and the files themselves still
show.

The one case this cannot cover is the process being killed outright (power
cut, ended in Task Manager) in the instant between moving an entry's files
and writing the manifest. That entry's files are then in `_archived/<time>/`
while the manifest still calls it current, and the log has no `invalidate`
line for it. To repair it, move the files back from `_archived/<time>/` (or
run the invalidation again after moving them back).

## When the manifest cannot be read

If `registry.json` is damaged (not valid JSON, or not a manifest), nothing
reads it as empty, because registering into an empty manifest would erase the
record of every earlier result.
Instead:

- `mousebrain-registry` prints `[FAIL]` with the reason and changes nothing --
  with any option, `--refresh-approved` included. An invalidation stops the
  same way and moves nothing.
- The next registration **moves** the damaged file to
  `logs/<analysis>.registry.json.unreadable-<time>` under the registry
  folder (its contents untouched) and fails that one registration with a
  message saying so. In the same moment it writes a NEW `registry.json` with
  no entries in the damaged one's place, so the analysis folder is never
  without a manifest (see below). It changes no result file (see "When a
  registration is refused"); what it does change is the manifest (moved
  aside, a new empty one started) and the audit log (a
  `set_aside_unreadable` line). The registrations after it add to the new
  manifest. The pipeline step prints the message as a warning line and
  carries on; the result of that ONE registration is in the step's own
  output folder, not in the registry -- run the step (or its registration)
  again for that sample.
- If the new manifest cannot be written (a network drive dropping out at
  that moment), the damaged file is put back exactly where it was and the
  registration fails with a message saying so; the next registration tries
  again.
- From then on, until a person deals with it, the set-aside file is never
  silent: `mousebrain-registry --name <analysis>` lists it first, under
  `[!] 1 damaged manifest of this analysis set aside`, with the time it was
  set aside and about how many entries it holds; every program reading the
  registry prints a `[!]` line naming it; and every new manifest of that
  analysis lists it in its header under `set_aside_manifests` (file, time,
  size, estimated entries), so a lab database pulling the registry can show
  it too -- but only if that database reads the field: check that yours
  does, or a set-aside analysis just looks smaller there. The entries in it
  are NOT counted anywhere until they are merged back.
- WHY a new manifest at once: anything that pulls the registry (a lab
  database) finds analyses by their `exports/<analysis>/registry.json`. A
  folder left without one -- for days, for an analysis registered once per
  brain -- would not be listed at all, and a puller that takes away what it
  no longer lists would drop the analysis and everything it had copied of
  it. With a manifest always there, the analysis stays listed and its files
  keep being copied; until the old entries are merged back it simply shows
  fewer entries.

To recover the old entries: open the `.unreadable-<time>` file, repair it (or
take the manifest from a backup), and copy its `"entries"` into
`exports/<analysis>/registry.json` (keep any entries the new file already
has: those are newer). An entry in the set-aside file describes the files as
they were when the manifest was set aside: the registration that set it aside
changed no file. A sample registered AGAIN since then has newer files, and
its entry in the new manifest describes them -- keep that one, and do not
copy the old entry for that sample back. Then
run `mousebrain-registry --name <analysis> --refresh-approved --summary` to
check it reads. The repaired file must be plain JSON (a text editor, not a
word processor). A UTF-8 byte-order mark, which some editors and Windows
PowerShell add when saving, is accepted: the registry reads the file normally
and that refresh rewrites it without the mark, so a lab database pulling the
registry reads it too. Finally delete (or move out of `logs/`) the
`.unreadable-<time>` file: the `[!]` lines stop, and the next write drops
`set_aside_manifests` from the header.

A manifest that cannot even be OPENED (a network drive dropping out) is left
exactly where it is; that registration fails and the next one tries again.
