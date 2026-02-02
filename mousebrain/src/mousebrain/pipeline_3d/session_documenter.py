"""
Session Documenter for auto-documentation of optimization sessions.

Automatically records all actions, parameter changes, and results during a tuning session.
Generates markdown reports summarizing what was tried and what worked.

Data Hierarchy:
    Brain (374) -> Subject (CNT_01_02) -> Cohort (CNT_01) -> Project (CNT)
    See 2_Data_Summary/DATA_HIERARCHY.md for full documentation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

try:
    from mousebrain.config import parse_brain_name, get_brain_hierarchy
except ImportError:
    # Fallback if config not available
    def parse_brain_name(brain_name: str) -> dict:
        return {"raw": brain_name}
    def get_brain_hierarchy(brain_name: str) -> str:
        return brain_name


class SessionDocumenter:
    """
    Auto-documents optimization sessions.

    Usage:
        doc = SessionDocumenter(brain_name="349_CNT_01_02")
        doc.start_session()

        # As user works...
        doc.log_param_change("threshold", 10, 8)
        doc.log_detection_run(run_id, params, cell_count)
        doc.log_curation(confirmed=47, rejected=12)
        doc.log_export(filepath)

        # On close
        doc.end_session()  # Generates report
    """

    def __init__(self, brain_name: str, output_dir: Optional[Path] = None):
        self.brain_name = brain_name
        # Parse hierarchy from brain name
        self.hierarchy = parse_brain_name(brain_name)
        self.hierarchy_string = get_brain_hierarchy(brain_name)

        # Save sessions to the project data folder, not C: drive
        if output_dir is None:
            # Try to find the project sessions folder using config
            try:
                from mousebrain.config import DATA_SUMMARY_DIR
                project_sessions = DATA_SUMMARY_DIR / "sessions"
            except ImportError:
                project_sessions = Path("Y:/2_Connectome/Tissue/3D_Cleared/2_Data_Summary/sessions")
            if project_sessions.parent.exists():
                self.output_dir = project_sessions
            else:
                # Fallback to home dir if project path doesn't exist
                self.output_dir = Path.home() / ".sci_connectome" / "sessions"
        else:
            self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_start: Optional[datetime] = None
        self.session_end: Optional[datetime] = None
        self.session_id: Optional[str] = None

        # Event log
        self.events: List[Dict[str, Any]] = []

        # Tracked state
        self.detection_runs: List[Dict[str, Any]] = []
        self.param_changes: List[Dict[str, Any]] = []
        self.exports: List[str] = []
        self.curations: List[Dict[str, Any]] = []
        self.notes: List[str] = []

        # Current parameters (for tracking changes)
        self.current_params: Dict[str, Any] = {}

    def start_session(self):
        """Start a new session."""
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        self._log_event("session_start", {"brain": self.brain_name})

    def end_session(self, generate_report: bool = True) -> Optional[Path]:
        """End the session and optionally generate report."""
        self.session_end = datetime.now()
        self._log_event("session_end", {})

        if generate_report:
            return self.generate_report()
        return None

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a timestamped event."""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        })

    def set_initial_params(self, params: Dict[str, Any]):
        """Set the initial parameter state."""
        self.current_params = params.copy()
        self._log_event("params_initial", params)

    def log_param_change(self, param_name: str, old_value: Any, new_value: Any):
        """Log a parameter change."""
        change = {
            "param": param_name,
            "from": old_value,
            "to": new_value,
            "timestamp": datetime.now().isoformat()
        }
        self.param_changes.append(change)
        self.current_params[param_name] = new_value
        self._log_event("param_change", change)

    def log_detection_run(self, run_id: str, params: Dict[str, Any],
                          cell_count: int, preset: str = "custom",
                          z_range: Optional[tuple] = None,
                          reference_match: Optional[float] = None):
        """Log a detection run."""
        run = {
            "run_id": run_id,
            "params": params.copy(),
            "preset": preset,
            "cell_count": cell_count,
            "z_range": z_range,
            "reference_match": reference_match,
            "timestamp": datetime.now().isoformat()
        }
        self.detection_runs.append(run)
        self._log_event("detection_run", run)

    def log_brain_loaded(self, source: str):
        """Log when a brain is loaded into napari."""
        self._log_event("brain_loaded", {"source": source, "brain": self.brain_name})

    def log_classification_run(self, run_id: str, model: str,
                                cells_found: int, rejected: int,
                                cube_size: int = 50, batch_size: int = 32):
        """Log a classification run."""
        run = {
            "run_id": run_id,
            "model": model,
            "cells_found": cells_found,
            "rejected": rejected,
            "acceptance_rate": cells_found / (cells_found + rejected) * 100 if (cells_found + rejected) > 0 else 0,
            "cube_size": cube_size,
            "batch_size": batch_size,
            "timestamp": datetime.now().isoformat()
        }
        self._log_event("classification_run", run)
        # Save checkpoint to update live log
        self.save_checkpoint()

    def log_curation(self, confirmed: int, rejected: int, skipped: int = 0):
        """Log a curation session."""
        curation = {
            "confirmed": confirmed,
            "rejected": rejected,
            "skipped": skipped,
            "timestamp": datetime.now().isoformat()
        }
        self.curations.append(curation)
        self._log_event("curation", curation)

    def log_export(self, filepath: str, export_type: str = "results"):
        """Log an export action."""
        self.exports.append(filepath)
        self._log_event("export", {"path": filepath, "type": export_type})

    def log_qc_action(self, action: str, approved: bool = True):
        """Log a QC action (approve/reject registration)."""
        self._log_event("qc_action", {"action": action, "approved": approved})

    def add_note(self, note: str):
        """Add a user note to the session."""
        self.notes.append(note)
        self._log_event("note", {"text": note})

    def log_previous_run_loaded(self, run_id: str, cell_count: int):
        """Log when a previous run is loaded for comparison."""
        self._log_event("previous_run_loaded", {"run_id": run_id, "cells": cell_count})

    # =========================================================================
    # COMPARISON AND ARCHIVAL EVENT TYPES
    # =========================================================================

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Generic event logger for custom event types.

        This is the public interface for logging arbitrary events from the
        tuning widget's new comparison/archival workflow.

        Event types:
            mark_as_best - User marked a detection as best
            promote_to_best - Promoted recent to best
            keep_best - Kept current best, archived recent
            archive_layer - Layer was archived
            diff_generated - Difference layers were generated
            mark_layer - Layer was marked as prev/new classification
            training_data_added - Points added to training bucket
            comparison_decision - User made a comparison decision

        Args:
            event_type: Type of event (see above)
            data: Event-specific data dictionary
        """
        self._log_event(event_type, data)
        # Save checkpoint to keep live log updated
        self.save_checkpoint()

    def log_mark_as_best(self, layer_name: str, points_count: int, exp_id: Optional[str] = None):
        """Log when user marks a layer as best."""
        self._log_event("mark_as_best", {
            "layer_name": layer_name,
            "points_count": points_count,
            "exp_id": exp_id,
        })
        self.save_checkpoint()

    def log_archive_layer(self, layer_name: str, archived_by: str):
        """Log when a layer is archived (demoted)."""
        self._log_event("archive_layer", {
            "layer_name": layer_name,
            "archived_by": archived_by,
        })
        self.save_checkpoint()

    def log_diff_generated(self, diff_type: str, tolerance: float,
                            count_a: int, count_b: int,
                            overlap: int, only_a: int, only_b: int):
        """Log when difference layers are generated."""
        self._log_event("diff_generated", {
            "type": diff_type,  # 'detection' or 'classification'
            "tolerance": tolerance,
            "count_a": count_a,
            "count_b": count_b,
            "overlap": overlap,
            "only_a": only_a,
            "only_b": only_b,
            "net_change": count_b - count_a,
        })
        self.save_checkpoint()

    def log_comparison_decision(self, decision: str, from_layer: str, to_layer: str):
        """Log user's comparison decision (promote, keep, archive)."""
        self._log_event("comparison_decision", {
            "decision": decision,  # 'promote', 'keep', 'archive'
            "from_layer": from_layer,
            "to_layer": to_layer,
        })
        self.save_checkpoint()

    def log_training_data_added(self, data_type: str, count: int,
                                  source_layer: str, total_in_bucket: int):
        """Log when points are added to training bucket."""
        self._log_event("training_data_added", {
            "type": data_type,  # 'cells' or 'non_cells'
            "count": count,
            "source_layer": source_layer,
            "total_in_bucket": total_in_bucket,
        })
        self.save_checkpoint()

    def get_session_duration(self) -> str:
        """Get human-readable session duration."""
        if not self.session_start:
            return "Not started"

        end = self.session_end or datetime.now()
        duration = end - self.session_start

        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60

        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def get_best_run(self) -> Optional[Dict[str, Any]]:
        """Get the run with best reference match (if available)."""
        runs_with_ref = [r for r in self.detection_runs if r.get("reference_match")]
        if not runs_with_ref:
            return None
        return max(runs_with_ref, key=lambda r: r["reference_match"])

    def generate_report(self) -> Path:
        """Generate a markdown session report."""
        report_path = self.output_dir / f"{self.brain_name}_{self.session_id}_session.md"

        lines = []

        # Header
        lines.append(f"# Optimization Session: {self.brain_name}")
        lines.append("")

        # Data Hierarchy
        lines.append(f"## Data Hierarchy")
        lines.append(f"")
        lines.append(f"**{self.hierarchy_string}**")
        lines.append(f"")
        if self.hierarchy.get("brain_id"):
            lines.append(f"| Level | Value |")
            lines.append(f"|-------|-------|")
            lines.append(f"| Brain ID | {self.hierarchy.get('brain_id', '')} |")
            lines.append(f"| Subject | {self.hierarchy.get('subject_full', '')} |")
            lines.append(f"| Cohort | {self.hierarchy.get('cohort_full', '')} |")
            lines.append(f"| Project | {self.hierarchy.get('project_name', '')} ({self.hierarchy.get('project_code', '')}) |")
            if self.hierarchy.get("imaging_params"):
                lines.append(f"| Imaging | {self.hierarchy.get('imaging_params', '')} |")
            lines.append(f"")

        # Session timing
        lines.append(f"## Session")
        lines.append(f"")
        lines.append(f"**Date:** {self.session_start.strftime('%Y-%m-%d')}")
        lines.append(f"**Time:** {self.session_start.strftime('%H:%M')} - {(self.session_end or datetime.now()).strftime('%H:%M')}")
        lines.append(f"**Duration:** {self.get_session_duration()}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Detection runs:** {len(self.detection_runs)}")
        lines.append(f"- **Parameter changes:** {len(self.param_changes)}")
        lines.append(f"- **Curations:** {len(self.curations)}")
        lines.append(f"- **Exports:** {len(self.exports)}")
        lines.append("")

        # Detection runs table
        if self.detection_runs:
            lines.append("## Detection Runs")
            lines.append("")
            lines.append("| # | Preset | Threshold | Ball XY | Soma | Cells | vs Ref | Z Range |")
            lines.append("|---|--------|-----------|---------|------|-------|--------|---------|")

            for i, run in enumerate(self.detection_runs, 1):
                params = run.get("params", {})
                preset = run.get("preset", "custom")
                threshold = params.get("threshold", "-")
                ball_xy = params.get("ball_xy_size", "-")
                soma = params.get("soma_diameter", "-")
                cells = run.get("cell_count", 0)
                ref_match = run.get("reference_match")
                ref_str = f"{ref_match:.0f}%" if ref_match else "-"
                z_range = run.get("z_range")
                z_str = f"{z_range[0]}-{z_range[1]}" if z_range else "full"

                lines.append(f"| {i} | {preset} | {threshold} | {ball_xy} | {soma} | {cells} | {ref_str} | {z_str} |")

            lines.append("")

            # Best run highlight
            best = self.get_best_run()
            if best:
                lines.append("### Best Run")
                lines.append("")
                lines.append(f"Run with best reference match: **{best.get('reference_match', 0):.0f}%**")
                lines.append("")
                lines.append("Parameters:")
                lines.append("```json")
                lines.append(json.dumps(best.get("params", {}), indent=2))
                lines.append("```")
                lines.append("")

        # Parameter changes
        if self.param_changes:
            lines.append("## Parameter Changes")
            lines.append("")
            lines.append("| Time | Parameter | From | To |")
            lines.append("|------|-----------|------|-----|")

            for change in self.param_changes:
                time = change.get("timestamp", "")[:16].replace("T", " ")
                param = change.get("param", "")
                old = change.get("from", "")
                new = change.get("to", "")
                lines.append(f"| {time} | {param} | {old} | {new} |")

            lines.append("")

        # Curation summary
        if self.curations:
            lines.append("## Curation Summary")
            lines.append("")
            total_confirmed = sum(c.get("confirmed", 0) for c in self.curations)
            total_rejected = sum(c.get("rejected", 0) for c in self.curations)
            total_skipped = sum(c.get("skipped", 0) for c in self.curations)

            lines.append(f"- Confirmed cells: {total_confirmed}")
            lines.append(f"- Rejected candidates: {total_rejected}")
            lines.append(f"- Skipped: {total_skipped}")
            lines.append("")

        # Exports
        if self.exports:
            lines.append("## Exports")
            lines.append("")
            for export in self.exports:
                lines.append(f"- `{export}`")
            lines.append("")

        # Notes
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        # Final state
        if self.current_params:
            lines.append("## Final Parameters")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(self.current_params, indent=2))
            lines.append("```")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Auto-generated by SCI-Connectome Session Documenter*")

        # Write report
        report_path.write_text("\n".join(lines), encoding="utf-8")

        # Also save raw JSON for programmatic access
        json_path = self.output_dir / f"{self.brain_name}_{self.session_id}_session.json"
        session_data = {
            "brain": self.brain_name,
            "session_id": self.session_id,
            # Data hierarchy (auto-parsed from brain name)
            "hierarchy": {
                "brain_id": self.hierarchy.get("brain_id"),
                "subject": self.hierarchy.get("subject_full"),
                "cohort": self.hierarchy.get("cohort_full"),
                "project_code": self.hierarchy.get("project_code"),
                "project_name": self.hierarchy.get("project_name"),
                "imaging_params": self.hierarchy.get("imaging_params"),
            },
            "start": self.session_start.isoformat() if self.session_start else None,
            "end": self.session_end.isoformat() if self.session_end else None,
            "detection_runs": self.detection_runs,
            "param_changes": self.param_changes,
            "curations": self.curations,
            "exports": self.exports,
            "notes": self.notes,
            "final_params": self.current_params,
            "events": self.events,
        }
        json_path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")

        return report_path

    def save_checkpoint(self):
        """Save current session state (for crash recovery) AND update live log."""
        if not self.session_id:
            return

        # Save JSON checkpoint (for crash recovery)
        checkpoint_path = self.output_dir / f".checkpoint_{self.brain_name}_{self.session_id}.json"
        session_data = {
            "brain": self.brain_name,
            "session_id": self.session_id,
            # Data hierarchy
            "hierarchy": {
                "brain_id": self.hierarchy.get("brain_id"),
                "subject": self.hierarchy.get("subject_full"),
                "cohort": self.hierarchy.get("cohort_full"),
                "project_code": self.hierarchy.get("project_code"),
                "project_name": self.hierarchy.get("project_name"),
                "imaging_params": self.hierarchy.get("imaging_params"),
            },
            "start": self.session_start.isoformat() if self.session_start else None,
            "detection_runs": self.detection_runs,
            "param_changes": self.param_changes,
            "curations": self.curations,
            "exports": self.exports,
            "notes": self.notes,
            "current_params": self.current_params,
            "events": self.events,
        }
        checkpoint_path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")

        # Also write human-readable live log
        self._write_live_log()

    def _write_live_log(self):
        """Write a human-readable live log file that updates in real-time."""
        if not self.session_id:
            return

        live_log_path = self.output_dir / f"LIVE_SESSION_{self.brain_name}_{self.session_id}.md"

        lines = []
        lines.append(f"# Live Session Log: {self.brain_name}")
        lines.append(f"")

        # Data Hierarchy section
        lines.append(f"## Data Hierarchy")
        lines.append(f"")
        lines.append(f"**{self.hierarchy_string}**")
        lines.append(f"")
        if self.hierarchy.get("brain_id"):
            lines.append(f"| Level | Value |")
            lines.append(f"|-------|-------|")
            lines.append(f"| Brain ID | {self.hierarchy.get('brain_id', '')} |")
            lines.append(f"| Subject | {self.hierarchy.get('subject_full', '')} |")
            lines.append(f"| Cohort | {self.hierarchy.get('cohort_full', '')} |")
            lines.append(f"| Project | {self.hierarchy.get('project_name', '')} ({self.hierarchy.get('project_code', '')}) |")
            if self.hierarchy.get("imaging_params"):
                lines.append(f"| Imaging | {self.hierarchy.get('imaging_params', '')} |")
            lines.append(f"")

        # Session info
        lines.append(f"## Session Info")
        lines.append(f"")
        lines.append(f"**Session ID:** {self.session_id}")
        lines.append(f"**Started:** {self.session_start.strftime('%Y-%m-%d %H:%M:%S') if self.session_start else 'N/A'}")
        lines.append(f"**Duration:** {self.get_session_duration()}")
        lines.append(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Current parameters
        if self.current_params:
            lines.append(f"## Current Parameters")
            for key, val in self.current_params.items():
                lines.append(f"- **{key}:** {val}")
            lines.append(f"")

        # Detection runs
        if self.detection_runs:
            lines.append(f"## Detection Runs ({len(self.detection_runs)} total)")
            for i, run in enumerate(self.detection_runs, 1):
                lines.append(f"")
                lines.append(f"### Run {i}: {run.get('run_id', 'unknown')}")
                lines.append(f"- **Cells found:** {run.get('cell_count', 'N/A')}")
                lines.append(f"- **Z range:** {run.get('z_range', 'full')}")
                lines.append(f"- **Time:** {run.get('timestamp', 'N/A')}")
                if run.get('params'):
                    lines.append(f"- **Parameters:**")
                    for k, v in run['params'].items():
                        lines.append(f"  - {k}: {v}")
            lines.append(f"")

        # Classification runs (from events)
        classification_events = [e for e in self.events if e.get('type') == 'classification_run']
        if classification_events:
            lines.append(f"## Classification Runs ({len(classification_events)} total)")
            for i, evt in enumerate(classification_events, 1):
                data = evt.get('data', {})
                lines.append(f"")
                lines.append(f"### Classification {i}: {data.get('run_id', 'unknown')}")
                lines.append(f"- **Model:** {data.get('model', 'default')}")
                lines.append(f"- **Cells found:** {data.get('cells_found', 0)}")
                lines.append(f"- **Rejected:** {data.get('rejected', 0)}")
                lines.append(f"- **Acceptance rate:** {data.get('acceptance_rate', 0):.1f}%")
                lines.append(f"- **Time:** {data.get('timestamp', 'N/A')}")
            lines.append(f"")

        # Curation
        if self.curations:
            lines.append(f"## Curation Sessions ({len(self.curations)} total)")
            for i, cur in enumerate(self.curations, 1):
                lines.append(f"- Session {i}: **{cur.get('confirmed', 0)} confirmed**, {cur.get('rejected', 0)} rejected, {cur.get('skipped', 0)} skipped ({cur.get('timestamp', '')})")
            lines.append(f"")

        # Exports
        if self.exports:
            lines.append(f"## Exports ({len(self.exports)} total)")
            for exp in self.exports:
                lines.append(f"- {exp}")
            lines.append(f"")

        # Notes
        if self.notes:
            lines.append(f"## Notes")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append(f"")

        # Event timeline
        lines.append(f"## Event Timeline")
        lines.append(f"")
        for event in self.events[-50:]:  # Last 50 events
            ts = event.get('timestamp', '')
            if ts:
                ts = ts.split('T')[1].split('.')[0]  # Just time
            evt_type = event.get('type', 'unknown')
            data = event.get('data', {})

            if evt_type == 'session_start':
                lines.append(f"- [{ts}] **SESSION STARTED** for {data.get('brain', '')}")
            elif evt_type == 'brain_loaded':
                lines.append(f"- [{ts}] Loaded brain from {data.get('source', 'unknown')}")
            elif evt_type == 'detection_run':
                lines.append(f"- [{ts}] Detection run: found **{data.get('cell_count', '?')} cells**")
            elif evt_type == 'param_change':
                lines.append(f"- [{ts}] Changed {data.get('param', '?')}: {data.get('from', '?')} \u2192 {data.get('to', '?')}")
            elif evt_type == 'classification_run':
                lines.append(f"- [{ts}] Classification: **{data.get('cells_found', 0)} cells**, {data.get('rejected', 0)} rejected ({data.get('acceptance_rate', 0):.0f}% acceptance)")
            elif evt_type == 'curation':
                lines.append(f"- [{ts}] Curation: {data.get('confirmed', 0)} confirmed, {data.get('rejected', 0)} rejected")
            elif evt_type == 'export':
                lines.append(f"- [{ts}] Exported: {data.get('path', 'unknown')}")
            elif evt_type == 'note':
                lines.append(f"- [{ts}] NOTE: {data.get('text', '')}")
            else:
                lines.append(f"- [{ts}] {evt_type}")

        lines.append(f"")
        lines.append(f"---")
        lines.append(f"*This log updates automatically. Open in any text editor to view.*")

        live_log_path.write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, brain_name: str, session_id: str,
                        output_dir: Optional[Path] = None) -> Optional["SessionDocumenter"]:
        """Load a session from checkpoint."""
        if output_dir is None:
            try:
                from mousebrain.config import DATA_SUMMARY_DIR
                project_sessions = DATA_SUMMARY_DIR / "sessions"
            except ImportError:
                project_sessions = Path("Y:/2_Connectome/Tissue/3D_Cleared/2_Data_Summary/sessions")
            if project_sessions.exists():
                output_dir = project_sessions
            else:
                output_dir = Path.home() / ".sci_connectome" / "sessions"
        checkpoint_path = output_dir / f".checkpoint_{brain_name}_{session_id}.json"

        if not checkpoint_path.exists():
            return None

        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))

            doc = cls(brain_name, output_dir)
            doc.session_id = data.get("session_id")
            doc.session_start = datetime.fromisoformat(data["start"]) if data.get("start") else None
            doc.detection_runs = data.get("detection_runs", [])
            doc.param_changes = data.get("param_changes", [])
            doc.curations = data.get("curations", [])
            doc.exports = data.get("exports", [])
            doc.notes = data.get("notes", [])
            doc.current_params = data.get("current_params", {})
            doc.events = data.get("events", [])

            return doc
        except Exception:
            return None


class SessionDocumenterWidget:
    """
    Mixin or helper to add session documentation to a widget.

    Usage in TuningWidget:
        self.session_doc = SessionDocumenterWidget(self)
        self.session_doc.start_for_brain("349_CNT_01_02")

        # When parameter changes:
        self.session_doc.log_param_change("threshold", old, new)

        # On widget close:
        self.session_doc.end_session()
    """

    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.documenter: Optional[SessionDocumenter] = None

    def start_for_brain(self, brain_name: str):
        """Start a new session for the given brain."""
        # End any existing session
        if self.documenter:
            self.end_session()

        self.documenter = SessionDocumenter(brain_name)
        self.documenter.start_session()

    def end_session(self) -> Optional[Path]:
        """End current session and generate report."""
        if not self.documenter:
            return None

        report_path = self.documenter.end_session()
        self.documenter = None
        return report_path

    def log_param_change(self, param: str, old_value: Any, new_value: Any):
        """Log a parameter change."""
        if self.documenter:
            self.documenter.log_param_change(param, old_value, new_value)
            self.documenter.save_checkpoint()

    def log_detection_run(self, run_id: str, params: Dict[str, Any],
                          cell_count: int, **kwargs):
        """Log a detection run."""
        if self.documenter:
            self.documenter.log_detection_run(run_id, params, cell_count, **kwargs)
            self.documenter.save_checkpoint()

    def log_brain_loaded(self, source: str):
        """Log brain loading."""
        if self.documenter:
            self.documenter.log_brain_loaded(source)

    def log_curation(self, confirmed: int, rejected: int, skipped: int = 0):
        """Log curation results."""
        if self.documenter:
            self.documenter.log_curation(confirmed, rejected, skipped)
            self.documenter.save_checkpoint()

    def log_export(self, filepath: str, export_type: str = "results"):
        """Log an export."""
        if self.documenter:
            self.documenter.log_export(filepath, export_type)

    def add_note(self, note: str):
        """Add a note."""
        if self.documenter:
            self.documenter.add_note(note)
            self.documenter.save_checkpoint()

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a generic event."""
        if self.documenter:
            self.documenter.log_event(event_type, data)

    def log_region_count(self, brain_name: str, total_cells: int = 0,
                          regions_counted: int = 0, output_path: str = ""):
        """Log a region counting run."""
        if self.documenter:
            self.documenter._log_event("region_count", {
                "brain": brain_name,
                "total_cells": total_cells,
                "regions_counted": regions_counted,
                "output_path": output_path,
            })
            self.documenter.save_checkpoint()

    def log_classification_run(self, run_id: str, model: str,
                                cells_found: int, rejected: int,
                                cube_size: int = 50, batch_size: int = 32):
        """Log a classification run."""
        if self.documenter:
            self.documenter.log_classification_run(
                run_id, model, cells_found, rejected, cube_size, batch_size
            )

    def is_active(self) -> bool:
        """Check if a session is active."""
        return self.documenter is not None
