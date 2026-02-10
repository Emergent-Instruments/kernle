#!/usr/bin/env python3
"""Dev Dashboard — Local Memory Stack Inspector.

A self-contained dev tool for visually inspecting kernle memory stacks.
Uses only stdlib (http.server, json) plus kernle itself.

Usage:
    python dev/dashboard.py --stack corpus-test
    # → opens http://localhost:8420
"""

import argparse
import dataclasses
import json
import logging
import re
import sys
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# Add parent dir to path so kernle is importable when running from dev/
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from kernle import Kernle

logger = logging.getLogger(__name__)


def serialize(obj):
    """Convert kernle dataclasses/objects to JSON-serializable dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: serialize(v) for k, v in dataclasses.asdict(obj).items()}
    return str(obj)


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    kernle_instance: Kernle = None  # Set by server setup

    def log_message(self, format, *args):
        logger.info(format, *args)

    def _send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, message):
        self._send_json({"error": message}, status)

    def _get_int_param(self, params, key, default):
        try:
            vals = params.get(key, [])
            return int(vals[0]) if vals else default
        except (ValueError, IndexError):
            return default

    def _get_bool_param(self, params, key):
        """Return True, False, or None if not specified."""
        vals = params.get(key, [])
        if not vals or vals[0] == "":
            return None
        return vals[0].lower() in ("true", "1", "yes")

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query, keep_blank_values=True)
        k = self.__class__.kernle_instance

        try:
            if path == "" or path == "/":
                return self._send_html(DASHBOARD_HTML)

            elif path == "/api/stats":
                stats = k.stack.get_stats()
                return self._send_json(serialize(stats))

            elif path == "/api/anxiety":
                report = k.get_anxiety_report(detailed=True)
                return self._send_json(serialize(report))

            elif path == "/api/raw":
                limit = self._get_int_param(params, "limit", 200)
                processed = self._get_bool_param(params, "processed")
                entries = k.stack.list_raw(processed=processed, limit=limit)
                return self._send_json(serialize(entries))

            elif m := re.match(r"^/api/raw/([^/]+)$", path):
                raw_id = m.group(1)
                entry = k.stack.get_raw(raw_id)
                if entry is None:
                    return self._send_error(404, "Not found")
                return self._send_json(serialize(entry))

            elif path == "/api/episodes":
                limit = self._get_int_param(params, "limit", 100)
                items = k.stack.get_episodes(limit=limit, include_forgotten=True, include_weak=True)
                return self._send_json(serialize(items))

            elif path == "/api/beliefs":
                limit = self._get_int_param(params, "limit", 100)
                items = k.stack.get_beliefs(limit=limit, include_forgotten=True, include_weak=True)
                return self._send_json(serialize(items))

            elif path == "/api/values":
                limit = self._get_int_param(params, "limit", 100)
                items = k.stack.get_values(limit=limit, include_forgotten=True, include_weak=True)
                return self._send_json(serialize(items))

            elif path == "/api/goals":
                limit = self._get_int_param(params, "limit", 100)
                items = k.stack.get_goals(limit=limit, include_forgotten=True, include_weak=True)
                return self._send_json(serialize(items))

            elif path == "/api/notes":
                limit = self._get_int_param(params, "limit", 100)
                items = k.stack.get_notes(limit=limit, include_forgotten=True, include_weak=True)
                return self._send_json(serialize(items))

            elif path == "/api/relationships":
                items = k.stack.get_relationships()
                return self._send_json(serialize(items))

            elif path == "/api/drives":
                items = k.stack.get_drives()
                return self._send_json(serialize(items))

            elif path == "/api/suggestions":
                limit = self._get_int_param(params, "limit", 100)
                items = k.stack.get_suggestions(limit=limit)
                return self._send_json(serialize(items))

            elif path == "/api/processing":
                config = k.stack.get_processing_config()
                return self._send_json(serialize(config))

            elif m := re.match(r"^/api/provenance/(\w+)/([^/]+)$", path):
                mem_type, mem_id = m.group(1), m.group(2)
                valid_types = {
                    "raw",
                    "episode",
                    "belief",
                    "value",
                    "goal",
                    "note",
                    "drive",
                    "relationship",
                }
                if mem_type not in valid_types:
                    return self._send_error(400, "Invalid memory type")
                derived = k.stack.get_memories_derived_from(mem_type, mem_id)
                result = [{"type": t, "id": i} for t, i in derived]
                return self._send_json(result)

            elif path == "/api/audit":
                limit = self._get_int_param(params, "limit", 100)
                items = k.stack.get_audit_log(limit=limit)
                return self._send_json(serialize(items))

            elif path == "/api/settings":
                settings = k.stack.get_all_stack_settings()
                return self._send_json(serialize(settings))

            else:
                return self._send_error(404, "Not found")

        except Exception:
            logger.exception("Error handling %s", path)
            return self._send_error(500, "Internal server error")


# ---------------------------------------------------------------------------
# Embedded HTML/CSS/JS Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kernle Dashboard</title>
<style>
:root {
  --bg: #0d1117;
  --bg2: #161b22;
  --bg3: #21262d;
  --border: #30363d;
  --text: #c9d1d9;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --yellow: #d29922;
  --red: #f85149;
  --orange: #db6d28;
  --purple: #bc8cff;
  --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Header */
.header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 20px;
  position: sticky;
  top: 0;
  z-index: 100;
}
.header h1 { font-size: 16px; font-weight: 600; color: var(--text); white-space: nowrap; }
.header .stack-id { font-family: var(--mono); color: var(--accent); font-size: 13px; }
.header .stats-bar { display: flex; gap: 12px; font-size: 12px; color: var(--text-dim); flex-wrap: wrap; }
.header .stats-bar .stat { display: flex; align-items: center; gap: 4px; }
.header .stats-bar .stat-val { color: var(--text); font-weight: 600; font-family: var(--mono); }
.anxiety-badge {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
  font-family: var(--mono);
  white-space: nowrap;
}
.anxiety-calm { background: #0d2818; color: var(--green); }
.anxiety-aware { background: #1c1d00; color: var(--yellow); }
.anxiety-elevated { background: #291800; color: var(--orange); }
.anxiety-high { background: #2d0000; color: var(--red); }
.anxiety-critical { background: #1a0020; color: var(--purple); }
.header-right {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 12px;
}
.auto-refresh-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-dim);
  cursor: pointer;
  user-select: none;
}
.auto-refresh-toggle input { cursor: pointer; }

/* Tabs */
.tabs {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  display: flex;
  padding: 0 24px;
  gap: 0;
  overflow-x: auto;
}
.tab {
  padding: 10px 16px;
  font-size: 13px;
  color: var(--text-dim);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  white-space: nowrap;
  transition: color 0.15s, border-color 0.15s;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--text); border-bottom-color: var(--accent); }

/* Main */
.main { padding: 24px; max-width: 1400px; margin: 0 auto; }

/* Cards */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}
.card .card-val { font-size: 28px; font-weight: 700; font-family: var(--mono); color: var(--accent); }
.card .card-label { font-size: 12px; color: var(--text-dim); margin-top: 4px; text-transform: capitalize; }

/* Tables */
.table-wrap { overflow-x: auto; margin-bottom: 24px; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
thead th {
  background: var(--bg3);
  padding: 8px 12px;
  text-align: left;
  font-weight: 600;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  white-space: nowrap;
}
tbody td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
tbody tr { cursor: pointer; transition: background 0.1s; }
tbody tr:hover { background: var(--bg3); }
.id-cell { font-family: var(--mono); font-size: 11px; color: var(--text-dim); max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.blob-preview { max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-dim); font-family: var(--mono); font-size: 12px; }
.status-icon { font-size: 16px; }
.strength-bar { display: inline-block; width: 50px; height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; vertical-align: middle; }
.strength-fill { height: 100%; border-radius: 3px; }

/* Filters */
.filters { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
.filters select, .filters input {
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 13px;
}
.filters label { font-size: 12px; color: var(--text-dim); }

/* Sub-tabs */
.sub-tabs { display: flex; gap: 0; margin-bottom: 16px; border-bottom: 1px solid var(--border); }
.sub-tab {
  padding: 8px 14px;
  font-size: 12px;
  color: var(--text-dim);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color 0.15s;
}
.sub-tab:hover { color: var(--text); }
.sub-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* Detail Panel */
.detail-panel {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 24px;
  display: none;
}
.detail-panel.open { display: block; }
.detail-panel h3 { font-size: 14px; margin-bottom: 12px; color: var(--accent); }
.detail-panel .close-btn {
  float: right;
  background: none;
  border: none;
  color: var(--text-dim);
  cursor: pointer;
  font-size: 18px;
}
.detail-panel .close-btn:hover { color: var(--text); }
.detail-kv { display: grid; grid-template-columns: 140px 1fr; gap: 4px 12px; font-size: 13px; }
.detail-kv .dk { color: var(--text-dim); font-weight: 600; }
.detail-kv .dv { font-family: var(--mono); font-size: 12px; word-break: break-all; }
.detail-kv .dv.blob-full { white-space: pre-wrap; max-height: 300px; overflow-y: auto; background: var(--bg); padding: 8px; border-radius: 4px; }

/* Anxiety dimensions */
.anxiety-dims { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-bottom: 24px; }
.anxiety-dim {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
}
.anxiety-dim .dim-name { font-size: 13px; font-weight: 600; margin-bottom: 6px; text-transform: capitalize; }
.anxiety-dim .dim-bar { width: 100%; height: 8px; background: var(--bg); border-radius: 4px; overflow: hidden; margin-bottom: 4px; }
.anxiety-dim .dim-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.anxiety-dim .dim-detail { font-size: 11px; color: var(--text-dim); }

/* Processing config */
.proc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
.proc-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
}
.proc-card .pc-name { font-family: var(--mono); font-size: 13px; color: var(--accent); margin-bottom: 8px; }
.proc-card .pc-row { display: flex; justify-content: space-between; font-size: 12px; padding: 2px 0; }
.proc-card .pc-key { color: var(--text-dim); }
.proc-card .pc-val { font-family: var(--mono); }

/* Settings table */
.settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); border-radius: 8px; overflow: hidden; }
.settings-grid .sg-cell { background: var(--bg2); padding: 10px 14px; font-size: 13px; }
.settings-grid .sg-key { font-weight: 600; color: var(--text-dim); }
.settings-grid .sg-val { font-family: var(--mono); }

/* Section headings */
.section-heading { font-size: 15px; font-weight: 600; margin-bottom: 12px; color: var(--text); }

/* Provenance */
.prov-chain { margin-top: 12px; padding: 8px; background: var(--bg); border-radius: 4px; }
.prov-item { font-family: var(--mono); font-size: 12px; padding: 2px 0; }
.prov-item a { color: var(--accent); }

/* Loading */
.loading { text-align: center; padding: 40px; color: var(--text-dim); }

/* Hidden */
.hidden { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--border); }
</style>
</head>
<body>

<div class="header">
  <h1>Kernle Dashboard</h1>
  <span class="stack-id" id="stack-id">loading...</span>
  <div class="stats-bar" id="stats-bar"></div>
  <div id="anxiety-badge-wrap"></div>
  <div class="header-right">
    <label class="auto-refresh-toggle">
      <input type="checkbox" id="auto-refresh"> Auto-refresh
    </label>
  </div>
</div>

<div class="tabs" id="tabs">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="raw">Raw Entries</div>
  <div class="tab" data-tab="memories">Memories</div>
  <div class="tab" data-tab="suggestions">Suggestions</div>
  <div class="tab" data-tab="audit">Audit Log</div>
  <div class="tab" data-tab="settings">Settings</div>
</div>

<div class="main" id="main">
  <!-- Overview Tab -->
  <div id="tab-overview">
    <div class="section-heading">Memory Counts</div>
    <div class="cards" id="stats-cards"></div>
    <div class="section-heading">Anxiety Dimensions</div>
    <div class="anxiety-dims" id="anxiety-dims"></div>
    <div class="section-heading">Processing Pipeline</div>
    <div class="proc-grid" id="proc-grid"></div>
  </div>

  <!-- Raw Entries Tab -->
  <div id="tab-raw" class="hidden">
    <div class="filters">
      <label>Filter:</label>
      <select id="raw-filter">
        <option value="">All</option>
        <option value="false">Unprocessed</option>
        <option value="true">Processed</option>
      </select>
      <label>Limit:</label>
      <select id="raw-limit">
        <option value="50">50</option>
        <option value="100">100</option>
        <option value="200" selected>200</option>
        <option value="500">500</option>
      </select>
    </div>
    <div class="detail-panel" id="raw-detail">
      <button class="close-btn" onclick="closeDetail('raw-detail')">&times;</button>
      <h3>Raw Entry Detail</h3>
      <div class="detail-kv" id="raw-detail-kv"></div>
      <div class="prov-chain" id="raw-prov"></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Status</th><th>ID</th><th>Captured</th><th>Source</th><th>Blob</th><th>Processed Into</th><th>Strength</th>
        </tr></thead>
        <tbody id="raw-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Memories Tab -->
  <div id="tab-memories" class="hidden">
    <div class="sub-tabs" id="mem-sub-tabs">
      <div class="sub-tab active" data-mem="episodes">Episodes</div>
      <div class="sub-tab" data-mem="beliefs">Beliefs</div>
      <div class="sub-tab" data-mem="values">Values</div>
      <div class="sub-tab" data-mem="goals">Goals</div>
      <div class="sub-tab" data-mem="notes">Notes</div>
      <div class="sub-tab" data-mem="relationships">Relationships</div>
      <div class="sub-tab" data-mem="drives">Drives</div>
    </div>
    <div class="detail-panel" id="mem-detail">
      <button class="close-btn" onclick="closeDetail('mem-detail')">&times;</button>
      <h3 id="mem-detail-title">Memory Detail</h3>
      <div class="detail-kv" id="mem-detail-kv"></div>
      <div class="prov-chain" id="mem-prov"></div>
    </div>
    <div class="table-wrap" id="mem-table-wrap"></div>
  </div>

  <!-- Suggestions Tab -->
  <div id="tab-suggestions" class="hidden">
    <div class="detail-panel" id="sug-detail">
      <button class="close-btn" onclick="closeDetail('sug-detail')">&times;</button>
      <h3>Suggestion Detail</h3>
      <div class="detail-kv" id="sug-detail-kv"></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Status</th><th>Type</th><th>Confidence</th><th>Content Preview</th><th>Source Raw IDs</th><th>Created</th><th>Promoted To</th>
        </tr></thead>
        <tbody id="sug-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Audit Tab -->
  <div id="tab-audit" class="hidden">
    <div class="filters">
      <label>Limit:</label>
      <select id="audit-limit">
        <option value="50">50</option>
        <option value="100" selected>100</option>
        <option value="200">200</option>
        <option value="500">500</option>
      </select>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Timestamp</th><th>Operation</th><th>Memory Type</th><th>Memory ID</th><th>Actor</th><th>Details</th>
        </tr></thead>
        <tbody id="audit-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Settings Tab -->
  <div id="tab-settings" class="hidden">
    <div class="section-heading">Stack Settings</div>
    <div class="settings-grid" id="settings-grid"></div>
    <div class="section-heading" style="margin-top:24px">Processing Config</div>
    <div class="proc-grid" id="settings-proc-grid"></div>
  </div>
</div>

<script>
// ---- State ----
let currentTab = 'overview';
let currentMemType = 'episodes';
let refreshTimer = null;
const cache = {};

// ---- API ----
async function api(endpoint) {
  const res = await fetch(endpoint);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

// ---- Helpers ----
function esc(s) {
  if (s == null) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function truncId(id) {
  if (!id) return '';
  return id.length > 12 ? id.slice(0, 12) + '...' : id;
}

function fmtDate(d) {
  if (!d) return '';
  try { return new Date(d).toLocaleString(); } catch { return String(d); }
}

function strengthColor(s) {
  if (s >= 0.8) return 'var(--green)';
  if (s >= 0.5) return 'var(--yellow)';
  if (s > 0) return 'var(--orange)';
  return 'var(--red)';
}

function strengthBar(s) {
  const pct = Math.round((s || 0) * 100);
  return `<span class="strength-bar"><span class="strength-fill" style="width:${pct}%;background:${strengthColor(s)}"></span></span> ${pct}%`;
}

function statusIcon(processed) {
  return processed ? '<span class="status-icon" title="Processed" style="color:var(--green)">&#9679;</span>' :
                     '<span class="status-icon" title="Unprocessed" style="color:var(--yellow)">&#9675;</span>';
}

function anxietyClass(level) {
  const l = (level || '').toLowerCase();
  if (l === 'calm') return 'anxiety-calm';
  if (l === 'aware') return 'anxiety-aware';
  if (l === 'elevated') return 'anxiety-elevated';
  if (l === 'high') return 'anxiety-high';
  return 'anxiety-critical';
}

function dimColor(score) {
  if (score < 25) return 'var(--green)';
  if (score < 50) return 'var(--yellow)';
  if (score < 75) return 'var(--orange)';
  return 'var(--red)';
}

function closeDetail(id) {
  document.getElementById(id).classList.remove('open');
}

// ---- Tab Navigation ----
document.getElementById('tabs').addEventListener('click', e => {
  const tab = e.target.closest('.tab');
  if (!tab) return;
  currentTab = tab.dataset.tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
  document.querySelectorAll('#main > div').forEach(d => {
    d.classList.toggle('hidden', d.id !== 'tab-' + currentTab);
  });
  loadTab(currentTab);
});

// Memory sub-tabs
document.getElementById('mem-sub-tabs').addEventListener('click', e => {
  const st = e.target.closest('.sub-tab');
  if (!st) return;
  currentMemType = st.dataset.mem;
  document.querySelectorAll('#mem-sub-tabs .sub-tab').forEach(t => t.classList.toggle('active', t === st));
  closeDetail('mem-detail');
  loadMemories(currentMemType);
});

// Raw filter/limit
document.getElementById('raw-filter').addEventListener('change', () => loadRaw());
document.getElementById('raw-limit').addEventListener('change', () => loadRaw());
document.getElementById('audit-limit').addEventListener('change', () => loadAudit());

// Auto-refresh
document.getElementById('auto-refresh').addEventListener('change', e => {
  if (e.target.checked) {
    refreshTimer = setInterval(() => loadTab(currentTab, true), 5000);
  } else {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
});

// ---- Loaders ----
async function loadHeader() {
  try {
    const [stats, anxiety] = await Promise.all([api('/api/stats'), api('/api/anxiety')]);
    cache.stats = stats;
    cache.anxiety = anxiety;

    document.getElementById('stack-id').textContent = anxiety.stack_id || 'unknown';

    const bar = document.getElementById('stats-bar');
    const types = ['raw', 'episodes', 'beliefs', 'values', 'goals', 'notes', 'relationships', 'drives', 'suggestions'];
    bar.innerHTML = types.map(t =>
      `<span class="stat">${t}: <span class="stat-val">${stats[t] || 0}</span></span>`
    ).join('');

    const badge = document.getElementById('anxiety-badge-wrap');
    const cls = anxietyClass(anxiety.overall_level);
    badge.innerHTML = `<span class="anxiety-badge ${cls}">${anxiety.overall_emoji || ''} ${anxiety.overall_score || 0} ${anxiety.overall_level || 'Unknown'}</span>`;
  } catch (e) {
    console.error('Failed to load header:', e);
  }
}

async function loadTab(tab, silent) {
  if (!silent) loadHeader().catch(e => console.error('Header load failed:', e));
  switch (tab) {
    case 'overview': return loadOverview();
    case 'raw': return loadRaw();
    case 'memories': return loadMemories(currentMemType);
    case 'suggestions': return loadSuggestions();
    case 'audit': return loadAudit();
    case 'settings': return loadSettings();
  }
}

async function loadOverview() {
  try {
    const [stats, anxiety, processing] = await Promise.all([
      api('/api/stats'),
      api('/api/anxiety'),
      api('/api/processing'),
    ]);

    // Stats cards
    const cards = document.getElementById('stats-cards');
    const types = [
      ['raw', 'Raw Entries'], ['episodes', 'Episodes'], ['beliefs', 'Beliefs'],
      ['values', 'Values'], ['goals', 'Goals'], ['notes', 'Notes'],
      ['relationships', 'Relationships'], ['drives', 'Drives'],
      ['suggestions', 'Suggestions'], ['pending_suggestions', 'Pending'],
    ];
    cards.innerHTML = types.map(([k, label]) =>
      `<div class="card"><div class="card-val">${stats[k] || 0}</div><div class="card-label">${label}</div></div>`
    ).join('');

    // Anxiety dimensions
    const dims = document.getElementById('anxiety-dims');
    if (anxiety.dimensions) {
      dims.innerHTML = Object.entries(anxiety.dimensions).map(([name, dim]) => {
        const score = dim.score || 0;
        const detail = dim.detail || '';
        const displayName = name.replace(/_/g, ' ');
        return `<div class="anxiety-dim">
          <div class="dim-name">${esc(displayName)} <span style="float:right;font-family:var(--mono);font-size:12px;color:${dimColor(score)}">${score}</span></div>
          <div class="dim-bar"><div class="dim-fill" style="width:${score}%;background:${dimColor(score)}"></div></div>
          <div class="dim-detail">${esc(detail)}</div>
        </div>`;
      }).join('');
    }

    // Processing
    const pg = document.getElementById('proc-grid');
    pg.innerHTML = processing.map(p => `<div class="proc-card">
      <div class="pc-name">${esc(p.layer_transition || 'unknown')}</div>
      <div class="pc-row"><span class="pc-key">Enabled</span><span class="pc-val" style="color:${p.enabled ? 'var(--green)' : 'var(--red)'}">${p.enabled ? 'Yes' : 'No'}</span></div>
      <div class="pc-row"><span class="pc-key">Model</span><span class="pc-val">${esc(p.model_id || 'none')}</span></div>
      <div class="pc-row"><span class="pc-key">Batch Size</span><span class="pc-val">${p.batch_size || '-'}</span></div>
      <div class="pc-row"><span class="pc-key">Qty Threshold</span><span class="pc-val">${p.quantity_threshold || '-'}</span></div>
      <div class="pc-row"><span class="pc-key">Time Threshold</span><span class="pc-val">${p.time_threshold_hours || '-'}h</span></div>
    </div>`).join('');
  } catch (e) {
    console.error('Failed to load overview:', e);
  }
}

async function loadRaw() {
  try {
    const filter = document.getElementById('raw-filter').value;
    const limit = document.getElementById('raw-limit').value;
    let url = `/api/raw?limit=${limit}`;
    if (filter !== '') url += `&processed=${filter}`;
    const entries = await api(url);

    const tbody = document.getElementById('raw-tbody');
    tbody.innerHTML = entries.map(e => `<tr data-raw-id="${esc(e.id)}">
      <td>${statusIcon(e.processed)}</td>
      <td class="id-cell" title="${esc(e.id)}">${esc(truncId(e.id))}</td>
      <td>${esc(fmtDate(e.captured_at))}</td>
      <td>${esc(e.source || '')}</td>
      <td class="blob-preview" title="${esc((e.blob || e.content || '').slice(0, 300))}">${esc((e.blob || e.content || '').slice(0, 120))}</td>
      <td style="font-family:var(--mono);font-size:11px">${esc((e.processed_into || []).join(', '))}</td>
      <td>${strengthBar(e.strength != null ? e.strength : 1.0)}</td>
    </tr>`).join('');

    tbody.querySelectorAll('tr').forEach(tr => {
      tr.addEventListener('click', () => showRawDetail(tr.dataset.rawId, entries));
    });
  } catch (e) {
    console.error('Failed to load raw:', e);
  }
}

async function showRawDetail(rawId, entries) {
  const entry = entries.find(e => e.id === rawId);
  if (!entry) return;

  const panel = document.getElementById('raw-detail');
  const kv = document.getElementById('raw-detail-kv');
  const fields = [
    ['ID', entry.id],
    ['Captured At', fmtDate(entry.captured_at)],
    ['Source', entry.source],
    ['Source Type', entry.source_type],
    ['Processed', entry.processed ? 'Yes' : 'No'],
    ['Processed Into', (entry.processed_into || []).join(', ') || 'none'],
    ['Strength', entry.strength != null ? entry.strength : 1.0],
    ['Blob', entry.blob || entry.content || ''],
  ];
  kv.innerHTML = fields.map(([k, v]) =>
    `<span class="dk">${esc(k)}</span><span class="dv${k === 'Blob' ? ' blob-full' : ''}">${esc(v)}</span>`
  ).join('');

  // Load provenance
  const prov = document.getElementById('raw-prov');
  try {
    const derived = await api(`/api/provenance/raw/${rawId}`);
    if (derived.length > 0) {
      prov.innerHTML = '<strong>Derived memories:</strong><br>' +
        derived.map(d => `<div class="prov-item">${esc(d.type)}:${esc(truncId(d.id))}</div>`).join('');
    } else {
      prov.innerHTML = '<em style="color:var(--text-dim)">No derived memories</em>';
    }
  } catch {
    prov.innerHTML = '';
  }

  panel.classList.add('open');
}

// Memory type table definitions
const memColumns = {
  episodes: {
    cols: ['ID', 'Objective', 'Outcome', 'Confidence', 'Strength', 'Emotional', 'Created'],
    row: e => [truncId(e.id), (e.objective||'').slice(0,80), (e.outcome||'').slice(0,60),
               (e.confidence||0).toFixed(2), null, `${(e.emotional_valence||0).toFixed(1)}v ${(e.emotional_arousal||0).toFixed(1)}a`, fmtDate(e.created_at)],
    strengthIdx: 4, idField: 'id', typeLabel: 'episode'
  },
  beliefs: {
    cols: ['ID', 'Statement', 'Type', 'Confidence', 'Strength', 'Scope', 'Created'],
    row: e => [truncId(e.id), (e.statement||'').slice(0,100), e.belief_type, (e.confidence||0).toFixed(2), null, e.belief_scope, fmtDate(e.created_at)],
    strengthIdx: 4, idField: 'id', typeLabel: 'belief'
  },
  values: {
    cols: ['ID', 'Name', 'Statement', 'Priority', 'Confidence', 'Strength', 'Created'],
    row: e => [truncId(e.id), e.name, (e.statement||'').slice(0,80), e.priority, (e.confidence||0).toFixed(2), null, fmtDate(e.created_at)],
    strengthIdx: 5, idField: 'id', typeLabel: 'value'
  },
  goals: {
    cols: ['ID', 'Title', 'Type', 'Priority', 'Status', 'Strength', 'Created'],
    row: e => [truncId(e.id), (e.title||'').slice(0,80), e.goal_type, e.priority, e.status, null, fmtDate(e.created_at)],
    strengthIdx: 5, idField: 'id', typeLabel: 'goal'
  },
  notes: {
    cols: ['ID', 'Content', 'Type', 'Confidence', 'Strength', 'Processed', 'Created'],
    row: e => [truncId(e.id), (e.content||'').slice(0,100), e.note_type, (e.confidence||0).toFixed(2), null, {__safeHtml: statusIcon(e.processed)}, fmtDate(e.created_at)],
    strengthIdx: 4, idField: 'id', typeLabel: 'note'
  },
  relationships: {
    cols: ['ID', 'Entity', 'Type', 'Relationship', 'Sentiment', 'Strength', 'Interactions'],
    row: e => [truncId(e.id), e.entity_name, e.entity_type, e.relationship_type, (e.sentiment||0).toFixed(2), null, e.interaction_count],
    strengthIdx: 5, idField: 'id', typeLabel: 'relationship'
  },
  drives: {
    cols: ['ID', 'Type', 'Intensity', 'Strength', 'Focus', 'Created'],
    row: e => [truncId(e.id), e.drive_type, (e.intensity||0).toFixed(2), null, (e.focus_areas||[]).join(', '), fmtDate(e.created_at)],
    strengthIdx: 3, idField: 'id', typeLabel: 'drive'
  },
};

async function loadMemories(memType) {
  try {
    const data = await api(`/api/${memType}`);
    const def = memColumns[memType];
    if (!def) return;

    const wrap = document.getElementById('mem-table-wrap');
    const thead = def.cols.map(c => `<th>${esc(c)}</th>`).join('');
    const rows = data.map(item => {
      const cells = def.row(item);
      return `<tr data-mem-id="${esc(item[def.idField])}" data-mem-type="${esc(def.typeLabel)}">` +
        cells.map((c, i) => {
          if (i === def.strengthIdx && c === null) return `<td>${strengthBar(item.strength != null ? item.strength : 1.0)}</td>`;
          if (i === 0) return `<td class="id-cell" title="${esc(item[def.idField])}">${esc(c)}</td>`;
          if (c && typeof c === 'object' && c.__safeHtml) return `<td>${c.__safeHtml}</td>`;
          return `<td>${esc(c)}</td>`;
        }).join('') +
        '</tr>';
    }).join('');

    wrap.innerHTML = `<table><thead><tr>${thead}</tr></thead><tbody>${rows}</tbody></table>`;

    wrap.querySelectorAll('tr[data-mem-id]').forEach(tr => {
      tr.addEventListener('click', () => showMemDetail(tr.dataset.memType, tr.dataset.memId, data));
    });
  } catch (e) {
    console.error('Failed to load memories:', e);
  }
}

async function showMemDetail(memType, memId, data) {
  const item = data.find(d => d.id === memId);
  if (!item) return;

  document.getElementById('mem-detail-title').textContent = `${memType} Detail`;
  const kv = document.getElementById('mem-detail-kv');

  const skip = new Set(['local_updated_at', 'cloud_synced_at', 'version', 'deleted', 'embedding']);
  const fields = Object.entries(item).filter(([k]) => !skip.has(k));

  kv.innerHTML = fields.map(([k, v]) => {
    let display = v;
    if (Array.isArray(v)) display = v.join(', ') || 'none';
    else if (v && typeof v === 'object') display = JSON.stringify(v, null, 2);
    else if (v === null || v === undefined) display = 'null';
    const isLong = String(display).length > 200;
    return `<span class="dk">${esc(k)}</span><span class="dv${isLong ? ' blob-full' : ''}">${esc(display)}</span>`;
  }).join('');

  // Load provenance
  const prov = document.getElementById('mem-prov');
  try {
    const derived = await api(`/api/provenance/${memType}/${memId}`);
    let html = '';
    if (item.derived_from && item.derived_from.length > 0) {
      html += '<strong>Derived from:</strong><br>' +
        item.derived_from.map(ref => `<div class="prov-item">${esc(ref)}</div>`).join('');
    }
    if (derived.length > 0) {
      html += '<br><strong>Derived memories (children):</strong><br>' +
        derived.map(d => `<div class="prov-item">${esc(d.type)}:${esc(truncId(d.id))}</div>`).join('');
    }
    if (!html) html = '<em style="color:var(--text-dim)">No provenance chain</em>';
    prov.innerHTML = html;
  } catch {
    prov.innerHTML = '';
  }

  document.getElementById('mem-detail').classList.add('open');
}

async function loadSuggestions() {
  try {
    const data = await api('/api/suggestions');
    const tbody = document.getElementById('sug-tbody');
    tbody.innerHTML = data.map(s => {
      const statusColor = s.status === 'pending' ? 'var(--yellow)' :
                          s.status === 'promoted' ? 'var(--green)' :
                          s.status === 'rejected' ? 'var(--red)' : 'var(--text-dim)';
      const preview = s.content ? JSON.stringify(s.content).slice(0, 120) : '';
      return `<tr data-sug-id="${esc(s.id)}">
        <td style="color:${statusColor};font-weight:600">${esc(s.status)}</td>
        <td>${esc(s.memory_type)}</td>
        <td style="font-family:var(--mono)">${(s.confidence||0).toFixed(2)}</td>
        <td class="blob-preview">${esc(preview)}</td>
        <td class="id-cell">${esc((s.source_raw_ids||[]).map(truncId).join(', '))}</td>
        <td>${esc(fmtDate(s.created_at))}</td>
        <td class="id-cell">${esc(s.promoted_to || '')}</td>
      </tr>`;
    }).join('');

    tbody.querySelectorAll('tr').forEach(tr => {
      tr.addEventListener('click', () => {
        const s = data.find(d => d.id === tr.dataset.sugId);
        if (!s) return;
        const kv = document.getElementById('sug-detail-kv');
        const fields = Object.entries(s);
        kv.innerHTML = fields.map(([k, v]) => {
          let display = v;
          if (Array.isArray(v)) display = v.join(', ') || 'none';
          else if (v && typeof v === 'object') display = JSON.stringify(v, null, 2);
          else if (v === null || v === undefined) display = 'null';
          const isLong = String(display).length > 200;
          return `<span class="dk">${esc(k)}</span><span class="dv${isLong ? ' blob-full' : ''}">${esc(display)}</span>`;
        }).join('');
        document.getElementById('sug-detail').classList.add('open');
      });
    });
  } catch (e) {
    console.error('Failed to load suggestions:', e);
  }
}

async function loadAudit() {
  try {
    const limit = document.getElementById('audit-limit').value;
    const data = await api(`/api/audit?limit=${limit}`);
    const tbody = document.getElementById('audit-tbody');
    tbody.innerHTML = data.map(a => {
      const details = typeof a.details === 'object' ? JSON.stringify(a.details) : (a.details || '');
      return `<tr>
        <td>${esc(fmtDate(a.timestamp))}</td>
        <td style="font-weight:600">${esc(a.operation)}</td>
        <td>${esc(a.memory_type)}</td>
        <td class="id-cell" title="${esc(a.memory_id)}">${esc(truncId(a.memory_id))}</td>
        <td>${esc(a.actor)}</td>
        <td class="blob-preview">${esc(details.slice(0, 120))}</td>
      </tr>`;
    }).join('');
  } catch (e) {
    console.error('Failed to load audit:', e);
  }
}

async function loadSettings() {
  try {
    const [settings, processing] = await Promise.all([api('/api/settings'), api('/api/processing')]);

    const sg = document.getElementById('settings-grid');
    sg.innerHTML = Object.entries(settings).map(([k, v]) =>
      `<div class="sg-cell sg-key">${esc(k)}</div><div class="sg-cell sg-val">${esc(v)}</div>`
    ).join('');

    const pg = document.getElementById('settings-proc-grid');
    pg.innerHTML = processing.map(p => `<div class="proc-card">
      <div class="pc-name">${esc(p.layer_transition || 'unknown')}</div>
      ${Object.entries(p).filter(([k]) => k !== 'layer_transition').map(([k, v]) =>
        `<div class="pc-row"><span class="pc-key">${esc(k)}</span><span class="pc-val">${esc(v)}</span></div>`
      ).join('')}
    </div>`).join('');
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
}

// ---- Init ----
loadHeader();
loadTab('overview');
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Kernle Dev Dashboard — Local Memory Stack Inspector"
    )
    parser.add_argument("--stack", "-s", required=True, help="Stack ID to inspect")
    parser.add_argument(
        "--port", "-p", type=int, default=8420, help="Port to serve on (default: 8420)"
    )
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Loading stack: %s", args.stack)
    k = Kernle(stack_id=args.stack)
    DashboardHandler.kernle_instance = k

    server = HTTPServer(("127.0.0.1", args.port), DashboardHandler)
    url = f"http://localhost:{args.port}"
    logger.info("Dashboard running at %s", url)

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
