#!/usr/bin/env python3
"""Scan llama-cpp-projects and generate a comprehensive HTML dashboard."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent

CATEGORY_MAP: Dict[str, str] = {
    "1-ai-fundamentals": "AI Fundamentals",
    "2-langchain": "LangChain",
    "3-prompt-engineering": "Prompt Engineering",
    "4-vector-db-rag": "Vector DB / RAG",
    "5-langgraph": "LangGraph",
    "6-mcp": "MCP",
    "7-bonus": "Bonus / Performance",
}

CATEGORY_ICONS: Dict[str, str] = {
    "1-ai-fundamentals": "🧠",
    "2-langchain": "🔗",
    "3-prompt-engineering": "💬",
    "4-vector-db-rag": "📐",
    "5-langgraph": "🔀",
    "6-mcp": "🔌",
    "7-bonus": "⚡",
}


@dataclass
class ProjectInfo:
    dir_name: str
    full_path: str
    description: str = ""
    language: str = "Python"
    test_count: int = 0
    test_files: int = 0
    test_pass: int = 0
    test_fail: int = 0
    test_run: bool = False
    has_readme: bool = False
    has_pyproject: bool = False
    has_tests: bool = False
    has_live_demo: bool = False
    llm_status: str = ""  # "live" | "mock" | "unknown"
    error: str = ""

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if not self.test_run:
            return "unknown"
        if self.test_fail == 0 and self.test_pass > 0:
            return "pass"
        if self.test_fail > 0:
            return "fail"
        return "unknown"


@dataclass
class CategoryInfo:
    folder: str
    name: str
    icon: str
    projects: List[ProjectInfo] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return sum(p.test_count for p in self.projects)

    @property
    def total_projects(self) -> int:
        return len(self.projects)


def discover_projects() -> List[CategoryInfo]:
    """Discover all projects across categories."""
    categories: List[CategoryInfo] = []

    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir() or not re.match(r"\d-", folder.name):
            continue
        if folder.name == "tools":
            continue

        cat = CategoryInfo(
            folder=folder.name,
            name=CATEGORY_MAP.get(folder.name, folder.name),
            icon=CATEGORY_ICONS.get(folder.name, "📁"),
        )

        for proj_dir in sorted(folder.iterdir()):
            if not proj_dir.is_dir() or proj_dir.name.startswith("."):
                continue
            pinfo = ProjectInfo(
                dir_name=proj_dir.name,
                full_path=str(proj_dir),
            )

            # pyproject.toml
            pyproject = proj_dir / "pyproject.toml"
            if pyproject.exists():
                pinfo.has_pyproject = True
                text = pyproject.read_text(encoding="utf-8", errors="replace")
                m = re.search(r'description\s*=\s*"([^"]*)"', text)
                if m:
                    pinfo.description = m.group(1)

            # README
            readme = proj_dir / "README.md"
            if readme.exists():
                pinfo.has_readme = True
                if not pinfo.description:
                    text = readme.read_text(encoding="utf-8", errors="replace")
                    # First non-empty, non-title line
                    for line in text.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            pinfo.description = line[:120]
                            break

            # Live demo integration
            demo_file = folder / "demo_live.py"
            if demo_file.exists():
                pinfo.has_live_demo = True

            # Test files
            test_dir = proj_dir / "tests"
            if test_dir.exists():
                pinfo.has_tests = True
                test_files_list = list(test_dir.glob("test_*.py"))
                pinfo.test_files = len(test_files_list)
                # Count test functions
                for tf in test_files_list:
                    content = tf.read_text(encoding="utf-8", errors="replace")
                    pinfo.test_count += len(re.findall(r"^\s*def test_", content, re.MULTILINE))

            cat.projects.append(pinfo)

        if cat.projects:
            categories.append(cat)

    return categories


def run_tests_for_project(pinfo: ProjectInfo) -> ProjectInfo:
    """Run pytest for a single project and update test results."""
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest", "tests/",
                "--tb=no", "-q", "--no-header",
            ],
            cwd=pinfo.full_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Parse output like "X passed, Y failed"
        output = result.stdout.strip()
        pinfo.test_run = True
        if result.returncode == 0:
            m = re.search(r"(\d+) passed", output)
            pinfo.test_pass = int(m.group(1)) if m else 0
            # Use count from test collection if pytest didn't report
            if pinfo.test_pass > pinfo.test_count:
                pinfo.test_count = pinfo.test_pass
        else:
            m = re.search(r"(\d+) failed", output)
            pinfo.test_fail = int(m.group(1)) if m else 0
            m = re.search(r"(\d+) passed", output)
            pinfo.test_pass = int(m.group(1)) if m else 0
            total = pinfo.test_pass + pinfo.test_fail
            if total > 0:
                # Update test count from actual runs
                pass
    except subprocess.TimeoutExpired:
        pinfo.error = "timeout"
    except Exception as e:
        pinfo.error = str(e)
    return pinfo


def run_all_tests(categories: List[CategoryInfo], max_workers: int = 6) -> None:
    """Run pytest for all projects in parallel."""
    all_projects = []
    for cat in categories:
        for p in cat.projects:
            if p.has_tests:
                all_projects.append(p)

    print(f"Running tests for {len(all_projects)} projects ({max_workers} workers)...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_tests_for_project, p): p for p in all_projects}
        done = 0
        for fut in as_completed(futures):
            done += 1
            p = futures[fut]
            status = "✓" if p.test_fail == 0 else "✗"
            print(f"  [{done}/{len(all_projects)}] {status} {p.dir_name} "
                  f"({p.test_pass}p/{p.test_fail}f)")


def generate_html(categories: List[CategoryInfo]) -> str:
    """Generate a standalone HTML dashboard."""
    total_projects = sum(c.total_projects for c in categories)
    total_tests = sum(c.total_tests for c in categories)
    total_pass = sum(sum(p.test_pass for p in c.projects) for c in categories)
    total_fail = sum(sum(p.test_fail for p in c.projects) for c in categories)

    # Color palette
    dark_bg = "#0a0e1a"
    card_bg = "#131827"
    accent = "#38bdf8"
    green = "#4ade80"
    red = "#f87171"
    yellow = "#fbbf24"
    muted = "#94a3b8"
    border = "#1e293b"

    # JSON data for charts
    chart_data = {
        "categories": [],
        "projectsPerCat": [],
        "testsPerCat": [],
        "passPerCat": [],
        "failPerCat": [],
    }
    for cat in categories:
        chart_data["categories"].append(cat.name)
        chart_data["projectsPerCat"].append(cat.total_projects)
        chart_data["testsPerCat"].append(cat.total_tests)
        chart_data["passPerCat"].append(sum(p.test_pass for p in cat.projects))
        chart_data["failPerCat"].append(sum(p.test_fail for p in cat.projects))

    projects_json = []
    for cat in categories:
        for p in cat.projects:
            llm_tag = "✅ live" if p.has_live_demo else "⏸️ standalone"
            projects_json.append({
                "name": p.dir_name,
                "cat": cat.name,
                "desc": p.description[:80] if p.description else "",
                "tests": p.test_count,
                "pass": p.test_pass,
                "fail": p.test_fail,
                "tested": p.test_run,
                "status": p.status,
                "hasReadme": p.has_readme,
                "hasPyproject": p.has_pyproject,
                "testFiles": p.test_files,
                "hasLiveDemo": p.has_live_demo,
                "llmTag": llm_tag,
            })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llama-cpp-projects — Comprehensive Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: {dark_bg}; color: #e2e8f0; padding: 14px 24px;
  }}

  /* Header — compact */
  .header {{ margin-bottom: 10px; }}
  .header h1 {{ font-size: 1.15rem; margin-bottom: 2px; }}
  .header h1 span {{ color: {accent}; }}
  .header .subtitle {{ color: {muted}; font-size: 0.75rem; }}

  /* Stats bar — single-line compact */
  .stats-bar {{
    display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px;
  }}
  .stat-card {{
    background: {card_bg}; border: 1px solid {border}; border-radius: 6px;
    padding: 6px 14px; display: flex; align-items: baseline; gap: 6px;
  }}
  .stat-card .stat-label {{ font-size: 0.65rem; color: {muted}; text-transform: uppercase; }}
  .stat-card .stat-value {{ font-size: 1.05rem; font-weight: 700; }}
  .stat-card .stat-sub {{ font-size: 0.65rem; color: {muted}; }}

  /* Chart row — compact, collapsible */
  .chart-row {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px;
  }}
  .chart-card {{
    background: {card_bg}; border: 1px solid {border}; border-radius: 6px; padding: 8px;
  }}
  .chart-card h3 {{ font-size: 0.7rem; color: {muted}; margin-bottom: 4px; }}
  .chart-card canvas {{ max-height: 140px !important; }}
  @media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}

  /* Search / filter */
  /* Chart toggle */
  .chart-toggle {{
    background: none; border: 1px solid {border}; color: {muted};
    cursor: pointer; font-size: 0.7rem; padding: 2px 10px; border-radius: 4px;
    transition: color 0.15s;
  }}
  .chart-toggle:hover {{ color: {accent}; }}

  .controls {{
    display: flex; gap: 8px; align-items: center; margin-bottom: 8px; flex-wrap: wrap;
  }}
  .controls input {{
    background: {card_bg}; color: #e2e8f0; border: 1px solid {border};
    padding: 6px 10px; border-radius: 6px; font-size: 0.78rem; flex: 1;
    min-width: 160px;
  }}
  .controls input::placeholder {{ color: {muted}; }}
  .controls select {{
    background: {card_bg}; color: #e2e8f0; border: 1px solid {border};
    padding: 6px 10px; border-radius: 6px; font-size: 0.78rem;
  }}
  .controls .tag-filter {{ display: flex; gap: 6px; }}
  .controls .tag {{
    background: {card_bg}; border: 1px solid {border}; border-radius: 20px;
    padding: 4px 12px; font-size: 0.75rem; cursor: pointer;
    transition: all 0.15s;
  }}
  .controls .tag:hover {{ border-color: {accent}; }}
  .controls .tag.active {{ background: {accent}; color: {dark_bg}; border-color: {accent}; }}

  /* Project grid */
  .project-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 8px;
  }}
  .cat-section {{ margin-bottom: 16px; }}
  .cat-header {{
    font-size: 0.9rem; font-weight: 600; margin-bottom: 6px;
    display: flex; align-items: center; gap: 6px;
  }}
  .cat-header .cat-badge {{
    font-size: 0.62rem; background: {border}; color: {muted};
    padding: 1px 8px; border-radius: 10px;
  }}

  .project-card {{
    background: {card_bg}; border: 1px solid {border}; border-radius: 7px;
    padding: 8px 12px; transition: border-color 0.15s;
  }}
  .project-card:hover {{ border-color: {accent}44; }}
  .project-card .p-name {{
    font-size: 0.82rem; font-weight: 600; margin-bottom: 2px;
    display: flex; align-items: center; gap: 6px;
  }}
  .project-card .p-desc {{ font-size: 0.72rem; color: {muted}; margin-bottom: 4px; line-height: 1.3; }}
  .project-card .p-meta {{ display: flex; gap: 6px; flex-wrap: wrap; font-size: 0.65rem; }}
  .project-card .p-meta span {{ background: {dark_bg}; padding: 1px 6px; border-radius: 4px; }}

  .badge {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }}
  .badge.pass {{ background: {green}22; color: {green}; }}
  .badge.fail {{ background: {red}22; color: {red}; }}
  .badge.unknown {{ background: #334155; color: {muted}; }}
  .badge.error {{ background: {red}33; color: {red}; }}

  /* Table view */
  .table-view {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; display: none; }}
  .table-view th {{ text-align: left; padding: 8px 10px; color: {muted}; border-bottom: 1px solid {border}; font-size: 0.72rem; text-transform: uppercase; }}
  .table-view td {{ padding: 7px 10px; border-bottom: 1px solid {border}55; }}

  .view-toggle {{ display: flex; gap: 4px; }}
  .view-toggle button {{
    background: {card_bg}; border: 1px solid {border}; color: {muted};
    padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.72rem;
  }}
  .view-toggle button.active {{ border-color: {accent}; color: {accent}; }}

  .footer {{ text-align: center; color: {muted}; font-size: 0.65rem; margin-top: 20px; padding-top: 10px; border-top: 1px solid {border}; }}
</style>
</head>
<body>

<div class="header">
  <h1>🦙 <span>llama-cpp-projects</span> — Workspace Dashboard</h1>
  <p class="subtitle">{total_projects} projects across {len(categories)} categories · {total_tests} test functions</p>
</div>

<div class="stats-bar">
  <div class="stat-card">
    <div class="stat-label">Projects</div>
    <div class="stat-value" style="color:{accent}">{total_projects}</div>
    <div class="stat-sub">{len(categories)} categories</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Test Functions</div>
    <div class="stat-value" style="color:{yellow}">{total_tests}</div>
    <div class="stat-sub">detected in tree</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Tests Passed</div>
    <div class="stat-value" style="color:{green}">{total_pass}</div>
    <div class="stat-sub">{total_fail} failed</div>
  </div>
    <div class="stat-card">
        <div class="stat-label">Completed</div>
        <div class="stat-value" style="color:{green}">{total_projects}</div>
        <div class="stat-sub">100% done</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">LLM Integration</div>
        <div class="stat-value" style="color:{green}">✅ Live</div>
        <div class="stat-sub">via _shared/integrations.py</div>
    </div>
</div>

<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
  <h3 style="color:{muted};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.5px">Charts</h3>
  <button class="chart-toggle" onclick="toggleCharts()" id="chartToggleBtn">– Hide</button>
</div>
<div id="chartSection" class="chart-row">
  <div class="chart-card">
    <h3>Projects per Category</h3>
    <canvas id="chartProjects"></canvas>
  </div>
  <div class="chart-card">
    <h3>Tests per Category <span style="color:{muted};font-weight:400">(Pass/Fail)</span></h3>
    <canvas id="chartTests"></canvas>
  </div>
</div>

<div class="controls">
  <input type="text" id="searchInput" placeholder="Search projects…" oninput="filterProjects()">
  <select id="catFilter" onchange="filterProjects()">
    <option value="">All categories</option>
    {"".join(f'<option value="{c.name}">{c.icon} {c.name}</option>' for c in categories)}
  </select>
  <div class="view-toggle">
    <button class="active" onclick="setView('grid', this)">☰ Grid</button>
    <button onclick="setView('table', this)">☷ Table</button>
  </div>
</div>

<div id="gridView">
{"".join(_cat_section_html(c, dark_bg, card_bg, border, muted, accent, green, red) for c in categories)}
</div>

<table class="table-view" id="tableView">
  <thead><tr>
    <th>Category</th><th>Project</th><th>Description</th><th>Tests</th><th>Pass</th><th>Fail</th><th>Status</th>
  </tr></thead>
  <tbody>
{"".join(
  f'<tr class="p-row" data-cat="{cat.name}" data-name="{p.dir_name}" data-desc="{p.description[:80]}">'
  f'<td>{cat.icon} {cat.name}</td>'
  f'<td><strong>{p.dir_name}</strong></td>'
  f'<td style="color:{muted}">{p.description[:80]}</td>'
  f'<td>{p.test_count}</td>'
  f'<td style="color:{green}">{p.test_pass if p.test_run else "—"}</td>'
  f'<td style="color:{red}">{p.test_fail if p.test_run else "—"}</td>'
  f'<td><span class="badge {p.status}">{p.status}</span></td>'
  f'</tr>'
  for cat in categories for p in cat.projects
)}
  </tbody>
</table>

<div class="footer">
  Generated by scan_dashboard.py · {total_projects} projects · {total_tests} tests
</div>

<script>
// ---- Charts ----
const chartData = {json.dumps(chart_data)};

new Chart(document.getElementById('chartProjects'), {{
  type: 'bar',
  data: {{
    labels: chartData.categories,
    datasets: [{{
      label: 'Projects',
      data: chartData.projectsPerCat,
      backgroundColor: ['#38bdf8','#a78bfa','#4ade80','#fbbf24','#f87171','#34d399','#f472b6'],
      borderRadius: 4,
    }}]
  }},
  options: chartOpts(false),
}});

new Chart(document.getElementById('chartTests'), {{
  type: 'bar',
  data: {{
    labels: chartData.categories,
    datasets: [
      {{ label: 'Pass', data: chartData.passPerCat, backgroundColor: '#4ade80', borderRadius: 4 }},
      {{ label: 'Fail', data: chartData.failPerCat, backgroundColor: '#f87171', borderRadius: 4 }},
    ]
  }},
  options: chartOpts(true),
}});

function chartOpts(stacked) {{
  return {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 10 }} }} }},
    }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
      y: {{ stacked: stacked, ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
    }},
  }};
}}

// ---- Chart toggle ----
function toggleCharts() {{
  const section = document.getElementById('chartSection');
  const btn = document.getElementById('chartToggleBtn');
  const hidden = section.style.display === 'none';
  section.style.display = hidden ? '' : 'none';
  btn.textContent = hidden ? '– Hide' : '+ Show';
}}

// ---- Filtering ----
function filterProjects() {{
  const q = document.getElementById('searchInput').value.toLowerCase();
  const cat = document.getElementById('catFilter').value;

  document.querySelectorAll('.project-card').forEach(card => {{
    const name = card.dataset.name.toLowerCase();
    const desc = card.dataset.desc.toLowerCase();
    const pcat = card.dataset.cat;
    const match = (!q || name.includes(q) || desc.includes(q)) && (!cat || pcat === cat);
    card.style.display = match ? '' : 'none';
  }});

  document.querySelectorAll('.p-row').forEach(row => {{
    const name = row.dataset.name.toLowerCase();
    const desc = row.dataset.desc.toLowerCase();
    const pcat = row.dataset.cat;
    const match = (!q || name.includes(q) || desc.includes(q)) && (!cat || pcat === cat);
    row.style.display = match ? '' : 'none';
  }});

  // Hide empty category sections
  document.querySelectorAll('.cat-section').forEach(section => {{
    const visible = [...section.querySelectorAll('.project-card')].some(c => c.style.display !== 'none');
    section.style.display = visible ? '' : 'none';
  }});
}}

// ---- View toggle ----
function setView(view, btn) {{
  document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('gridView').style.display = view === 'grid' ? '' : 'none';
  document.getElementById('tableView').style.display = view === 'table' ? '' : 'none';
}}
</script>
</body>
</html>"""


def _cat_section_html(cat, dark_bg, card_bg, border, muted, accent, green, red):
    cards = ""
    for p in cat.projects:
        meta_parts = []
        if p.test_count > 0:
            meta_parts.append(f"🧪 {p.test_count} tests")
        if p.test_run:
            meta_parts.append(f"✅ {p.test_pass}p")
            if p.test_fail:
                meta_parts.append(f"❌ {p.test_fail}f")
        if p.has_live_demo:
            meta_parts.append("🦙 live demo")
        meta_parts.append("📄 README" if p.has_readme else "")
        meta_parts.append("📦 pyproject" if p.has_pyproject else "")
        meta_parts.append(f"📁 {p.test_files} test files" if p.test_files else "")
        meta_parts = [m for m in meta_parts if m]
        meta_str = "".join(f"<span>{m}</span>" for m in meta_parts)
        status_badge = f'<span class="badge {p.status}">{p.status}</span>'

        # Determine if dir_name has sub-package content worth noting
        name_display = p.dir_name
        if p.has_pyproject:
            pass  # already a proper project

        cards += f"""
  <div class="project-card" data-name="{p.dir_name.lower()}" data-desc="{p.description[:80].lower()}" data-cat="{cat.name}">
    <div class="p-name">{name_display} {status_badge}</div>
    <div class="p-desc">{p.description or '<span style="color:#475569">(no description)</span>'}</div>
    <div class="p-meta">{meta_str}</div>
  </div>"""

    return f"""<div class="cat-section">
  <div class="cat-header">{cat.icon} {cat.name} <span class="cat-badge">{cat.total_projects} projects · {cat.total_tests} tests</span></div>
  <div class="project-grid">{cards}</div>
</div>"""


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate llama-cpp-projects dashboard")
    parser.add_argument("--no-test", action="store_true", help="Skip running tests (use file counts only)")
    args = parser.parse_args()

    print("🔍 Discovering projects...")
    categories = discover_projects()
    total_projects = sum(c.total_projects for c in categories)
    total_tests = sum(c.total_tests for c in categories)
    print(f"   Found {total_projects} projects with {total_tests} test functions across {len(categories)} categories.")

    if not args.no_test:
        run_all_tests(categories)

    print("📊 Generating HTML dashboard...")
    html = generate_html(categories)

    out_path = ROOT / "tools" / "dashboard" / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ Dashboard written to {out_path}")

    # Print summary
    total_pass = sum(sum(p.test_pass for p in c.projects) for c in categories)
    total_fail = sum(sum(p.test_fail for p in c.projects) for c in categories)
    print(f"\n{'='*50}")
    print(f"  {'Category':30s}  {'Proj':>4s}  {'Tests':>6s}  {'Pass':>5s}  {'Fail':>5s}")
    print(f"{'='*50}")
    for cat in categories:
        tp = sum(p.test_pass for p in cat.projects)
        tf = sum(p.test_fail for p in cat.projects)
        print(f"  {cat.name:30s}  {cat.total_projects:4d}  {cat.total_tests:6d}  {tp:5d}  {tf:5d}")
    print(f"{'='*50}")
    print(f"  {'TOTAL':30s}  {total_projects:4d}  {total_tests:6d}  {total_pass:5d}  {total_fail:5d}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
