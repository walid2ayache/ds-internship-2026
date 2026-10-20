const state = {
  meta: null,
  tx: [],
  filter: { step: "all", label: "all", outcome: "all", risk: "all", q: "" },
  selectedId: null,
};

const $ = (id) => document.getElementById(id);
const fmt3 = (x) => Number(x).toFixed(3);
const flagOf = (row) => row.prob >= state.meta.threshold;

function showError(message) {
  let el = $("error-banner");
  if (!el) {
    el = document.createElement("div");
    el.id = "error-banner";
    el.className = "error-banner";
    document.querySelector("main").prepend(el);
  }
  el.textContent = message;
  el.classList.remove("hidden");
}

async function boot() {
  try {
    const [metaRes, txRes] = await Promise.all([
      fetch("/api/meta"),
      fetch("/api/transactions"),
    ]);
    if (!metaRes.ok || !txRes.ok) throw new Error("API returned an error");
    state.meta = await metaRes.json();
    state.tx = await txRes.json();
    renderMeta();
    renderFilters();
    renderTable();
    if (state.tx.length) {
      const first = filteredRows()[0];
      if (first) select(first.id);
    }
  } catch (err) {
    showError(`Could not load the demo API: ${err.message}`);
  }
}

function renderMeta() {
  const m = state.meta;
  $("model-name").textContent = "XGBoost · final fit";
  $("threshold-value").textContent = m.threshold.toFixed(2);

  const stats = [
    ["PR-AUC", m.metrics.final_fit.pr_auc, "illicit · test 35–49"],
    ["F1", m.metrics.final_fit.f1, "at threshold " + m.threshold.toFixed(2)],
    ["Precision", m.metrics.final_fit.precision, "of flagged transactions"],
    ["Recall", m.metrics.final_fit.recall, "of illicit transactions"],
  ];
  $("stat-grid").innerHTML = stats.map(([label, value, sub]) => `
    <div class="stat">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${fmt3(value)}</div>
      <div class="stat-sub">${sub}</div>
    </div>`).join("");

  const f = m.metrics.final_fit;
  const t = m.metrics.train_only;
  const s = m.metrics.sage_local;
  const rows = [
    ["XGBoost · final fit", f.pr_auc, f.f1, f.precision, f.recall, true],
    ["XGBoost · train-only", t.pr_auc, t.f1, t.precision, t.recall, false],
    ["SAGE · local features", s.test_pr_auc, s.test_f1, s.precision, s.recall, false],
  ];
  $("metric-table").querySelector("tbody").innerHTML = rows.map((r) => `
    <tr>
      <td>${r[0]}${r[5] ? " <span class=\"muted\">(served)</span>" : ""}</td>
      <td class="${r[5] ? "best" : ""}">${fmt3(r[1])}</td>
      <td class="${r[5] ? "best" : ""}">${fmt3(r[2])}</td>
      <td>${fmt3(r[3])}</td>
      <td>${fmt3(r[4])}</td>
    </tr>`).join("");
}

function renderFilters() {
  const steps = [...new Set(state.tx.map((t) => t.step))].sort((a, b) => a - b);
  $("filter-step").innerHTML = `<option value="all">All</option>` +
    steps.map((s) => `<option value="${s}">${s}</option>`).join("");
}

function filteredRows() {
  const { step, label, outcome, risk, q } = state.filter;
  const query = q.trim().toLowerCase();
  return state.tx
    .filter((t) => {
      const flag = flagOf(t);
      if (step !== "all" && String(t.step) !== step) return false;
      if (label !== "all" && String(t.y) !== label) return false;
      const correct = (t.y === 1) === flag;
      if (outcome === "correct" && !correct) return false;
      if (outcome === "errors" && correct) return false;
      if (risk !== "all" && flag !== (risk === "flagged")) return false;
      if (query && !String(t.node).includes(query) && !String(t.id).includes(query)) return false;
      return true;
    })
    .sort((a, b) => b.prob - a.prob);
}

function renderTable() {
  const rows = filteredRows();
  $("result-count").textContent = `${rows.length} transactions · sorted by risk`;
  $("table-foot").textContent = `Showing ${rows.length} of ${state.tx.length} demo rows`;
  $("tx-body").innerHTML = rows.map((t) => {
    const flag = flagOf(t);
    return `
      <tr data-id="${t.id}" class="${state.selectedId === t.id ? "selected" : ""}">
        <td class="node">#${t.node}</td>
        <td class="step">${t.step}</td>
        <td>
          <div class="prob-cell">
            <span class="prob-num">${fmt3(t.prob)}</span>
            <span class="mini-bar"><i style="width:${(t.prob * 100).toFixed(1)}%"></i></span>
          </div>
        </td>
        <td><span class="badge ${flag ? "flag" : "clear"}">${flag ? "Flagged" : "Clear"}</span></td>
        <td class="${t.y === 1 ? "label-illicit" : "label-licit"}">${t.y === 1 ? "Illicit" : "Licit"}</td>
      </tr>`;
  }).join("");

  document.querySelectorAll("#tx-body tr").forEach((tr) => {
    tr.addEventListener("click", () => select(Number(tr.dataset.id)));
  });
}

function contribRows(items, cls, maxAbs) {
  if (!items || !items.length) return `<p class="muted">No contributions in this direction.</p>`;
  return items.map((c) => {
    const width = maxAbs > 0 ? (Math.abs(c.contrib) / maxAbs) * 100 : 0;
    return `
      <div class="contrib">
        <div class="contrib-head">
          <span class="name">${c.name}</span>
          <span class="val ${cls}">${c.contrib >= 0 ? "+" : ""}${c.contrib.toFixed(3)}</span>
        </div>
        <div class="contrib-track"><i class="${cls}" style="width:${width.toFixed(1)}%"></i></div>
        <div class="contrib-meta">feature value ${c.value.toFixed(3)}</div>
      </div>`;
  }).join("");
}

function renderDetail(d) {
  const flag = d.prob >= d.threshold;
  const all = [...d.top_positive, ...d.top_negative];
  const maxAbs = Math.max(0, ...all.map((c) => Math.abs(c.contrib)));
  const label = d.y === 1 ? "Illicit" : "Licit";

  $("detail-empty").classList.add("hidden");
  const content = $("detail-content");
  content.classList.remove("hidden");
  content.innerHTML = `
    <div class="detail-inner">
      <div class="detail-top">
        <div>
          <div class="detail-kicker">Demo transaction</div>
          <h3 class="detail-title">Node #${d.node}</h3>
        </div>
        <button class="close-btn" id="close-detail" aria-label="Close">×</button>
      </div>

      <div class="score-row">
        <span class="score-value">${d.prob.toFixed(3)}</span>
        <span class="score-unit">risk score</span>
      </div>
      <div class="gauge">
        <i style="width:${(d.prob * 100).toFixed(1)}%"></i>
        <b style="left:${(d.threshold * 100).toFixed(1)}%"></b>
      </div>
      <div class="gauge-labels"><span>0.00</span><span>threshold ${d.threshold.toFixed(2)}</span><span>1.00</span></div>

      <div class="meta-row">
        <span class="chip">Step <strong>${d.step}</strong></span>
        <span class="chip">True label <strong class="${d.y === 1 ? "label-illicit" : "label-licit"}">${label}</strong></span>
        <span class="chip">${flag ? "Flagged" : "Clear"}</span>
      </div>

      <div class="contrib-section">
        <h4>Pushes toward illicit</h4>
        <p>Positive XGBoost contributions, in log-odds.</p>
        ${contribRows(d.top_positive, "pos", maxAbs)}
      </div>
      <div class="contrib-section">
        <h4>Pushes toward licit</h4>
        <p>Negative XGBoost contributions, in log-odds.</p>
        ${contribRows(d.top_negative, "neg", maxAbs)}
      </div>
      <p class="micro">Ground truth is shown because this is a labelled historical demo row;
        it is never used by the model.</p>
    </div>`;

  $("close-detail").addEventListener("click", closeDetail);
  document.querySelectorAll("#tx-body tr").forEach((tr) => {
    tr.classList.toggle("selected", Number(tr.dataset.id) === state.selectedId);
  });
}

function closeDetail() {
  state.selectedId = null;
  $("detail-content").classList.add("hidden");
  $("detail-empty").classList.remove("hidden");
  document.querySelectorAll("#tx-body tr.selected").forEach((tr) => tr.classList.remove("selected"));
}

async function select(id) {
  state.selectedId = id;
  document.querySelectorAll("#tx-body tr").forEach((tr) => {
    tr.classList.toggle("selected", Number(tr.dataset.id) === id);
  });
  try {
    const res = await fetch(`/api/transaction/${id}`);
    if (!res.ok) throw new Error("detail request failed");
    renderDetail(await res.json());
  } catch (err) {
    showError(`Could not load transaction ${id}: ${err.message}`);
  }
}

function activateTab(name) {
  document.querySelectorAll(".tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".panel-page").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${name}`);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function init() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });
  document.querySelectorAll("[data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.jump));
  });
  $("filter-step").addEventListener("change", (e) => {
    state.filter.step = e.target.value; renderTable();
  });
  $("filter-label").addEventListener("change", (e) => {
    state.filter.label = e.target.value; renderTable();
  });
  $("filter-outcome").addEventListener("change", (e) => {
    state.filter.outcome = e.target.value; renderTable();
  });
  $("filter-risk").addEventListener("change", (e) => {
    state.filter.risk = e.target.value; renderTable();
  });
  $("filter-q").addEventListener("input", (e) => {
    state.filter.q = e.target.value; renderTable();
  });
  $("random-btn").addEventListener("click", () => {
    if (!state.tx.length) return;
    const id = Math.floor(Math.random() * state.tx.length);
    select(id);
    $("detail-panel").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  const hash = location.hash.replace("#", "");
  if (["explorer", "performance", "about"].includes(hash)) activateTab(hash);
  boot();
}

document.addEventListener("DOMContentLoaded", init);
