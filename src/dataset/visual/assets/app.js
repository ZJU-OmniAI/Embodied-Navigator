const state = {
  data: null,
  chartInstances: [],
  resizeObservers: [],
  availableRanks: [],
  activeRanks: null,
  smoothingWindow: 1,
};

const dom = {
  cards: document.getElementById("cards"),
  charts: document.getElementById("charts"),
  logDirInput: document.getElementById("log-dir-input"),
  rankChips: document.getElementById("rank-chips"),
  refreshBtn: document.getElementById("refresh-btn"),
  smoothSlider: document.getElementById("smooth-slider"),
  smoothValue: document.getElementById("smooth-value"),
  metaLogDir: document.getElementById("meta-log-dir"),
  metaRanks: document.getElementById("meta-ranks"),
  metaCache: document.getElementById("meta-cache"),
};

function formatValue(value, mode) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (mode === "int") {
    return Number(value).toLocaleString();
  }
  if (mode === "percent") {
    return `${(Number(value) * 100).toFixed(2)}%`;
  }
  if (mode === "float3") {
    return Number(value).toFixed(3);
  }
  return String(value);
}

function formatTooltipNumber(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return String(value);
  }
  const abs = Math.abs(num);
  if (abs >= 1000) {
    return num.toFixed(2);
  }
  if (abs >= 1) {
    return num.toFixed(4);
  }
  if (abs > 0) {
    return num.toExponential(2);
  }
  return "0";
}

function buildCompactTooltipFormatter(maxItems) {
  return (rawParams) => {
    const params = Array.isArray(rawParams) ? rawParams : [rawParams];
    if (!params.length) {
      return "";
    }
    const head = params[0].axisValueLabel || params[0].name || "";
    const lines = [];
    if (head) {
      lines.push(`<div style="margin-bottom:4px;color:#cfd8e3;">${head}</div>`);
    }

    const visible = params.slice(0, maxItems);
    for (const item of visible) {
      const rawValue = Array.isArray(item.value) ? item.value[item.value.length - 1] : item.value;
      const valueText = formatTooltipNumber(rawValue);
      lines.push(
        `${item.marker || ""}<span style="color:#d8e1ee;">${item.seriesName}</span>: <b style="color:#fff;">${valueText}</b>`,
      );
    }
    if (params.length > maxItems) {
      lines.push(`<span style="color:#9fb0c6;">+${params.length - maxItems} more</span>`);
    }
    return lines.join("<br/>");
  };
}

function getRankQueryParam() {
  if (!state.activeRanks || state.activeRanks.size === 0) {
    return "all";
  }
  return [...state.activeRanks].sort((a, b) => a - b).join(",");
}

function normalizeActiveRanks() {
  if (!state.activeRanks) {
    return;
  }
  const filtered = new Set([...state.activeRanks].filter((r) => state.availableRanks.includes(r)));
  state.activeRanks = filtered.size > 0 ? filtered : null;
}

function renderRankChips() {
  dom.rankChips.innerHTML = "";

  const allChip = document.createElement("button");
  allChip.className = `chip ${state.activeRanks ? "" : "active"}`;
  allChip.textContent = "All";
  allChip.onclick = () => {
    state.activeRanks = null;
    renderRankChips();
    fetchAndRender();
  };
  dom.rankChips.appendChild(allChip);

  for (const rank of state.availableRanks) {
    const chip = document.createElement("button");
    const active = state.activeRanks ? state.activeRanks.has(rank) : false;
    chip.className = `chip ${active ? "active" : ""}`;
    chip.textContent = `rank ${rank}`;
    chip.onclick = () => {
      if (!state.activeRanks) {
        state.activeRanks = new Set();
      }
      if (state.activeRanks.has(rank)) {
        state.activeRanks.delete(rank);
      } else {
        state.activeRanks.add(rank);
      }
      if (state.activeRanks.size === 0 || state.activeRanks.size === state.availableRanks.length) {
        state.activeRanks = null;
      }
      renderRankChips();
      fetchAndRender();
    };
    dom.rankChips.appendChild(chip);
  }
}

function renderCards(cards) {
  dom.cards.innerHTML = "";
  for (const card of cards) {
    const el = document.createElement("article");
    el.className = "card";
    el.innerHTML = `
      <p class="label">${card.label}</p>
      <p class="value">${formatValue(card.value, card.format)}</p>
    `;
    dom.cards.appendChild(el);
  }
}

function smoothPoints(points, window) {
  if (!window || window <= 1) {
    return points;
  }
  const out = [];
  const buf = [];
  for (const [x, y] of points) {
    if (Number.isFinite(y)) {
      buf.push(y);
    }
    if (buf.length > window) {
      buf.shift();
    }
    const m = buf.length > 0 ? buf.reduce((a, b) => a + b, 0) / buf.length : null;
    out.push([x, m]);
  }
  return out;
}

function buildLineOption(chart) {
  const options = chart.options || {};
  const lineSeries = chart.series.map((series) => {
    const rawPoints = series.points || [];
    const points = series.smoothable ? smoothPoints(rawPoints, state.smoothingWindow) : rawPoints;
    return {
      type: "line",
      name: series.name,
      data: points,
      showSymbol: false,
      smooth: false,
      clip: true,
      connectNulls: true,
      stack: options.stack ? "stack_group" : undefined,
      lineStyle: {
        width: series.line_width || 1.7,
        type: series.line_style || "solid",
      },
      areaStyle: options.area ? { opacity: 0.16 } : undefined,
      itemStyle: {
        color: series.color || undefined,
      },
      emphasis: { focus: "series" },
    };
  });
  const maxPointCount = Math.max(0, ...lineSeries.map((s) => s.data.length));
  const dataZoom = [];
  if (maxPointCount > 120) {
    dataZoom.push({ type: "inside", realtime: true });
  }
  if (maxPointCount > 240) {
    dataZoom.push({ type: "slider", height: 18, bottom: 8 });
  }

  return {
    animation: false,
    color: lineSeries.map((s) => s.itemStyle.color).filter(Boolean),
    legend: { type: "scroll", top: 2 },
    tooltip: {
      trigger: "axis",
      confine: false,
      appendToBody: true,
      alwaysShowContent: false,
      enterable: false,
      padding: [6, 8],
      backgroundColor: "rgba(24, 28, 34, 0.95)",
      borderWidth: 0,
      textStyle: { color: "#f3f6fb", fontSize: 11, lineHeight: 14 },
      extraCssText: "max-width: 340px; white-space: normal; overflow-wrap: anywhere;",
      formatter: buildCompactTooltipFormatter(7),
    },
    grid: { left: 44, right: 16, top: 44, bottom: 58, containLabel: true },
    xAxis: {
      type: "value",
      name: chart.x_label,
      nameLocation: "middle",
      nameGap: 32,
      axisLabel: { color: "#516179", hideOverlap: true },
      splitLine: { show: false },
      axisLine: { lineStyle: { color: "#cad5e6" } },
    },
    yAxis: {
      type: "value",
      name: chart.y_label,
      scale: true,
      axisLabel: { color: "#516179" },
      splitLine: { lineStyle: { color: "rgba(82, 99, 130, 0.14)" } },
      min: options.y_min ?? null,
      max: options.y_max ?? null,
    },
    dataZoom,
    series: lineSeries,
  };
}

function buildBarOption(chart) {
  const options = chart.options || {};
  const categories = [];
  const categorySet = new Set();

  for (const series of chart.series) {
    for (const [x] of series.points || []) {
      const key = String(x);
      if (!categorySet.has(key)) {
        categorySet.add(key);
        categories.push(key);
      }
    }
  }

  const barSeries = chart.series.map((series) => {
    const map = new Map((series.points || []).map(([x, y]) => [String(x), y]));
    return {
      type: "bar",
      name: series.name,
      clip: true,
      data: categories.map((c) => map.get(c) ?? null),
      itemStyle: { color: series.color || undefined },
      barMaxWidth: 26,
    };
  });
  const hasManyCategories = categories.length > 18;

  return {
    animation: false,
    color: barSeries.map((s) => s.itemStyle.color).filter(Boolean),
    legend: { type: "scroll", top: 2 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      confine: false,
      appendToBody: true,
      alwaysShowContent: false,
      enterable: false,
      padding: [6, 8],
      backgroundColor: "rgba(24, 28, 34, 0.95)",
      borderWidth: 0,
      textStyle: { color: "#f3f6fb", fontSize: 11, lineHeight: 14 },
      extraCssText: "max-width: 320px; white-space: normal; overflow-wrap: anywhere;",
      formatter: buildCompactTooltipFormatter(6),
    },
    grid: { left: 44, right: 16, top: 44, bottom: 60, containLabel: true },
    xAxis: {
      type: "category",
      data: categories,
      name: chart.x_label,
      nameLocation: "middle",
      nameGap: 36,
      axisLabel: {
        color: "#516179",
        rotate: categories.length > 12 ? 30 : 0,
        hideOverlap: true,
        overflow: "truncate",
        width: 88,
      },
      axisLine: { lineStyle: { color: "#cad5e6" } },
    },
    yAxis: {
      type: "value",
      name: chart.y_label,
      scale: true,
      axisLabel: { color: "#516179" },
      splitLine: { lineStyle: { color: "rgba(82, 99, 130, 0.14)" } },
      min: options.y_min ?? null,
      max: options.y_max ?? null,
    },
    dataZoom: hasManyCategories
      ? [
          { type: "inside", realtime: true },
          { type: "slider", height: 18, bottom: 8 },
        ]
      : [],
    series: barSeries,
  };
}

function disposeCharts() {
  for (const observer of state.resizeObservers) {
    observer.disconnect();
  }
  state.resizeObservers = [];
  for (const ins of state.chartInstances) {
    ins.dispose();
  }
  state.chartInstances = [];
}

function renderCharts(charts) {
  disposeCharts();
  dom.charts.innerHTML = "";

  for (const chart of charts) {
    const panel = document.createElement("article");
    panel.className = "chart-panel";

    const title = document.createElement("h3");
    title.className = "chart-title";
    title.textContent = chart.title;

    const desc = document.createElement("p");
    desc.className = "chart-desc";
    desc.textContent = chart.description || "";

    const canvas = document.createElement("div");
    canvas.className = "chart-canvas";

    panel.appendChild(title);
    panel.appendChild(desc);
    panel.appendChild(canvas);
    dom.charts.appendChild(panel);

    const instance = echarts.init(canvas, null, { renderer: "canvas" });
    const option = chart.kind === "bar" ? buildBarOption(chart) : buildLineOption(chart);
    instance.setOption(option);
    if (window.ResizeObserver) {
      const observer = new ResizeObserver(() => instance.resize());
      observer.observe(canvas);
      state.resizeObservers.push(observer);
    }
    requestAnimationFrame(() => instance.resize());
    state.chartInstances.push(instance);
  }
}

function updateMeta(meta) {
  dom.metaLogDir.textContent = meta.requested_log_dir || "-";
  dom.metaRanks.textContent = (meta.ranks || []).join(", ") || "all";
  dom.metaCache.textContent = meta.cache_hit ? "hit" : "miss";
}

async function fetchDashboard() {
  const params = new URLSearchParams();
  const logDir = dom.logDirInput.value.trim();
  if (logDir) {
    params.set("log_dir", logDir);
  }
  params.set("ranks", getRankQueryParam());

  const res = await fetch(`/api/data?${params.toString()}`);
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return await res.json();
}

async function fetchAndRender() {
  dom.refreshBtn.disabled = true;
  dom.refreshBtn.textContent = "Loading...";
  try {
    const data = await fetchDashboard();
    state.data = data;
    state.availableRanks = data.meta.available_ranks || [];
    normalizeActiveRanks();
    renderRankChips();
    updateMeta(data.meta);
    if (!dom.logDirInput.value) {
      dom.logDirInput.value = data.meta.requested_log_dir || "";
    }
    renderCards(data.cards || []);
    renderCharts(data.charts || []);
  } catch (err) {
    dom.cards.innerHTML = `<article class="card"><p class="label">Error</p><p class="value">${err.message}</p></article>`;
    dom.charts.innerHTML = "";
  } finally {
    dom.refreshBtn.disabled = false;
    dom.refreshBtn.textContent = "Refresh";
  }
}

function bindEvents() {
  dom.refreshBtn.onclick = () => fetchAndRender();
  dom.smoothSlider.oninput = () => {
    state.smoothingWindow = Number(dom.smoothSlider.value || 1);
    dom.smoothValue.textContent = String(state.smoothingWindow);
    if (state.data) {
      renderCharts(state.data.charts || []);
    }
  };

  window.addEventListener("resize", () => {
    for (const ins of state.chartInstances) {
      ins.resize();
    }
  });
}

(function boot() {
  dom.smoothValue.textContent = String(state.smoothingWindow);
  bindEvents();
  fetchAndRender();
})();
