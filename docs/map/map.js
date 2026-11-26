// Structural-system map of the paper. Display only: every number comes
// precomputed from analysis/map_data.py (data/*.pmtiles, data/experiments.json).

const BASEMAPS = {
  light: { label: "Light", tone: "light", style: "https://tiles.openfreemap.org/styles/positron" },
  dark: { label: "Dark", tone: "dark", style: "https://tiles.openfreemap.org/styles/dark" },
  satellite: {
    label: "Satellite", tone: "dark",
    style: {
      version: 8,
      sources: { sat: { type: "raster", tileSize: 256, attribution: "&copy; Esri, Maxar, Earthstar Geographics",
        tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"] } },
      layers: [{ id: "sat", type: "raster", source: "sat" }],
    },
  },
};
// Outline of surveyed buildings: black on the light basemap, a white glow on dark and satellite ones.
const OUTLINE = { light: "#000000", dark: "#ffffff" };
// Near the background, so buildings without a value read as blank rather than as a class.
const NO_DATA = { light: "#f3f3f1", dark: "#2a2f35" };
const LAYER = "buildings";  // layer name inside the .pmtiles files

const CLASS_NAMES = { ADO: "Adobe", CR: "Reinforced concrete", M: "Masonry", S: "Light steel and wood (S/W)" };
const CLASS_COLORS = {
  light: { ADO: "#B24A2A", CR: "#8A949E", M: "#F0D78C", S: "#2F6DB5" },
  dark: { ADO: "#D2694A", CR: "#A9B3BC", M: "#F0D78C", S: "#5FA2EC" },
};
const RAMP = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];
const MODEL_NAMES = { LogisticRegression: "Logistic regression", RandomForest: "Random forest",
  XGBoost: "XGBoost", CatBoost: "CatBoost", HistGradientBoosting: "Histogram gradient boosting" };

const VIEWS = {
  cityPred: { label: "Predicted structural system (every building)", type: "cityPred" },
  cls: { label: "Surveyed structural system", type: "class" },
  height: { label: "Height (m)", type: "ramp", domain: [0, 30] },
  area: { label: "Footprint area (m²)", type: "ramp", domain: [0, 800] },
  year: { label: "Construction era", type: "ramp", domain: [1950, 2025] },
  code: { label: "Seismic-code level", type: "cat", values: {
    pre_code: ["Pre-code", "#b2182b"], low_code: ["Low code", "#ef8a62"],
    medium_code: ["Medium code", "#67a9cf"], high_code: ["High code", "#2166ac"] } },
  roof: { label: "Roof material", type: "cat", values: {
    metallic: ["Metallic", "#9e9ac8"], bright_concrete: ["Bright concrete", "#e0e0e0"],
    dark_shadow: ["Dark or shadow", "#525252"], asphalt: ["Asphalt", "#8c6d31"] } },
  position: { label: "Position in the block", type: "cat", values: {
    isolated: ["Isolated", "#DDDDDD"], lateral: ["Lateral", "#77AADD"], corner: ["Corner", "#EEDD88"],
    confined: ["Confined", "#44BB99"], torque: ["Torque", "#BBCC33"] } },
  hull: { label: "Convex-hull irregularity", type: "ramp", domain: [0, 0.4] },
  eccentricity: { label: "Eccentricity ratio (Eurocode 8)", type: "ramp", domain: [0, 0.4] },
  compact: { label: "Compactness (Eurocode 8)", type: "ramp", domain: [0.6, 1] },
  confinement: { label: "Contact confinement ratio", type: "ramp", domain: [0, 1] },
  pred: { label: "Predicted structural system (test buildings)", type: "pred" },
  correct: { label: "Correct or wrong prediction", type: "pred" },
  agree: { label: "Agreement among the five models", type: "pred" },
};
const POPUP_FIELDS = [["survey", "Survey"], ["area", "Footprint area (m²)"], ["height", "Height (m)"],
  ["year", "Construction era"], ["code", "Seismic-code level"], ["roof", "Roof material"],
  ["position", "Position in the block"], ["hull", "Convex-hull irregularity"],
  ["eccentricity", "Eccentricity ratio (Eurocode 8)"], ["compact", "Compactness (Eurocode 8)"],
  ["confinement", "Contact confinement ratio"]];

const $ = (id) => document.getElementById(id);
const state = { city: "san_jose", survey: "all", view: "cityPred", basemap: "light", threeD: true,
  experiment: null, model: null };
let meta = null;
let testIndex = new Map();  // key -> index in the current experiment's test set

const tone = () => BASEMAPS[state.basemap].tone;
const tilesUrl = (city) => `pmtiles://${new URL(`data/buildings_${city}.pmtiles`, window.location.href).href}`;

const protocol = new pmtiles.Protocol();
maplibregl.addProtocol("pmtiles", protocol.tile);

const map = new maplibregl.Map({
  container: "map", style: BASEMAPS.light.style, center: [-84, 12], zoom: 4, pitch: 50, maxPitch: 70,
  attributionControl: { compact: true },
});
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");

// 3D, presentation mode and plots panel: one button each, next to the zoom buttons.
const ICONS = {
  present: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="12" rx="1"/><path d="M12 16v4M8 20h8"/></svg>',
  plots: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg>',
  moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
  sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
};
class ToggleControl {
  onAdd() {
    this.el = document.createElement("div");
    this.el.className = "maplibregl-ctrl maplibregl-ctrl-group";
    const add = (id, title, html, onClick) => {
      const b = document.createElement("button");
      b.id = id; b.type = "button"; b.title = title; b.className = "toggle"; b.innerHTML = html;
      b.addEventListener("click", onClick);
      this.el.appendChild(b);
    };
    add("btnTheme", "Dark mode", ICONS.moon, () => setDark(!document.body.classList.contains("dark")));
    add("btn3d", "3D view", "3D", () => setThreeD(!state.threeD));
    add("btnPresent", "Presentation mode (P)", ICONS.present, () => setPresentation(!document.body.classList.contains("presentation")));
    add("btnPlots", "Plots", ICONS.plots, () => setPlots($("right").classList.contains("closed")));
    return this.el;
  }
  onRemove() { this.el.remove(); }
}
map.addControl(new ToggleControl(), "top-right");

function setThreeD(on) {
  state.threeD = on;
  $("btn3d").classList.toggle("on", on);
  restyle();
  map.easeTo({ pitch: on ? 50 : 0, duration: 600 });
}

// Dark or light panels. The basemap follows (light <-> dark), except the satellite view.
function setDark(on) {
  document.body.classList.toggle("dark", on);
  $("btnTheme").innerHTML = on ? ICONS.sun : ICONS.moon;
  $("btnTheme").title = on ? "Light mode" : "Dark mode";
  if (state.basemap !== "satellite") setBasemap(on ? "dark" : "light");
}

function setBasemap(id) {
  if (state.basemap === id) return;
  state.basemap = id;
  $("basemap").value = id;
  map.setStyle(BASEMAPS[id].style, { diff: false });
  map.once("style.load", () => { addLayers(); restyle(); collapseAttribution(); });
}

function setPlots(open) {
  $("right").classList.toggle("closed", !open);
  $("btnPlots").classList.toggle("on", open);
}

function setPresentation(on) {
  document.body.classList.toggle("presentation", on);
  $("btnPresent").classList.toggle("on", on);
}

// The plots panel sits right under the map buttons, against the right edge.
function placePlots() {
  const ctrl = document.querySelector(".maplibregl-ctrl-top-right");
  if (window.innerWidth > 800 && ctrl) $("right").style.top = `${ctrl.getBoundingClientRect().bottom + 8}px`;
}

// Attribution starts collapsed (the "i" button opens it).
function collapseAttribution() {
  const a = document.querySelector(".maplibregl-ctrl-attrib");
  if (a) { a.classList.remove("maplibregl-compact-show"); a.removeAttribute("open"); }
}

function experimentData() {
  const e = meta.experiments[state.experiment];
  return e && e.cities[state.city] ? { e, d: e.cities[state.city] } : null;
}

// ---------------------------------------------------------------- styling
function classMatch(value) {
  return ["match", ["coalesce", value, ""], ...Object.entries(CLASS_COLORS[tone()]).flat(), NO_DATA[tone()]];
}

function colorExpression() {
  const v = VIEWS[state.view];
  const nodata = NO_DATA[tone()];
  if (v.type === "cityPred") return classMatch(["get", `p_${meta.model_keys[state.model]}`]);
  if (v.type === "class") return classMatch(["get", "cls"]);
  if (v.type === "cat") {
    return ["match", ["coalesce", ["get", state.view], ""], ...Object.entries(v.values).flatMap(([k, [, c]]) => [k, c]), nodata];
  }
  if (v.type === "ramp") {
    const [lo, hi] = v.domain;
    const stops = RAMP.flatMap((c, i) => [lo + (hi - lo) * i / (RAMP.length - 1), c]);
    return ["case", ["==", ["typeof", ["get", state.view]], "number"],
      ["interpolate", ["linear"], ["get", state.view], ...stops], nodata];
  }
  const isTest = ["boolean", ["feature-state", "t"], false];
  if (state.view === "pred") return ["case", isTest, classMatch(["feature-state", "p"]), nodata];
  if (state.view === "correct") {
    return ["case", isTest, ["case", ["boolean", ["feature-state", "ok"], false], "#2e9e5b", "#d64541"], nodata];
  }
  return ["case", isTest, ["match", ["coalesce", ["feature-state", "ag"], 0],
    5, "#2c7fb8", 4, "#7fcdbb", 3, "#fdae61", "#d7191c"], nodata];
}

function heightExpression() {
  return state.threeD ? ["case", ["has", "height"], ["get", "height"], 3] : 0;
}

const surveyed = ["has", "cls"];
function surveyFilter(extra) {
  const parts = [extra, state.survey === "all" ? null : ["==", ["get", "survey"], state.survey]].filter(Boolean);
  return parts.length ? ["all", ...parts] : null;
}

function addLayers() {
  if (map.getSource("b")) return;
  // The basemap's own buildings (3D in the dark style) would cover ours.
  for (const l of map.getStyle().layers) {
    if (l["source-layer"] === "building") map.setLayoutProperty(l.id, "visibility", "none");
  }
  map.addSource("b", { type: "vector", url: tilesUrl(state.city), promoteId: { [LAYER]: "k" },
    attribution: "Buildings: field surveys of the paper, partly Microsoft Building Footprints. Until the paper is published, the shapes and attributes of unsurveyed buildings are altered" });
  // Below the basemap's labels, so place names stay readable.
  const firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
  const common = { source: "b", "source-layer": LAYER };
  map.addLayer({ id: "b-fill", type: "fill-extrusion", ...common,
    paint: { "fill-extrusion-color": colorExpression(), "fill-extrusion-height": heightExpression(),
      "fill-extrusion-opacity": 1 } }, firstLabel);
  // Every footprint: a faint thin outline.
  map.addLayer({ id: "b-line", type: "line", ...common,
    paint: { "line-color": OUTLINE[tone()], "line-opacity": 0.25,
      "line-width": ["interpolate", ["linear"], ["zoom"], 13, 0.1, 17, 0.5] } }, firstLabel);
  // Surveyed buildings: a black border on the light basemap, a white glowing one (soft halo + solid core) on dark ones.
  map.addLayer({ id: "b-glow", type: "line", ...common, filter: surveyed,
    paint: { "line-color": "#ffffff", "line-opacity": tone() === "dark" ? 0.55 : 0, "line-blur": 3,
      "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.5, 14, 2, 16, 5, 18, 8] } }, firstLabel);
  map.addLayer({ id: "b-surveyed", type: "line", ...common, filter: surveyed,
    paint: { "line-color": OUTLINE[tone()], "line-opacity": 1,
      "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.3, 14, 0.8, 16, 1.6, 18, 2.5] } }, firstLabel);
  applySurvey();
  map.once("idle", applyPredictions);
}

function removeLayers() {
  for (const id of ["b-surveyed", "b-glow", "b-line", "b-fill"]) if (map.getLayer(id)) map.removeLayer(id);
  if (map.getSource("b")) map.removeSource("b");
}

function restyle() {
  if (!map.getLayer("b-fill")) return;
  map.setPaintProperty("b-fill", "fill-extrusion-color", colorExpression());
  map.setPaintProperty("b-fill", "fill-extrusion-height", heightExpression());
  // Outlines lie on the ground: in 3D they go below the buildings, or they show through them.
  if (state.threeD) for (const id of ["b-line", "b-glow", "b-surveyed"]) map.moveLayer(id, "b-fill");
  else map.moveLayer("b-fill", "b-line");
  for (const id of ["b-line", "b-surveyed"]) map.setPaintProperty(id, "line-color", OUTLINE[tone()]);
  map.setPaintProperty("b-glow", "line-opacity", tone() === "dark" ? 0.55 : 0);
  renderLegend();
}

function applySurvey() {
  if (!map.getLayer("b-fill")) return;
  map.setFilter("b-fill", surveyFilter());
  map.setFilter("b-line", surveyFilter());
  map.setFilter("b-surveyed", surveyFilter(surveyed));
  map.setFilter("b-glow", surveyFilter(surveyed));
}

// Marks the current experiment's test buildings with the chosen model's prediction.
function applyPredictions() {
  if (!map.getSource("b")) return;
  map.removeFeatureState({ source: "b", sourceLayer: LAYER });
  testIndex = new Map();
  const x = experimentData();
  if (!x) return;
  const { d } = x;
  d.keys.forEach((k, i) => {
    testIndex.set(k, i);
    const p = d.pred[state.model][i];
    const votes = {};
    Object.values(d.pred).forEach((preds) => { votes[preds[i]] = (votes[preds[i]] || 0) + 1; });
    map.setFeatureState({ source: "b", sourceLayer: LAYER, id: k },
      { t: true, p, ok: p === d.true[i], ag: Math.max(...Object.values(votes)) });
  });
}

// ---------------------------------------------------------------- panels
function renderLegend() {
  const v = VIEWS[state.view];
  const swatch = (c, label) => `<div class="item"><span class="swatch" style="background:${c}"></span>${label}</div>`;
  let html = `<h2>${v.label}</h2>`;
  if (["class", "cityPred"].includes(v.type) || state.view === "pred") {
    html += Object.entries(CLASS_COLORS[tone()]).map(([k, c]) => swatch(c, `${CLASS_NAMES[k]} (${k})`)).join("");
  } else if (v.type === "cat") {
    html += Object.values(v.values).map(([label, c]) => swatch(c, label)).join("");
  } else if (v.type === "ramp") {
    html += `<div class="ramp" style="background:linear-gradient(to right, ${RAMP.join(",")})"></div>
      <div class="ends"><span>${v.domain[0]} or less</span><span>${v.domain[1]} or more</span></div>`;
  } else if (state.view === "correct") {
    html += swatch("#2e9e5b", "Correct") + swatch("#d64541", "Wrong");
  } else {
    html += swatch("#2c7fb8", "All five models agree") + swatch("#7fcdbb", "Four agree")
      + swatch("#fdae61", "Three agree") + swatch("#d7191c", "Two or fewer agree");
  }
  if (v.type === "pred") html += swatch(NO_DATA[tone()], "Not in this experiment's test set");
  else if (v.type === "class") html += swatch(NO_DATA[tone()], "Not surveyed");
  else html += swatch(NO_DATA[tone()], "No data");
  // Surveyed buildings only get a different border, so the symbol is an empty square in that border style.
  const ring = tone() === "light" ? "border:2px solid #000" : "border:2px solid #fff;box-shadow:0 0 0 1px #888,0 0 5px 1px rgba(255,255,255,0.9)";
  html += `<div class="item"><span class="swatch" style="background:transparent;${ring}"></span>Surveyed building</div>`;
  $("legend").innerHTML = html;
}

function renderResults() {
  const x = experimentData();
  $("right").classList.toggle("hidden", !x);
  if (!x) return;
  const { e, d } = x;
  const s = d.scores[state.model];
  $("resultsTitle").textContent = `${e.title} · ${MODEL_NAMES[state.model]}`;
  $("scores").innerHTML = [["Test buildings", d.n.toLocaleString()], ["Macro F1", s.macro_f1.toFixed(2)],
    ["Accuracy", s.accuracy.toFixed(2)], ["Cohen's κ", s.kappa.toFixed(2)], ["True Skill Statistic", s.tss.toFixed(2)]]
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  const cls = e.classes;
  // "Predicted" runs along the columns, "Surveyed" (turned 90°) along the rows.
  let html = `<tr><td colspan="2" class="corner"></td><th colspan="${cls.length}" class="axis">Predicted</th></tr>`
    + `<tr><td colspan="2" class="corner"></td>${cls.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  s.confusion.forEach((row, i) => {
    const pcts = s.percent[i];  // precomputed, rounded as in Figure 4
    const axis = i === 0 ? `<th rowspan="${cls.length}" class="axis vertical"><span>Surveyed</span></th>` : "";
    html += `<tr>${axis}<th>${cls[i]}</th>` + row.map((n, j) => {
      if (!pcts) return "<td>–</td>";
      const pct = pcts[j];
      return `<td class="${pct > 60 ? "dark" : ""}" style="background:rgba(var(--cell),${(pct / 100 * 0.85).toFixed(2)})">`
        + `${pct}%<small>(${n})</small></td>`;
    }).join("") + "</tr>";
  });
  $("confusion").innerHTML = html;
}

function fillExperiments() {
  const names = Object.entries(meta.experiments).filter(([, e]) => e.cities[state.city]);
  $("experiment").innerHTML = names.map(([k, e]) => `<option value="${k}">${e.title}</option>`).join("");
  state.experiment = names.some(([k]) => k === state.city) ? state.city : names[0][0];
  $("experiment").value = state.experiment;
  fillModels(true);
}

function fillModels(pickBest) {
  const d = experimentData()?.d;
  $("model").innerHTML = meta.models.map((m) => `<option value="${m}">${MODEL_NAMES[m]}${d && m === d.best ? " (best)" : ""}</option>`).join("");
  if (pickBest || !state.model) state.model = d ? d.best : meta.models[0];
  $("model").value = state.model;
}

function fillSurveys() {
  $("survey").innerHTML = `<option value="all">All surveys</option>`
    + meta.surveys[state.city].map((v) => `<option value="${v}">${v}</option>`).join("");
  state.survey = "all";
}

// ---------------------------------------------------------------- popup
map.on("click", "b-fill", (ev) => {
  const p = ev.features[0].properties;
  const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(2)) : String(v).replace(/_/g, " "));
  let html = `<div class="popup"><b>${p.cls ? `Surveyed: ${CLASS_NAMES[p.cls]} (${p.cls})` : "Not surveyed"}</b><table>`;
  html += POPUP_FIELDS.filter(([k]) => p[k] !== undefined && p[k] !== null)
    .map(([k, label]) => `<tr><td>${label}</td><td>${fmt(p[k])}</td></tr>`).join("");
  html += `</table><b>City model predictions</b><table>`;
  html += meta.models.map((m) => {
    const pr = p[`p_${meta.model_keys[m]}`];
    return `<tr><td>${MODEL_NAMES[m]}</td><td class="${p.cls && pr && pr !== p.cls ? "wrong" : ""}">${pr ?? "–"}</td></tr>`;
  }).join("") + "</table>";
  const x = experimentData();
  if (x && testIndex.has(p.k) && state.experiment !== state.city) {
    const i = testIndex.get(p.k);
    html += `<b>${x.e.title}</b><table>` + meta.models.map((m) => {
      const pr = x.d.pred[m][i];
      return `<tr><td>${MODEL_NAMES[m]}</td><td class="${pr === p.cls ? "" : "wrong"}">${pr}</td></tr>`;
    }).join("") + "</table>";
  }
  new maplibregl.Popup({ maxWidth: "300px" }).setLngLat(ev.lngLat).setHTML(html + "</div>").addTo(map);
});
map.on("mouseenter", "b-fill", () => { map.getCanvas().style.cursor = "pointer"; });
map.on("mouseleave", "b-fill", () => { map.getCanvas().style.cursor = ""; });

// ---------------------------------------------------------------- controls
function fitCity() {
  const [w, s, e, n] = meta.extents[state.city][state.survey];
  const wide = window.innerWidth > 800;
  map.fitBounds([[w, s], [e, n]], { pitch: state.threeD ? 50 : 0, duration: 1200,
    padding: wide ? { top: 40, bottom: 40, left: 340, right: 330 } : 30 });
}

function selectCity(city) {
  state.city = city;
  fillSurveys();
  fillExperiments();
  removeLayers();
  addLayers();
  restyle();
  renderResults();
  fitCity();
}

async function init() {
  meta = await (await fetch("data/experiments.json")).json();
  $("city").innerHTML = Object.entries(meta.cities).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  $("city").value = state.city;
  $("colorBy").innerHTML = Object.entries(VIEWS).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join("");
  $("colorBy").value = state.view;
  $("basemap").innerHTML = Object.entries(BASEMAPS).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join("");

  $("city").addEventListener("change", (e) => selectCity(e.target.value));
  $("survey").addEventListener("change", (e) => { state.survey = e.target.value; applySurvey(); fitCity(); });
  $("colorBy").addEventListener("change", (e) => { state.view = e.target.value; restyle(); });
  $("experiment").addEventListener("change", (e) => {
    state.experiment = e.target.value; fillModels(true); applyPredictions(); restyle(); renderResults();
  });
  $("model").addEventListener("change", (e) => { state.model = e.target.value; applyPredictions(); restyle(); renderResults(); });
  $("basemap").addEventListener("change", (e) => setBasemap(e.target.value));
  $("closePlots").addEventListener("click", () => setPlots(false));
  window.addEventListener("resize", placePlots);
  document.addEventListener("keydown", (e) => {
    if ((e.key === "p" || e.key === "P") && !(e.target instanceof HTMLSelectElement)) {
      setPresentation(!document.body.classList.contains("presentation"));
    }
  });
  if (new URLSearchParams(window.location.search).has("presentation")) setPresentation(true);


  fillSurveys();
  fillExperiments();
  $("btn3d").classList.add("on");
  setPlots(true);
  placePlots();
  const start = () => { addLayers(); restyle(); renderResults(); fitCity(); collapseAttribution(); };
  if (map.isStyleLoaded()) start(); else map.once("load", start);
}

init();
