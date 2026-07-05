// vivarium-tyssue viewer — renders exported tyssue vertex-model runs as an
// interactive mesh: faces fan-filled from cell centroids, edge wireframe, 2D
// (orthographic) or 3D (orbit) camera, color-by a per-cell field, full playback
// transport, and hover cell inspection. Zero build: three.js via importmap.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Surface any runtime error on-page (a blank WebGL canvas is otherwise silent).
window.addEventListener("error", (e) => {
  const el = document.getElementById("loading");
  if (el) { el.style.display = "flex"; el.style.color = "#ff6b6b";
    el.textContent = "viewer error: " + (e.message || e.error); }
});

const $ = (s) => document.querySelector(s);
const wrap = $("#canvas-wrap");
const loadingEl = $("#loading");
const tip = $("#hover-tip");
const badge = $("#viewbadge");
const scrub = $("#scrub"), playBtn = $("#play"), speedEl = $("#speed"), loopEl = $("#loop");
const frameLabel = $("#framelabel"), colormodeEl = $("#colormode");
const showEdgesEl = $("#showedges"), showVertsEl = $("#showverts"), spinEl = $("#spin");
const statsEl = $("#stats"), sparkCanvas = $("#spark"), sparkLabel = $("#sparklabel");
const cbar = $("#colorbar"), cbLabel = $("#cbLabel"), cbMin = $("#cbMin"), cbMax = $("#cbMax");
const legendSection = $("#legend-section"), legendEl = $("#legend");

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
wrap.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();

let camera, controls, current = null, playing = false, frameIdx = 0, lastStep = 0;
let colorMode = "type", hoveredFace = -1;

// ---- color helpers ----
function heat(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[13,17,23],[40,40,110],[40,100,140],[30,150,130],[120,195,70],[250,232,60]];
  const f = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f));
  const a = f - i, s0 = stops[i], s1 = stops[i + 1];
  return [(s0[0]+(s1[0]-s0[0])*a)/255, (s0[1]+(s1[1]-s0[1])*a)/255, (s0[2]+(s1[2]-s0[2])*a)/255];
}
function cellJit(f) {
  const j = ((f * 2654435761) >>> 0) / 4294967295;
  const c = new THREE.Color(); c.setHSL(j, 0.55, 0.55);
  return [c.r, c.g, c.b];
}
// categorical palette for cell types
const TYPE_HUES = [0.58, 0.09, 0.33, 0.78, 0.50, 0.00, 0.16, 0.66, 0.42, 0.88, 0.25, 0.72];
function typeRGB(t) {
  const c = new THREE.Color(); c.setHSL(TYPE_HUES[t % TYPE_HUES.length], 0.6, 0.55);
  return [c.r, c.g, c.b];
}
function typeSwatch(t) {
  const c = typeRGB(t);
  return `rgb(${(c[0]*255)|0},${(c[1]*255)|0},${(c[2]*255)|0})`;
}
const FLAT = [0.30, 0.62, 0.86];

// ---- lights ----
function ensureLights() {
  if (scene.userData.lit) return;
  scene.add(new THREE.AmbientLight(0xffffff, 0.72));
  const d = new THREE.DirectionalLight(0xffffff, 0.85); d.position.set(0.5, 0.8, 1.0); scene.add(d);
  const d2 = new THREE.DirectionalLight(0x88aaff, 0.3); d2.position.set(-0.6, -0.4, -0.7); scene.add(d2);
  scene.userData.lit = true;
}
function clearScene() {
  for (let i = scene.children.length - 1; i >= 0; i--) {
    const o = scene.children[i];
    if (o.isLight) continue;
    scene.remove(o); o.geometry?.dispose?.(); o.material?.dispose?.();
  }
}

// topology for a frame: model-level when static, else per-frame
function topo(model, fr) {
  return model.static_topology
    ? { tris: model.tris, edges: model.edges, face_of_tri: model.face_of_tri }
    : { tris: fr.tris, edges: fr.edges, face_of_tri: fr.face_of_tri };
}

// Derive per-face fan centroids from vertex positions (mean of the face's
// vertices). Lets big static-topology models omit the per-frame `centroids`
// array entirely — the single largest chunk of the file — and reconstruct it
// here. `tris` are [srce, trgt, Nv+face] triples, so T[t*3] is a face vertex.
function deriveCentroids(P, tp, nFaces) {
  const T = tp.tris, fot = tp.face_of_tri, ntri = T.length / 3;
  const sum = new Float32Array(nFaces * 3), cnt = new Uint32Array(nFaces);
  for (let t = 0; t < ntri; t++) {
    const f = fot[t], s = T[t * 3];
    sum[f*3] += P[s*3]; sum[f*3+1] += P[s*3+1]; sum[f*3+2] += P[s*3+2]; cnt[f]++;
  }
  for (let f = 0; f < nFaces; f++) {
    const c = cnt[f] || 1; sum[f*3] /= c; sum[f*3+1] /= c; sum[f*3+2] /= c;
  }
  return sum;
}

// global min/max per field across all frames, for a stable color scale
function fieldRange(model, field) {
  let mn = Infinity, mx = -Infinity;
  for (const fr of model.frames) {
    const a = fr.fields[field]; if (!a) continue;
    for (const v of a) { if (v < mn) mn = v; if (v > mx) mx = v; }
  }
  if (!isFinite(mn)) { mn = 0; mx = 1; }
  return { mn, mx: mx > mn ? mx : mn + 1 };
}

function setupModel(model) {
  clearScene(); ensureLights();
  // buffer sizes: max triangles / edges / verts across frames
  let maxTri = 0, maxEdge = 0, maxPos = 0;
  for (const fr of model.frames) {
    const tp = topo(model, fr);
    maxTri = Math.max(maxTri, tp.tris.length / 3);
    maxEdge = Math.max(maxEdge, tp.edges.length / 2);
    const cLen = fr.centroids ? fr.centroids.length : model.n_cells * 3;
    maxPos = Math.max(maxPos, (fr.verts.length + cLen) / 3);
  }
  // face mesh (triangle soup so each cell gets a flat color)
  const mg = new THREE.BufferGeometry();
  const mpos = new Float32Array(maxTri * 9), mcol = new Float32Array(maxTri * 9);
  mg.setAttribute("position", new THREE.BufferAttribute(mpos, 3));
  mg.setAttribute("color", new THREE.BufferAttribute(mcol, 3));
  const mesh = new THREE.Mesh(mg, new THREE.MeshStandardMaterial({
    vertexColors: true, flatShading: true, side: THREE.DoubleSide,
    roughness: 0.85, metalness: 0.0 }));
  scene.add(mesh);
  // edge wireframe
  const eg = new THREE.BufferGeometry();
  const epos = new Float32Array(maxEdge * 6);
  eg.setAttribute("position", new THREE.BufferAttribute(epos, 3));
  const lines = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({
    color: 0x0b0f14, transparent: true, opacity: 0.55 }));
  scene.add(lines);
  // vertices
  const vg = new THREE.BufferGeometry();
  vg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(maxPos * 3), 3));
  const pts = new THREE.Points(vg, new THREE.PointsMaterial({ color: 0xe6edf3, size: 3, sizeAttenuation: false }));
  pts.visible = false; scene.add(pts);

  const center = [0, 1, 2].map((i) => (model.bounds[0][i] + model.bounds[1][i]) / 2);
  const span = Math.max(...[0, 1, 2].map((i) => model.bounds[1][i] - model.bounds[0][i]), 1e-3);
  setupCamera(model, center, span);

  current = {
    model, mesh, lines, pts, mpos, mcol, epos,
    ranges: { area: fieldRange(model, "area"), perimeter: fieldRange(model, "perimeter"),
              num_sides: fieldRange(model, "num_sides") },
    spark: null,
    render(fi) {
      const fr = model.frames[fi], tp = topo(model, fr);
      const P = fr.verts, nV = P.length / 3;
      const Cn = fr.centroids || deriveCentroids(P, tp, model.n_cells);
      const pos = (idx) => idx < nV
        ? [P[idx*3], P[idx*3+1], P[idx*3+2]]
        : [Cn[(idx-nV)*3], Cn[(idx-nV)*3+1], Cn[(idx-nV)*3+2]];
      // faces
      const T = tp.tris, fot = tp.face_of_tri, ntri = T.length / 3;
      const rng = this.ranges[colorMode];
      const fld = fr.fields[colorMode];
      const ctype = fr.fields.cell_type;
      for (let t = 0; t < ntri; t++) {
        const f = fot[t];
        let rgb;
        if (colorMode === "type") rgb = ctype ? typeRGB(ctype[f]) : cellJit(f);
        else if (colorMode === "cell") rgb = cellJit(f);
        else if (colorMode === "uniform") rgb = FLAT;
        else if (!fld) rgb = cellJit(f);  // field absent in this model → per-cell tint
        else rgb = heat((fld[f] - rng.mn) / (rng.mx - rng.mn));
        if (f === hoveredFace) rgb = [1, 1, 1];
        for (let k = 0; k < 3; k++) {
          const vi = T[t*3+k], p = pos(vi), o = (t*3+k)*3;
          this.mpos[o]=p[0]; this.mpos[o+1]=p[1]; this.mpos[o+2]=p[2];
          this.mcol[o]=rgb[0]; this.mcol[o+1]=rgb[1]; this.mcol[o+2]=rgb[2];
        }
      }
      mesh.geometry.setDrawRange(0, ntri * 3);
      mesh.geometry.attributes.position.needsUpdate = true;
      mesh.geometry.attributes.color.needsUpdate = true;
      mesh.geometry.computeVertexNormals();
      mesh.geometry.computeBoundingSphere();
      // edges
      const E = tp.edges, ne = E.length / 2;
      for (let e = 0; e < ne; e++) {
        const a = pos(E[e*2]), b = pos(E[e*2+1]), o = e*6;
        this.epos[o]=a[0]; this.epos[o+1]=a[1]; this.epos[o+2]=a[2];
        this.epos[o+3]=b[0]; this.epos[o+4]=b[1]; this.epos[o+5]=b[2];
      }
      lines.geometry.setDrawRange(0, ne * 2);
      lines.geometry.attributes.position.needsUpdate = true;
      // verts
      const vpos = pts.geometry.attributes.position.array;
      for (let i = 0; i < P.length; i++) vpos[i] = P[i];
      pts.geometry.setDrawRange(0, nV);
      pts.geometry.attributes.position.needsUpdate = true;
    },
    pick(faceIndex) {  // three.js triangle index -> tyssue face id
      const tp = topo(model, model.frames[frameIdx]);
      return faceIndex != null && faceIndex < tp.face_of_tri.length ? tp.face_of_tri[faceIndex] : -1;
    },
  };
}

function setupCamera(model, center, span) {
  controls?.dispose?.(); controls = null;
  const w = wrap.clientWidth || 1, h = wrap.clientHeight || 1;
  if (model.is3d) {
    const cam = new THREE.PerspectiveCamera(45, w / h, span * 0.01, span * 100);
    cam.position.set(center[0] + span * 1.6, center[1] + span * 1.1, center[2] + span * 1.8);
    camera = cam;
    controls = new OrbitControls(cam, renderer.domElement);
    controls.target.set(center[0], center[1], center[2]);
    controls.enableDamping = true; controls.update();
    badge.textContent = "3D · orbit / scroll to zoom";
  } else {
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, -span * 10, span * 10);
    cam.position.set(center[0], center[1], center[2] + span);
    cam.lookAt(center[0], center[1], center[2]);
    camera = cam;
    controls = new OrbitControls(cam, renderer.domElement);
    controls.target.set(center[0], center[1], center[2]);
    controls.enableRotate = false; controls.enableDamping = false;
    controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controls.update();
    camera.userData.span = span; camera.userData.center = center;
    fitOrtho();
    badge.textContent = "2D · drag to pan, scroll to zoom";
  }
}
function fitOrtho() {
  if (!current || current.model.is3d) return;
  const w = wrap.clientWidth, h = wrap.clientHeight, span = camera.userData.span * 0.62, va = w / h;
  camera.left = -span * va; camera.right = span * va; camera.top = span; camera.bottom = -span;
  camera.updateProjectionMatrix();
}

// ---------- measurements ----------
function meanArea(fi) {
  const a = current.model.frames[fi].fields.area; if (!a) return 0;
  let s = 0; for (const v of a) s += v; return a.length ? s / a.length : 0;
}
function computeSpark() {
  current.spark = current.model.frames.map((_, i) => meanArea(i));
}
function drawSpark() {
  const arr = current.spark; if (!arr) return;
  const cv = sparkCanvas, ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const mx = Math.max(...arr), mn = Math.min(...arr), span = mx - mn || 1, pad = 4;
  ctx.strokeStyle = "#4aa3ff"; ctx.lineWidth = 1.5; ctx.beginPath();
  arr.forEach((v, i) => {
    const x = pad + (W - 2*pad) * (arr.length > 1 ? i/(arr.length-1) : 0);
    const y = H - pad - (H - 2*pad) * (v - mn) / span;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  const cx = pad + (W - 2*pad) * (arr.length > 1 ? frameIdx/(arr.length-1) : 0);
  const cy = H - pad - (H - 2*pad) * (arr[frameIdx] - mn) / span;
  ctx.fillStyle = "#e6edf3"; ctx.beginPath(); ctx.arc(cx, cy, 2.6, 0, 7); ctx.fill();
}
function updateStats() {
  const fr = current.model.frames[frameIdx];
  const nCells = fr.fields.num_sides.length;
  const rows = [["cells", nCells.toLocaleString()],
                ["mean area", meanArea(frameIdx).toFixed(3)]];
  if (fr.fields.perimeter) {
    const p = fr.fields.perimeter; let s = 0; for (const v of p) s += v;
    rows.push(["mean perimeter", (s / p.length).toFixed(3)]);
  }
  statsEl.innerHTML = rows.map(([k,v]) => `<div class="stat"><span>${k}</span><span>${v}</span></div>`).join("");
  sparkLabel.textContent = `mean cell area over ${current.spark.length} frames`;
  drawSpark();
}

// ---------- color bar ----------
function updateColorbar() {
  if (colorMode === "cell" || colorMode === "uniform" || colorMode === "type") {
    cbar.style.display = "none"; return;
  }
  const r = current.ranges[colorMode];
  cbar.style.display = ""; cbLabel.textContent = colorMode.replace("_", " ");
  cbMin.textContent = r.mn.toFixed(colorMode === "num_sides" ? 0 : 2);
  cbMax.textContent = r.mx.toFixed(colorMode === "num_sides" ? 0 : 2);
}

// legend for cell types — shown when the model has named types and type mode is on
function buildLegend() {
  const names = current.model.type_names;
  if (!names || colorMode !== "type") { legendSection.style.display = "none"; return; }
  legendSection.style.display = "";
  legendEl.innerHTML = names.map((nm, t) =>
    `<div class="stat"><span><span style="display:inline-block;width:11px;height:11px;` +
    `border-radius:3px;margin-right:7px;vertical-align:-1px;background:${typeSwatch(t)}"></span>` +
    `${nm || "(none)"}</span></div>`).join("");
}

// ---------- model loading ----------
async function loadModel(entry) {
  loadingEl.style.display = "flex"; loadingEl.textContent = `loading ${entry.name}…`;
  // Cache-bust per-model on the content hash so a redeploy that changes the data
  // changes the URL — browsers/CDN can't serve a stale copy of an updated demo.
  const bust = entry.version ? "?v=" + entry.version : "";
  const model = await (await fetch("./data/" + entry.file + bust)).json();
  playing = false; playBtn.textContent = "▶"; hoveredFace = -1; tip.style.display = "none";
  colorMode = "type"; colormodeEl.value = "type";
  setupModel(model);
  frameIdx = 0; scrub.max = String(model.frames.length - 1); scrub.value = "0";
  computeSpark(); current.render(0);
  current.lines.visible = showEdgesEl.checked; current.pts.visible = showVertsEl.checked;
  updateFrameLabel(); updateStats(); updateColorbar(); buildLegend();
  $("#info").innerHTML = `<h2>${model.name}</h2><div class="desc">${model.blurb || ""}</div>` +
    `<div class="meta">${model.n_cells.toLocaleString()} cells · ${model.n_verts} vertices · ` +
    `${model.frames.length} frames · ${model.is3d ? "3D surface" : "2D sheet"}</div>`;
  loadingEl.style.display = "none"; onResize();
}

function updateFrameLabel() {
  const f = current.model.frames[frameIdx];
  frameLabel.textContent = `frame ${frameIdx}/${current.model.frames.length-1} · t=${f.t}`;
}
function showFrame(i) {
  frameIdx = Math.max(0, Math.min(current.model.frames.length-1, i|0));
  current.render(frameIdx); scrub.value = String(frameIdx);
  updateFrameLabel(); updateStats();
}

// ---------- transport ----------
function pause() { playing = false; playBtn.textContent = "▶"; }
scrub.addEventListener("input", () => { pause(); showFrame(+scrub.value); });
playBtn.addEventListener("click", () => {
  if (!current) return;
  playing = !playing; playBtn.textContent = playing ? "❚❚" : "▶";
  if (playing && frameIdx >= current.model.frames.length-1) showFrame(0);
});
$("#first").addEventListener("click", () => { pause(); showFrame(0); });
$("#stepback").addEventListener("click", () => { pause(); showFrame(frameIdx-1); });
$("#stepfwd").addEventListener("click", () => { pause(); showFrame(frameIdx+1); });
colormodeEl.addEventListener("change", () => {
  colorMode = colormodeEl.value; current.render(frameIdx); updateColorbar(); buildLegend();
});
showEdgesEl.addEventListener("change", () => { if (current) current.lines.visible = showEdgesEl.checked; });
showVertsEl.addEventListener("change", () => { if (current) current.pts.visible = showVertsEl.checked; });

// ---------- hover ----------
wrap.addEventListener("mousemove", (e) => {
  if (!current) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  ndc.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(current.mesh)[0];
  const face = hit ? current.pick(hit.faceIndex) : -1;
  if (face !== hoveredFace) { hoveredFace = face; current.render(frameIdx); }
  if (face < 0) { tip.style.display = "none"; return; }
  const fr = current.model.frames[frameIdx];
  const area = fr.fields.area ? fr.fields.area[face].toFixed(3) : "—";
  const per = fr.fields.perimeter ? fr.fields.perimeter[face].toFixed(3) : "—";
  const sides = fr.fields.num_sides ? fr.fields.num_sides[face] : "—";
  const names = current.model.type_names, ct = fr.fields.cell_type;
  const typeLine = names && ct ? `${names[ct[face]] || "(none)"} · ` : "";
  tip.innerHTML = `<b>cell ${face}</b> · ${typeLine}${sides} sides<br>area ${area} · perim ${per}`;
  tip.style.display = "block";
  const wr = wrap.getBoundingClientRect();
  let lx = e.clientX - wr.left + 14, ly = e.clientY - wr.top + 14;
  if (lx + 170 > wr.width) lx = e.clientX - wr.left - 170;
  tip.style.left = lx + "px"; tip.style.top = ly + "px";
});
wrap.addEventListener("mouseleave", () => {
  tip.style.display = "none"; if (hoveredFace !== -1) { hoveredFace = -1; if (current) current.render(frameIdx); }
});

// ---------- loop ----------
function onResize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  if (!current) return;
  if (current.model.is3d) { camera.aspect = w / h; camera.updateProjectionMatrix(); }
  else fitOrtho();
}
window.addEventListener("resize", onResize);

function animate(t) {
  requestAnimationFrame(animate);
  if (playing && current) {
    const interval = 1000 / (+speedEl.value);
    if (t - lastStep > interval) {
      lastStep = t;
      if (frameIdx < current.model.frames.length-1) showFrame(frameIdx+1);
      else if (loopEl.checked) showFrame(0);
      else pause();
    }
  }
  if (controls) {
    controls.autoRotate = spinEl.checked && current && current.model.is3d;
    controls.autoRotateSpeed = 1.5;
    controls.update();
  }
  if (camera) renderer.render(scene, camera);
}
requestAnimationFrame(animate);

// ---------- boot ----------
(async function () {
  // Always fetch a fresh manifest (it carries each model's current version token);
  // the models themselves are then cache-busted by that token in loadModel.
  const { models } = await (await fetch("./data/index.json?t=" + Date.now())).json();
  const ul = $("#models"); ul.innerHTML = "";
  models.forEach((m) => {
    const li = document.createElement("li");
    li.innerHTML = `<div class="mname">${m.name}</div>` +
      `<div class="mmeta">${m.is3d ? "3D" : "2D"} · ${(m.n_cells||0).toLocaleString()} cells · ${m.n_frames} frames</div>`;
    li.addEventListener("click", () => {
      [...ul.children].forEach((c) => c.classList.remove("active"));
      li.classList.add("active"); loadModel(m);
    });
    ul.appendChild(li);
  });
  loadingEl.textContent = "select a run →";
  if (models.length) ul.children[0].click();
})();
