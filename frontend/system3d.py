"""Interactive 3D model of the analyzed star system (Three.js inside a Streamlit iframe).

Everything in the scene comes from the measurement, not from a picture library:

* star color      <- effective temperature (TIC Teff; estimated from the radius if missing)
* star size       <- stellar radius (scene unit = 1 stellar radius)
* companion size  <- transit depth:  Rp / R* = sqrt(depth)
* orbit size      <- period + stellar mass (Kepler's third law)
* orbit tilt      <- transit duration (impact parameter b, then inclination)
* companion type  <- the classifier: planet (PC), second star (EB) or "unknown" (FP)

The "What the telescope sees" strip is computed live from the overlap of the two disks as seen
from Earth, so the dip in the animation has the same geometry as the real one.
Three.js (MIT) is vendored in ``frontend/static`` so the demo works without internet.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

STATIC = Path(__file__).resolve().parent / "static"
SUN_TEFF = 5772.0
RSUN_PER_AU = 215.032
REARTH_PER_RSUN = 109.2


@lru_cache(maxsize=1)
def _three_js() -> str:
    return (STATIC / "three.min.js").read_text(encoding="utf-8")


def teff_to_hex(teff: float) -> str:
    """Approximate blackbody color of a star (Tanner Helland's fit), as #rrggbb."""
    t = min(max(teff, 1000.0), 40000.0) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    r, g, b = (int(min(max(v, 0.0), 255.0)) for v in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def spectral_type(teff: float) -> str:
    for limit, letter in (
        (30000, "O"),
        (10000, "B"),
        (7500, "A"),
        (6000, "F"),
        (5200, "G"),
        (3700, "K"),
    ):
        if teff >= limit:
            return letter
    return "M"


def planet_kind(radius_earth: float | None) -> tuple[str, str]:
    """(look, human label) for a planet of the given radius."""
    if radius_earth is None:
        return "gas", "planet"
    if radius_earth < 1.25:
        return "rocky", "Earth-size rocky planet"
    if radius_earth < 2.0:
        return "rocky", "super-Earth"
    if radius_earth < 4.0:
        return "ice", "sub-Neptune"
    if radius_earth < 8.0:
        return "ice", "Neptune-like planet"
    return "gas", "gas giant"


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def system_params(result: dict[str, Any], cand: dict[str, Any] | None) -> dict[str, Any]:
    """Physical parameters for the 3D scene, derived from an analysis result and one candidate."""
    stellar = result.get("stellar") or {}
    radius = _num(stellar.get("radius"))
    teff = _num(stellar.get("teff"))
    mass = _num(stellar.get("mass"))
    assumed: list[str] = []
    if radius is None:
        radius = 1.0
        assumed.append("radius")
    if teff is None:
        # Main-sequence-ish guess from the radius; giants keep a solar-like color.
        teff = SUN_TEFF * min(max(radius, 0.25), 1.6) ** 0.6 if radius <= 1.6 else 5000.0
        assumed.append("teff")
    if mass is None:
        mass = min(max(radius**1.25, 0.1), 3.0) if radius <= 2.0 else 1.2
        assumed.append("mass")

    out: dict[str, Any] = {
        "name": result.get("name") or result.get("target_id", "Star"),
        "target_id": result.get("target_id", ""),
        "mission": (result.get("lightcurve") or {}).get("mission", ""),
        "star": {
            "radius_rsun": round(radius, 3),
            "teff": round(teff),
            "mass_msun": round(mass, 3),
            "color": teff_to_hex(teff),
            "spectral_type": spectral_type(teff),
            "luminosity_lsun": round(radius**2 * (teff / SUN_TEFF) ** 4, 4),
        },
        "assumed": assumed,
        "companion": None,
    }
    if not cand or not cand.get("significant"):
        return out

    period = float(cand["period"])
    depth = (cand.get("physical") or {}).get("fit_depth_ppm") or cand.get("depth_ppm") or 0.0
    depth = min(max(float(depth) / 1e6, 1e-6), 0.95)
    k = math.sqrt(depth)
    a_au = (mass * (period / 365.25) ** 2) ** (1.0 / 3.0)
    a_over_r = max(a_au * RSUN_PER_AU / radius, 1.0 + k + 0.05)
    dur = (
        float((cand.get("physical") or {}).get("fit_duration_hours") or cand["duration_hours"])
        / 24.0
    )
    x = math.pi * dur * a_over_r / period
    b2 = (1.0 + k) ** 2 - x**2
    b = math.sqrt(b2) if b2 > 0 else 0.0
    b = min(b, 1.0 + k - 0.02)
    incl = math.degrees(math.acos(min(b / a_over_r, 1.0)))

    pred = cand.get("predicted_class")
    kind = {"PC": "planet", "EB": "star"}.get(pred or "", "unknown")
    rp_earth = (cand.get("physical") or {}).get("planet_radius_rearth")
    rp_earth = float(rp_earth) if rp_earth else k * radius * REARTH_PER_RSUN
    look, label = planet_kind(rp_earth)
    teq = teff * math.sqrt(1.0 / (2.0 * a_over_r))
    comp: dict[str, Any] = {
        "kind": kind,
        "predicted_class": pred,
        "confidence": cand.get("confidence"),
        "period_days": round(period, 5),
        "depth_ppm": round(depth * 1e6, 1),
        "k": round(k, 5),
        "a_au": round(a_au, 5),
        "a_over_r": round(a_over_r, 3),
        "impact_b": round(b, 3),
        "inclination_deg": round(incl, 2),
        "radius_earth": round(rp_earth, 2),
        "look": look,
        "label": label,
        "teq_k": round(teq),
    }
    if kind == "star":
        t2 = 0.75 * teff
        comp.update(
            teff=round(t2),
            color=teff_to_hex(t2),
            label="companion star",
            radius_rsun=round(k * radius, 3),
        )
    elif kind == "unknown":
        comp["label"] = "unconfirmed signal"
    out["companion"] = comp
    # Habitable zone (Kopparapu-style rough bounds), in AU.
    lum = out["star"]["luminosity_lsun"]
    out["habitable_zone_au"] = [round(math.sqrt(lum / 1.1), 4), round(math.sqrt(lum / 0.53), 4)]
    return out


def scene_html(params: dict[str, Any], height: int = 520) -> str:
    """Self-contained HTML document for the scene (three.js inlined)."""
    return (
        _TEMPLATE.replace("__HEIGHT__", str(height))
        .replace("__PARAMS__", json.dumps(params))
        .replace("__THREE__", _three_js())
    )


def render_system(result: dict[str, Any], cand: dict[str, Any] | None, height: int = 520) -> None:
    import streamlit.components.v1 as components

    components.html(scene_html(system_params(result, cand), height), height=height + 4)


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden;
  font-family:"Source Sans Pro",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#f0efec}
#wrap{position:relative;width:100%;height:__HEIGHT__px;border-radius:16px;overflow:hidden;
  background:radial-gradient(ellipse at 30% 15%,#101a30 0%,#070912 55%,#04050a 100%);border:1px solid #2c2c2a}
#gl{display:block;width:100%;height:100%;cursor:grab;touch-action:none}
#gl:active{cursor:grabbing}
.panel{position:absolute;background:rgba(9,11,19,.66);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  border:1px solid rgba(255,255,255,.09);border-radius:12px;font-size:12px}
#info{top:12px;left:12px;width:236px;padding:11px 13px}
#info h3{margin:0;font-size:16px;font-weight:700;letter-spacing:.2px}
.sub{color:#8a897f;font-size:11.5px;margin-top:1px}
.row{display:flex;justify-content:space-between;gap:12px;margin-top:5px}
.row span:first-child{color:#8a897f}
.row span:last-child{text-align:right}
.chip{display:inline-flex;align-items:center;gap:6px;margin-top:8px;padding:3px 9px;border-radius:999px;font-size:11.5px;
  border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.04)}
.dot{width:8px;height:8px;border-radius:50%}
#controls{top:12px;right:12px;display:flex;flex-direction:column;gap:6px;padding:8px}
button{font:inherit;font-size:12px;background:rgba(255,255,255,.05);color:#f0efec;border:1px solid rgba(255,255,255,.12);
  border-radius:8px;padding:6px 10px;cursor:pointer;text-align:left;white-space:nowrap}
button:hover{background:rgba(57,135,229,.22);border-color:#3987e5}
button.on{background:rgba(57,135,229,.32);border-color:#3987e5}
#lc{left:12px;right:12px;bottom:12px;height:72px;padding:6px 12px 5px}
#lchead{display:flex;justify-content:space-between;color:#8a897f;font-size:11.5px}
#lchead b{color:#f0efec;font-weight:600}
#lcc{width:100%;height:48px;display:block;margin-top:3px}
#note{position:absolute;right:14px;bottom:92px;font-size:11px;color:#8a897f;text-align:right;pointer-events:none}
.lbl{position:absolute;transform:translate(-50%,-140%);font-size:11.5px;color:#f0efec;pointer-events:none;
  background:rgba(9,11,19,.6);padding:2px 7px;border-radius:6px;border:1px solid rgba(255,255,255,.1);white-space:nowrap}
#err{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#c3c2b7;font-size:14px;padding:30px;text-align:center}
</style></head>
<body><div id="wrap">
<canvas id="gl"></canvas>
<div id="info" class="panel"></div>
<div id="controls" class="panel">
  <button id="bEarth" title="Look from Earth, along the telescope's line of sight">🔭 View from Earth</button>
  <button id="bSide" title="Look at the orbit from above">🛰️ Orbit view</button>
  <button id="bPlay">⏸ Pause</button>
  <button id="bScale" title="Toggle between real proportions and an exaggerated view">📏 True scale</button>
  <button id="bHz" title="Where liquid water could exist on a rocky planet">🌿 Habitable zone</button>
</div>
<div id="note"></div>
<div id="lc" class="panel"><div id="lchead"><span>📉 What the telescope sees <b>(brightness of the system)</b></span><span id="lcval"></span></div><canvas id="lcc"></canvas></div>
<div id="lblStar" class="lbl"></div><div id="lblComp" class="lbl"></div><div id="lblEarth" class="lbl">→ toward Earth 🔭</div>
<div id="err"></div>
</div>
<script>__THREE__</script>
<script>
(function(){
const P = __PARAMS__;
const $ = id => document.getElementById(id);
const wrap = $("wrap"), canvas = $("gl");
const C = P.companion;
const S = P.star;
const CLASS = {PC:["Planet candidate","#3987e5"], EB:["Eclipsing binary","#d95926"], FP:["Other false positive","#199e70"]};
const fmt = (v, d=2) => Number(v).toLocaleString(undefined,{maximumFractionDigits:d, minimumFractionDigits:0});
const est = f => P.assumed.includes(f) ? " (est.)" : "";

// ---------------------------------------------------------------- info panel
let html = `<h3>${P.name}</h3><div class="sub">${P.target_id}${P.mission ? " · " + P.mission : ""}</div>`;
html += `<div class="row"><span>Star</span><span>${S.spectral_type}-type · ${fmt(S.teff,0)} K${est("teff")}</span></div>`;
html += `<div class="row"><span>Star size</span><span>${fmt(S.radius_rsun)} × Sun${est("radius")}</span></div>`;
if (C) {
  html += `<div class="row"><span>Orbits every</span><span>${C.period_days < 2 ? fmt(C.period_days*24,1)+" h" : fmt(C.period_days,2)+" days"}</span></div>`;
  html += `<div class="row"><span>Orbit size</span><span>${fmt(C.a_au,3)} AU · ${fmt(C.a_over_r,1)} R★</span></div>`;
  html += `<div class="row"><span>Tilt seen from Earth</span><span>${fmt(C.inclination_deg,1)}°</span></div>`;
  if (C.kind === "star") html += `<div class="row"><span>Companion</span><span>~${fmt(C.radius_rsun)} × Sun star</span></div>`;
  else html += `<div class="row"><span>Planet size</span><span>${fmt(C.radius_earth,1)} × Earth</span></div>`;
  if (C.kind !== "star") html += `<div class="row"><span>Planet temperature</span><span>~${fmt(C.teq_k,0)} K</span></div>`;
  const cl = CLASS[C.predicted_class];
  if (cl) html += `<div class="chip"><span class="dot" style="background:${cl[1]}"></span>${cl[0]}${C.confidence!=null ? " · " + Math.round(C.confidence*100) + "%" : ""}</div>`;
} else {
  html += `<div class="sub" style="margin-top:8px">No repeating dip strong enough to model an orbit. Showing the star only.</div>`;
}
$("info").innerHTML = html;

// ---------------------------------------------------------------- renderer
let renderer;
try { renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true}); }
catch (e) { $("err").style.display="flex"; $("err").textContent="3D view needs WebGL, which this browser has turned off."; return; }
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 20000);

// starfield
{
  const n = 1800, pos = new Float32Array(n*3), col = new Float32Array(n*3);
  for (let i=0;i<n;i++){
    const u=Math.random()*2-1, t=Math.random()*Math.PI*2, r=3000;
    const s=Math.sqrt(1-u*u); pos.set([r*s*Math.cos(t), r*u, r*s*Math.sin(t)], i*3);
    const w=0.55+Math.random()*0.45; const tint=Math.random();
    col.set(tint<0.15?[w*0.8,w*0.85,w]:tint>0.9?[w,w*0.85,w*0.7]:[w,w,w], i*3);
  }
  const g=new THREE.BufferGeometry(); g.setAttribute("position",new THREE.BufferAttribute(pos,3)); g.setAttribute("color",new THREE.BufferAttribute(col,3));
  scene.add(new THREE.Points(g,new THREE.PointsMaterial({size:2.2,sizeAttenuation:false,vertexColors:true,transparent:true,opacity:0.85,depthWrite:false})));
}

// ---------------------------------------------------------------- stars (shader: limb darkening + granulation)
const starVS = `varying vec3 vN; varying vec3 vP; varying vec3 vV;
void main(){ vN=normalize(normalMatrix*normal); vP=position; vec4 mv=modelViewMatrix*vec4(position,1.0); vV=normalize(-mv.xyz); gl_Position=projectionMatrix*mv; }`;
const starFS = `uniform vec3 uColor; uniform float uTime; varying vec3 vN; varying vec3 vP; varying vec3 vV;
float h(vec3 p){ p=fract(p*0.3183099+0.1); p*=17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
float n(vec3 x){ vec3 i=floor(x); vec3 f=fract(x); f=f*f*(3.0-2.0*f);
 return mix(mix(mix(h(i),h(i+vec3(1,0,0)),f.x),mix(h(i+vec3(0,1,0)),h(i+vec3(1,1,0)),f.x),f.y),
            mix(mix(h(i+vec3(0,0,1)),h(i+vec3(1,0,1)),f.x),mix(h(i+vec3(0,1,1)),h(i+vec3(1,1,1)),f.x),f.y),f.z); }
void main(){ float mu=clamp(dot(vN,vV),0.0,1.0); float ld=1.0-0.55*(1.0-mu)-0.2*(1.0-mu)*(1.0-mu);
 float g=0.6*n(vP*7.0+uTime*0.12)+0.4*n(vP*19.0-uTime*0.08);
 vec3 c=uColor*(0.92+0.16*g)*ld*1.18; c=pow(min(c,vec3(1.0)),vec3(1.0/2.2)); gl_FragColor=vec4(c,1.0); }`;
function hexToVec(hex){ const c=new THREE.Color(hex); return new THREE.Vector3(c.r,c.g,c.b); }
function glowTexture(hex){
  const cv=document.createElement("canvas"); cv.width=cv.height=256; const g=cv.getContext("2d");
  const c=new THREE.Color(hex); const rgb=`${Math.round(c.r*255)},${Math.round(c.g*255)},${Math.round(c.b*255)}`;
  const gr=g.createRadialGradient(128,128,0,128,128,128);
  gr.addColorStop(0,`rgba(${rgb},0.9)`); gr.addColorStop(0.28,`rgba(${rgb},0.45)`); gr.addColorStop(0.55,`rgba(${rgb},0.12)`); gr.addColorStop(1,`rgba(${rgb},0)`);
  g.fillStyle=gr; g.fillRect(0,0,256,256); const t=new THREE.CanvasTexture(cv); t.colorSpace=THREE.SRGBColorSpace; return t;
}
const shaders = [];
function makeStar(hex, radius){
  const grp=new THREE.Group();
  const mat=new THREE.ShaderMaterial({vertexShader:starVS, fragmentShader:starFS, uniforms:{uColor:{value:hexToVec(hex)}, uTime:{value:0}}});
  shaders.push(mat);
  const m=new THREE.Mesh(new THREE.SphereGeometry(radius,96,64),mat); grp.add(m);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:glowTexture(hex),blending:THREE.AdditiveBlending,depthWrite:false,transparent:true}));
  sp.scale.setScalar(radius*4.2); grp.add(sp); grp.userData.body=m; return grp;
}
const star = makeStar(S.color, 1.0); scene.add(star);
const light = new THREE.PointLight(0xffffff, 3.2, 0, 0); light.color.set(S.color).lerp(new THREE.Color(0xffffff),0.55); scene.add(light);
scene.add(new THREE.AmbientLight(0x404a66, 0.18));

// ---------------------------------------------------------------- companion
function planetTexture(look){
  const cv=document.createElement("canvas"); cv.width=512; cv.height=256; const g=cv.getContext("2d");
  const pal = look==="gas" ? ["#c9a77c","#a87e55","#e3cba4","#8f6a48","#d8b88f"] : look==="ice" ? ["#7fb8d6","#5e9cc2","#a7d3e6","#4f86ad"] : ["#8a7f74","#6f655c","#a3988b","#5c544d"];
  g.fillStyle=pal[0]; g.fillRect(0,0,512,256);
  if (look==="rocky"){ for(let i=0;i<220;i++){ g.fillStyle=pal[i%pal.length]; g.globalAlpha=0.35; g.beginPath(); g.arc(Math.random()*512,Math.random()*256,4+Math.random()*26,0,7); g.fill(); } }
  else { let y=0; while(y<256){ const hh=6+Math.random()*(look==="gas"?22:34); g.fillStyle=pal[Math.floor(Math.random()*pal.length)]; g.globalAlpha=0.55; g.fillRect(0,y,512,hh); y+=hh; } }
  g.globalAlpha=1; const t=new THREE.CanvasTexture(cv); t.colorSpace=THREE.SRGBColorSpace; return t;
}
let comp=null, compBody=null, sbRatio=0;
if (C) {
  if (C.kind === "star") {
    comp = makeStar(C.color, 1.0); compBody = comp.userData.body;
    sbRatio = Math.pow(C.teff / S.teff, 4);
  } else {
    const hot = Math.min(Math.max((C.teq_k - 1200)/2400, 0), 0.22);
    const mat = new THREE.MeshStandardMaterial({map:planetTexture(C.look), roughness:0.85, metalness:0.0,
      emissive:new THREE.Color(0xff4a1a), emissiveIntensity:hot, transparent:C.kind==="unknown", opacity:C.kind==="unknown"?0.8:1});
    if (C.kind === "unknown") mat.color.set(0x9a9a9a);
    compBody = new THREE.Mesh(new THREE.SphereGeometry(1,64,48), mat);
    compBody.rotation.z = 0.3;
    comp = new THREE.Group(); comp.add(compBody);
  }
  scene.add(comp);
}

// ---------------------------------------------------------------- geometry (enhanced vs true scale)
let trueScale=false, showHz=false, playing=true, hzR=0;
let aVis=3, kVis=0.1, incl=Math.PI/2;
const orbitGroup = new THREE.Group(); scene.add(orbitGroup);
const mapA = x => trueScale ? x : Math.min(x, 2.6 + 1.9*Math.log(Math.max(x,1.01)));
function rebuild(){
  orbitGroup.clear();
  if (!C) return;
  const aR = C.a_over_r;
  kVis = trueScale ? C.k : (C.kind==="star" ? Math.min(C.k,0.9) : Math.min(Math.max(C.k*2.5,0.13),0.45));
  aVis = Math.max(mapA(aR), 1 + kVis + 0.35);
  const cosi = Math.min(C.impact_b / aVis, 0.999);
  incl = Math.acos(cosi);
  if (C.kind==="star") comp.scale.setScalar(kVis); else compBody.scale.setScalar(kVis);
  const pts=[]; for(let i=0;i<=256;i++){ pts.push(pos(i/256*Math.PI*2)); }
  orbitGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color:0x3987e5,transparent:true,opacity:0.45})));
  // line of sight toward Earth (+z)
  const los=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,1.05),new THREE.Vector3(0,0,aVis*1.55)]),
    new THREE.LineDashedMaterial({color:0x8a897f,dashSize:0.12*aVis/3,gapSize:0.09*aVis/3,transparent:true,opacity:0.6}));
  los.computeLineDistances(); orbitGroup.add(los);
  if (showHz && P.habitable_zone_au){
    const toR = au => mapA(au*215.032/S.radius_rsun);
    const r0=toR(P.habitable_zone_au[0]), r1=toR(P.habitable_zone_au[1]);
    const ring=new THREE.Mesh(new THREE.RingGeometry(r0,r1,160), new THREE.MeshBasicMaterial({color:0x2fbf71,transparent:true,opacity:0.16,side:THREE.DoubleSide,depthWrite:false}));
    ring.rotation.x = incl; orbitGroup.add(ring);
    hzR = r1;
  }
  $("note").innerHTML = trueScale ? "True scale: real sizes and distances" : "Sizes and distances exaggerated so you can see them.<br>Press 📏 for true scale.";
  $("bScale").classList.toggle("on", trueScale); $("bHz").classList.toggle("on", showHz);
}
// orbit: transit (phi=0) puts the companion between the star and Earth (+z), offset b on the sky
function pos(phi){ return new THREE.Vector3(aVis*Math.sin(phi), aVis*Math.cos(phi)*Math.cos(incl), aVis*Math.cos(phi)*Math.sin(incl)); }

// ---------------------------------------------------------------- camera controls (drag / wheel / pinch)
const cam = {theta:0.75, phi:1.12, r:8, tTheta:0.75, tPhi:1.12, tR:8};
function fitRadius(){ const ext = C ? Math.max(aVis, showHz ? hzR : 0) + kVis : 1.6; return Math.max(ext*2.45, C ? 4.5 : 7); }
function preset(name){
  if (name==="earth"){ cam.tTheta=0; cam.tPhi=Math.PI/2; cam.tR = C ? Math.max(aVis*1.9, 5) : 5; }
  else if (name==="side"){ cam.tTheta=0.0; cam.tPhi=0.18; cam.tR=fitRadius()*1.05; }
  else { cam.tTheta=0.75; cam.tPhi=1.12; cam.tR=fitRadius(); }
  $("bEarth").classList.toggle("on", name==="earth"); $("bSide").classList.toggle("on", name==="side");
}
let drag=null, pinch=null;
canvas.addEventListener("pointerdown", e=>{ drag={x:e.clientX,y:e.clientY}; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e=>{ if(!drag||pinch) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag={x:e.clientX,y:e.clientY};
  cam.tTheta-=dx*0.006; cam.tPhi=Math.min(Math.max(cam.tPhi-dy*0.006,0.05),Math.PI-0.05); $("bEarth").classList.remove("on"); $("bSide").classList.remove("on"); });
canvas.addEventListener("pointerup", ()=>{ drag=null; });
canvas.addEventListener("wheel", e=>{ e.preventDefault(); cam.tR=Math.min(Math.max(cam.tR*(1+e.deltaY*0.0012),1.6),20000); }, {passive:false});
canvas.addEventListener("touchstart", e=>{ if(e.touches.length===2){ pinch=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);} }, {passive:true});
canvas.addEventListener("touchmove", e=>{ if(e.touches.length===2&&pinch){ const d=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY); cam.tR=Math.min(Math.max(cam.tR*pinch/d,1.6),20000); pinch=d; } }, {passive:true});
canvas.addEventListener("touchend", ()=>{ pinch=null; });

$("bEarth").onclick=()=>preset("earth");
$("bSide").onclick=()=>preset("side");
$("bPlay").onclick=()=>{ playing=!playing; $("bPlay").textContent = playing ? "⏸ Pause" : "▶ Play"; };
$("bScale").onclick=()=>{ trueScale=!trueScale; rebuild(); cam.tR=fitRadius(); };
$("bHz").onclick=()=>{ showHz=!showHz; rebuild(); cam.tR=Math.max(cam.tR, fitRadius()); };
if (!C){ $("controls").style.display="none"; $("lc").style.display="none"; $("lblEarth").style.display="none"; }
if (!P.habitable_zone_au) $("bHz").style.display="none";

// ---------------------------------------------------------------- photometry: overlap of two disks
function overlap(d, R, r){ // area of overlap of circles radius R and r, centers d apart
  if (d >= R + r) return 0;
  if (d <= Math.abs(R - r)) return Math.PI*Math.min(R,r)**2;
  const a = r*r*Math.acos((d*d + r*r - R*R)/(2*d*r)), b = R*R*Math.acos((d*d + R*R - r*r)/(2*d*R));
  const c = 0.5*Math.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)); return a + b - c;
}
const lcBuf=[]; const LC_N=420; let lcMin=1;
function flux(p){
  const d=Math.hypot(p.x,p.y); const A=overlap(d,1,kVis);
  const total=Math.PI*(1 + kVis*kVis*sbRatio);
  return p.z>0 ? (total - A)/total : (total - A*sbRatio)/total;
}
const lcc=$("lcc"), lctx=lcc.getContext("2d");
function drawLC(){
  const w=lcc.clientWidth, h=lcc.clientHeight, dpr=Math.min(window.devicePixelRatio||1,2);
  if (lcc.width!==w*dpr){ lcc.width=w*dpr; lcc.height=h*dpr; }
  lctx.setTransform(dpr,0,0,dpr,0,0); lctx.clearRect(0,0,w,h);
  const lo=Math.min(lcMin, 1-1e-6), pad=6, y=v=>pad+(1-v)/(1-lo)*(h-2*pad);
  lctx.strokeStyle="rgba(255,255,255,0.08)"; lctx.beginPath(); lctx.moveTo(0,y(1)); lctx.lineTo(w,y(1)); lctx.stroke();
  lctx.strokeStyle="#3987e5"; lctx.lineWidth=2; lctx.beginPath();
  lcBuf.forEach((v,i)=>{ const x=w*i/(LC_N-1); i?lctx.lineTo(x,y(v)):lctx.moveTo(x,y(v)); }); lctx.stroke();
  if (lcBuf.length){ const v=lcBuf[lcBuf.length-1]; lctx.fillStyle="#d95926"; lctx.beginPath(); lctx.arc(w*(lcBuf.length-1)/(LC_N-1), y(v), 4, 0, 7); lctx.fill(); }
}

// ---------------------------------------------------------------- labels
function place(el, v){
  const p=v.clone().project(camera); const w=wrap.clientWidth, h=wrap.clientHeight;
  if (p.z>1){ el.style.display="none"; return; } el.style.display="block";
  el.style.left=((p.x+1)/2*w)+"px"; el.style.top=((1-p.y)/2*h)+"px";
}
$("lblStar").textContent = P.name;
if (C) $("lblComp").textContent = C.kind==="star" ? "companion star" : (C.kind==="unknown" ? "? " + C.label : C.label);
else $("lblComp").style.display="none";

// ---------------------------------------------------------------- loop
function resize(){ const w=wrap.clientWidth, h=wrap.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h;
  if (C) camera.setViewOffset(w, h, 0, 40, w, h); else camera.clearViewOffset(); camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(wrap); resize();
rebuild(); preset("home"); cam.theta=cam.tTheta; cam.phi=cam.tPhi; cam.r=cam.tR*1.6;
const ORBIT_SECONDS = 11; let phase = -0.9, last = performance.now(), t0 = last;
const depthTxt = C ? (C.depth_ppm >= 10000 ? (C.depth_ppm/1e4).toFixed(1)+"%" : fmt(C.depth_ppm,0)+" ppm") : "";
function frame(now){
  const dt=Math.min((now-last)/1000,0.1); last=now;
  shaders.forEach(s=>s.uniforms.uTime.value=(now-t0)/1000);
  star.userData.body.rotation.y += dt*0.05;
  if (C){
    if (playing) phase += dt*2*Math.PI/ORBIT_SECONDS;
    const p=pos(phase); comp.position.copy(p); compBody.rotation.y += dt*0.6;
    if (playing){ const f=flux(p); lcBuf.push(f); if(lcBuf.length>LC_N) lcBuf.shift(); lcMin=Math.min(...lcBuf, 1-1e-6); }
    const cur = lcBuf.length ? lcBuf[lcBuf.length-1] : 1;
    $("lcval").innerHTML = (cur < 1-1e-7) ? `<b style="color:#d95926">dimming now</b> · real dip ${depthTxt}` : `real dip ${depthTxt}`;
    drawLC();
    place($("lblComp"), p.clone().add(new THREE.Vector3(0, kVis*1.1, 0)));
    if (Math.abs(cam.theta) < 0.25 && Math.abs(cam.phi - Math.PI/2) < 0.25) $("lblEarth").style.display="none";
    else place($("lblEarth"), new THREE.Vector3(0,0,aVis*1.55));
  }
  cam.theta+= (cam.tTheta-cam.theta)*0.08; cam.phi+=(cam.tPhi-cam.phi)*0.08; cam.r+=(cam.tR-cam.r)*0.08;
  camera.position.set(cam.r*Math.sin(cam.phi)*Math.sin(cam.theta), cam.r*Math.cos(cam.phi), cam.r*Math.sin(cam.phi)*Math.cos(cam.theta));
  camera.lookAt(0,0,0);
  place($("lblStar"), new THREE.Vector3(0, 1.15, 0));
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
})();
</script></body></html>
"""
