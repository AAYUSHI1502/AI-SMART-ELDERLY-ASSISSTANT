"""
app.py  –  AI Smart Elderly Assistant (Python + ML backend)
============================================================
Run:  python3 app.py
Open: http://localhost:5000
"""

import os, pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# ── Load ML artefacts ─────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "model", "intent_model.pkl"),     "rb") as f:
    clf = pickle.load(f)
with open(os.path.join(BASE, "model", "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)
with open(os.path.join(BASE, "model", "label_encoder.pkl"),    "rb") as f:
    le = pickle.load(f)

# ── Contacts (edit phone numbers here) ───────────────────────────────────────
CONTACTS = {
    "Mom":    "9711762381",
    "Dad":    "9311080730",
    "Doctor": "9311080725",
}

INTENT_TO_CONTACT = {
    "contact_mom":    "Mom",
    "contact_dad":    "Dad",
    "contact_doctor": "Doctor",
}

app = Flask(__name__)

# ── NLP prediction endpoint ───────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"intent": "unknown", "contact": None,
                        "confidence": 0.0, "label": "❓ Empty input"})

    X   = tfidf.transform([text])
    idx = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    conf  = float(proba.max())
    intent = le.inverse_transform([idx])[0]

    # Low-confidence → unknown
    if conf < 0.40:
        intent = "unknown"

    contact = INTENT_TO_CONTACT.get(intent)

    LABELS = {
        "emergency":      "🚨 Emergency",
        "contact_mom":    "💬 Message → Mom",
        "contact_dad":    "💬 Message → Dad",
        "contact_doctor": "💬 Message → Doctor",
        "unknown":        "❓ Not understood",
    }

    return jsonify({
        "intent":     intent,
        "contact":    contact,
        "confidence": round(conf, 3),
        "label":      LABELS.get(intent, "❓"),
        "number":     CONTACTS.get(contact) if contact else None,
    })

# ── Serve the single-page frontend ───────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

# ═════════════════════════════════════════════════════════════════════════════
#  FRONTEND  (embedded so the project is a single python file + model folder)
# ═════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Elder Assistant</title>
<style>
:root{
  --emergency-red:#d32f2f;
  --whatsapp-green:#25D366;
  --bg-color:#f4f7f6;
  --mic-active:#ff4444;
  --mic-idle:#333;
}
*{box-sizing:border-box;}
body{
  font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;
  background:var(--bg-color);
  display:flex;justify-content:center;align-items:center;
  min-height:100vh;margin:0;
}
.phone-container{
  width:350px;background:white;border-radius:30px;
  box-shadow:0 20px 40px rgba(0,0,0,.2);
  overflow:hidden;display:flex;flex-direction:column;
  border:8px solid #333;
}
.header{
  background:#333;color:white;padding:15px 20px;
  text-align:center;font-size:1.2rem;font-weight:bold;
}
.content{
  flex:1;padding:20px;display:flex;flex-direction:column;
  gap:15px;overflow-y:auto;
}
.status-bar{
  font-size:.9rem;color:#666;text-align:center;
  min-height:20px;transition:all .3s;
}
.btn{
  padding:20px;border:none;border-radius:15px;
  font-size:1.1rem;font-weight:bold;cursor:pointer;
  transition:transform .1s,box-shadow .2s;
  display:flex;align-items:center;justify-content:center;gap:10px;
}
.btn:active{transform:scale(.97);}
.btn-emergency{background:var(--emergency-red);color:white;height:100px;font-size:1.4rem;}
.btn-normal{background:var(--whatsapp-green);color:white;}
.btn-back{background:#eee;color:#333;margin-top:auto;}

/* AI section */
.ai-section{border-top:2px solid #eee;padding-top:15px;margin-top:10px;
  display:flex;flex-direction:column;gap:8px;}
.input-row{display:flex;gap:8px;align-items:center;}
input{
  flex:1;padding:14px;border:2px solid #ddd;border-radius:10px;
  font-size:1rem;outline:none;transition:border-color .2s;
}
input:focus{border-color:#25D366;}
.mic-btn{
  width:52px;height:52px;border-radius:50%;border:none;
  background:var(--mic-idle);color:white;font-size:1.3rem;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  flex-shrink:0;transition:background .2s,transform .1s,box-shadow .2s;
}
.mic-btn:active{transform:scale(.92);}
.mic-btn.listening{
  background:var(--mic-active);
  animation:pulse-ring 1.2s ease-out infinite;
}
@keyframes pulse-ring{
  0%{box-shadow:0 0 0 0 rgba(255,68,68,.6);}
  70%{box-shadow:0 0 0 14px rgba(255,68,68,0);}
  100%{box-shadow:0 0 0 0 rgba(255,68,68,0);}
}
.voice-bubble{
  background:#f0f8f0;border:1.5px solid #25D366;border-radius:10px;
  padding:10px 14px;font-size:.88rem;color:#333;min-height:36px;
  display:none;animation:fadeIn .3s;
}
.voice-bubble.visible{display:block;}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px);}to{opacity:1;transform:translateY(0);}}

/* Confidence + badge */
.nlp-result{display:none;flex-direction:column;gap:4px;animation:fadeIn .3s;}
.nlp-result.visible{display:flex;}
.nlp-badge{
  display:flex;align-items:center;gap:6px;padding:6px 12px;
  border-radius:20px;font-size:.82rem;font-weight:600;
}
.intent-emergency{background:#fdecea;color:#c62828;border:1px solid #ef9a9a;}
.intent-contact  {background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7;}
.intent-unknown  {background:#f5f5f5;color:#555;   border:1px solid #ddd;}
.conf-bar-wrap{height:6px;background:#eee;border-radius:3px;overflow:hidden;}
.conf-bar{height:100%;border-radius:3px;transition:width .5s ease;}
.conf-label{font-size:.74rem;color:#999;text-align:right;}

.run-btn{
  background:#333;color:white;width:100%;padding:16px;border:none;
  border-radius:10px;font-size:1rem;font-weight:bold;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:8px;
  transition:background .2s;
}
.run-btn:hover{background:#444;}
.run-btn.loading{opacity:.7;pointer-events:none;}
.voice-hint{font-size:.78rem;color:#aaa;text-align:center;}

/* model tag */
.model-tag{
  font-size:.7rem;color:#bbb;text-align:center;
  padding:4px 0;letter-spacing:.04em;
}
.hidden{display:none!important;}
</style>
</head>
<body>
<div class="phone-container">
  <div class="header">🤖 AI Smart Helper</div>

  <!-- MAIN MENU -->
  <div class="content" id="main-menu">
    <div class="status-bar" id="status">Ready to assist</div>

    <button class="btn btn-emergency" onclick="showContacts('emergency')">🚨 EMERGENCY</button>
    <button class="btn btn-normal"    onclick="showContacts('normal')">💬 WhatsApp</button>

    <div class="ai-section">
      <div class="input-row">
        <input type="text" id="ai-input" placeholder="e.g. 'Call Mom' or 'Help'">
        <button class="mic-btn" id="mic-btn" onclick="toggleVoice()" title="Speak a command">🎤</button>
      </div>
      <div class="voice-bubble" id="voice-bubble">🎙️ Listening…</div>

      <!-- ML result -->
      <div class="nlp-result" id="nlp-result">
        <div class="nlp-badge" id="nlp-badge"></div>
        <div class="conf-bar-wrap">
          <div class="conf-bar" id="conf-bar"></div>
        </div>
        <div class="conf-label" id="conf-label"></div>
      </div>

      <button class="run-btn" id="run-btn" onclick="processAI()">🤖 Run AI Command</button>
      <div class="voice-hint">Tap 🎤 and say "Emergency send to Mom" or "Call Doctor"</div>
      <div class="model-tag">Powered by TF-IDF + Logistic Regression · scikit-learn</div>
    </div>
  </div>

  <!-- CONTACT MENU -->
  <div class="content hidden" id="contact-menu">
    <div class="status-bar" id="contact-status">Select a Contact</div>
    <div id="contact-list" style="display:flex;flex-direction:column;gap:10px;"></div>
    <button class="btn btn-back" onclick="showMain()">⬅ Back</button>
  </div>
</div>

<script>
const CONTACTS = { Mom:"9711762381", Dad:"9311080730", Doctor:"9311080725" };
let currentMode = "normal";

// ── Speech synthesis ─────────────────────────────────────────────────────────
function speak(text){
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(text);
  u.lang="en-IN"; u.rate=0.9;
  window.speechSynthesis.speak(u);
}

// ── ML Prediction (calls Python Flask /predict) ───────────────────────────────
async function mlPredict(text){
  const resp = await fetch("/predict",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({text})
  });
  return resp.json();
}

// ── Show ML badge + confidence bar ───────────────────────────────────────────
function showMLResult(result){
  const wrap  = document.getElementById('nlp-result');
  const badge = document.getElementById('nlp-badge');
  const bar   = document.getElementById('conf-bar');
  const lbl   = document.getElementById('conf-label');

  const isEmerg = result.intent === "emergency";
  const isUnk   = result.intent === "unknown";
  badge.className = "nlp-badge " + (isUnk ? "intent-unknown" : isEmerg ? "intent-emergency" : "intent-contact");
  badge.textContent = result.label;

  const pct = Math.round(result.confidence * 100);
  bar.style.width = pct + "%";
  bar.style.background = isUnk ? "#bbb" : isEmerg ? "#e53935" : "#25D366";
  lbl.textContent = `ML confidence: ${pct}%`;

  wrap.classList.add('visible');
}

function hideMLResult(){
  document.getElementById('nlp-result').classList.remove('visible');
}

// ── Process AI command (text or voice → ML → action) ─────────────────────────
async function processAI(){
  const raw = document.getElementById('ai-input').value.trim();
  const btn = document.getElementById('run-btn');
  const status = document.getElementById('status');

  if(!raw){ speak("Please type or speak a command."); return; }

  btn.classList.add('loading');
  btn.textContent = "⏳ Thinking…";

  try {
    const result = await mlPredict(raw);
    showMLResult(result);

    if(result.intent === "unknown"){
      status.textContent = "Try: 'Emergency', 'Call Mom', 'Call Doctor'";
      speak("Command not understood. Try saying: help, call mom, or call doctor.");
    }
    else if(result.intent === "emergency"){
      status.textContent = "🚨 Emergency mode";
      if(result.contact && result.number){
        speak(`Sending emergency alert to ${result.contact}`);
        sendEmergencyWhatsApp(result.number, result.contact);
      } else {
        showContacts("emergency");
      }
    }
    else {
      // contact_mom / contact_dad / contact_doctor
      status.textContent = `Contacting ${result.contact}`;
      speak(`Opening WhatsApp for ${result.contact}`);
      openWhatsApp(result.number, "Hi");
    }
  } catch(e){
    status.textContent = "Server error – is Flask running?";
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = "🤖 Run AI Command";
  }
}

// ── Voice recognition ────────────────────────────────────────────────────────
let recognition=null, isListening=false, autoExec=false;

function initSpeech(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){ document.getElementById('mic-btn').style.opacity='.4'; return false; }
  recognition = new SR();
  recognition.lang="en-IN"; recognition.continuous=false; recognition.interimResults=true;

  const bubble = document.getElementById('voice-bubble');
  const input  = document.getElementById('ai-input');

  recognition.onresult = async (e) => {
    let interim="", final="";
    for(let i=e.resultIndex;i<e.results.length;i++){
      if(e.results[i].isFinal) final += e.results[i][0].transcript;
      else interim += e.results[i][0].transcript;
    }
    const shown = (final||interim).trim();
    bubble.textContent = "🎙️ " + (shown||"Listening…");
    bubble.classList.add('visible');
    if(final){
      input.value = final.trim();
      // Preview ML result while listening ends
      const r = await mlPredict(final.trim());
      showMLResult(r);
    }
  };

  recognition.onend = ()=>{
    setMicState(false);
    document.getElementById('voice-bubble').classList.remove('visible');
    if(autoExec && document.getElementById('ai-input').value.trim()){
      autoExec=false; processAI();
    }
  };
  recognition.onerror = (e)=>{ setMicState(false); };
  return true;
}

function setMicState(on){
  isListening=on;
  const btn=document.getElementById('mic-btn');
  btn.classList.toggle('listening',on);
  btn.textContent = on ? "⏹" : "🎤";
}

function toggleVoice(){
  if(!recognition && !initSpeech()) return;
  if(isListening){ recognition.stop(); autoExec=false; }
  else {
    hideMLResult();
    document.getElementById('ai-input').value="";
    const b=document.getElementById('voice-bubble');
    b.textContent="🎙️ Listening… speak now"; b.classList.add('visible');
    autoExec=true; setMicState(true);
    speak("Listening. Speak your command.");
    recognition.start();
  }
}

// ── Navigation ───────────────────────────────────────────────────────────────
function showContacts(mode){
  currentMode=mode;
  document.getElementById('main-menu').classList.add('hidden');
  document.getElementById('contact-menu').classList.remove('hidden');

  const list=document.getElementById('contact-list');
  list.innerHTML="";
  Object.keys(CONTACTS).forEach(name=>{
    const btn=document.createElement('button');
    btn.className="btn btn-normal";
    btn.innerHTML=`👤 ${name}`;
    btn.onclick=()=>handleAction(name);
    list.appendChild(btn);
  });

  const cs=document.getElementById('contact-status');
  if(mode==='emergency'){
    cs.textContent="⚠️ EMERGENCY MODE ACTIVE";
    cs.style.color="red"; cs.style.fontWeight="bold";
  } else {
    cs.textContent="Select Contact";
    cs.style.color="#666"; cs.style.fontWeight="normal";
  }
}

function showMain(){
  document.getElementById('contact-menu').classList.add('hidden');
  document.getElementById('main-menu').classList.remove('hidden');
}

// ── WhatsApp helpers ─────────────────────────────────────────────────────────
function sendEmergencyWhatsApp(number, name){
  const base="🚨 EMERGENCY! I need help. My location is being sent.";
  if(navigator.geolocation){
    navigator.geolocation.getCurrentPosition(
      pos => openWhatsApp(number, base+` https://maps.google.com/?q=${pos.coords.latitude},${pos.coords.longitude}`),
      ()  => openWhatsApp(number, base)
    );
  } else { openWhatsApp(number, base); }
}

function handleAction(name){
  const number=CONTACTS[name];
  if(currentMode==="emergency"){
    speak(`Sending emergency alert to ${name}`);
    sendEmergencyWhatsApp(number, name);
  } else {
    speak("Opening WhatsApp"); openWhatsApp(number,"Hi");
  }
}

function openWhatsApp(num,msg){
  window.open(`https://wa.me/${num}?text=${encodeURIComponent(msg)}`,'_blank');
}

window.addEventListener('load', initSpeech);
</script>
</body>
</html>
"""

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤖  AI Smart Elderly Assistant")
    print("   ML Model : TF-IDF + Logistic Regression (scikit-learn)")
    print("   Intents  :", list(le.classes_))
    print("   Open     : http://localhost:5000\n")
    app.run(debug=True, port=5000)
