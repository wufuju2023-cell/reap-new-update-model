// service_worker.js — AMD 控制桥扩展本体（MV3, 无 UI）
// 职责: 每 800ms 轮询 http://127.0.0.1:19826/poll
//       取到请求 → 带 cookie fetch 站点 API（host_permissions 免 CORS）→ 回填 /reply
const BRIDGE = "http://127.0.0.1:19826";
const PREFIX = "https://developer.amd.com.cn";

async function replied(id) {
  try {
    return (await (await fetch(BRIDGE + "/reply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(id),
    })).json()).ok;
  } catch (e) {
    return false;
  }
}

async function doPoll() {
  let job = null;
  try {
    job = await (await fetch(BRIDGE + "/poll", { cache: "no-store" })).json();
  } catch (e) { /* bridge 未启动 → 静默 */ }

  if (!job || !job.id || !job.path) return;

  const url = PREFIX + job.path;
  const init = {
    method: job.method || "GET",
    credentials: "include",            // 同登录态 cookie
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
  };
  if (job.body !== undefined && job.body !== null) init.body = typeof job.body === "string" ? job.body : JSON.stringify(job.body);

  let result;
  try {
    const r = await fetch(url, init);
    const text = await r.text();
    result = { id: job.id, status: r.status, text: text.slice(0, 2_000_000) }; // 限 2MB，防止超大响应拖垮轮询
  } catch (e) {
    result = { id: job.id, status: -2, text: "ext-fetch-error: " + e.message };
  }
  replied(result);
  if (job.id) { /* continue */ }
}

// service worker 保活: 定时 + alarm
chrome.alarms?.create("poll", { periodInMinutes: 0.1 });
chrome.alarms?.onAlarm.addListener((a) => { if (a.name === "poll") doPoll(); });
setInterval(doPoll, 800);
chrome.runtime.onStartup.addListener(() => doPoll());
