// bridge.js — AMD Radeon Cloud 控制桥（Windows 常驻，纯 node 无依赖）
// 通道: WSL/CLI → HTTP(127.0.0.1:19826) → pending → 扩展轮询 → 站点API → reply
// 启动:  node bridge.js   (或 install.ps1)
const http = require("http");
const PORT = 19826;

let nextId = 1;
const pending = new Map();   // id -> {resolve, payload}
const replies = new Map();   // id -> {status, body}   (扩展已回、待客户端取走)

function json(res, code, obj) {
  const b = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(b) });
  res.end(b);
}

let stPolls = 0, stReplies = 0;
const server = http.createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    try {
      // /poll    扩展短轮询: 取一个待办请求  {id, method, path, body}
      // /reply   扩展回填结果              {id, status, text, json}
      // /req     客户端同步调用            {method, path, body?, timeout?}
      const u = new URL(req.url, "http://x");
      if (u.pathname === "/poll") {
        stPolls++;
        if (pending.size > 0) {
          const [id, p] = pending.entries().next().value;
          pending.delete(id);
          return json(res, 200, { id, ...p });
        }
        return json(res, 200, {});
      }
      if (u.pathname === "/reply") {
        stReplies++;
        const r = JSON.parse(body || "{}");
        replies.set(r.id, r);
        return json(res, 200, { ok: true });
      }
      if (u.pathname === "/req") {
        const q = JSON.parse(body || "{}");
        const id = nextId++;
        const timeout = q.timeout || 90;
        const t = setTimeout(() => {
          if (replies.has(id)) return;
          replies.set(id, { id, status: -1, text: "bridge timeout" });
          const cb = pending.get(id);
          if (cb) { pending.delete(id); cb.resolve(); }
        }, timeout * 1000);
        pending.set(id, { method: q.method, path: q.path, body: q.body });
        const wait = setInterval(() => {
          if (replies.has(id)) {
            clearInterval(wait); clearTimeout(t);
            const r = replies.get(id); replies.delete(id);
            return json(res, 200, r);
          }
        }, 150);
        return;
      }
      if (u.pathname === "/stat") return json(res, 200, { polls: stPolls, replies: stReplies, pending: pending.size });
      json(res, 404, { error: "not found" });
    } catch (e) {
      json(res, 500, { error: String(e) });
    }
  });
});

server.listen(PORT, "127.0.0.1", () => console.log("[bridge] listening 127.0.0.1:" + PORT));
