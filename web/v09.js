(() => {
  const main = document.querySelector('main');
  if (!main) return;

  const panel = document.createElement('section');
  panel.className = 'panel wide';
  panel.innerHTML = `
    <h2>V0.9 Fast Path / AI Ensemble</h2>
    <div class="row">
      <div class="stat"><div class="label">Fast path</div><div class="value" id="v09FastStatus">—</div></div>
      <div class="stat"><div class="label">Target rate</div><div class="value" id="v09TargetRate">—</div></div>
      <div class="stat"><div class="label">Cycle compute</div><div class="value" id="v09CycleMs">—</div></div>
      <div class="stat"><div class="label">Valid cycles</div><div class="value" id="v09Cycles">0</div></div>
      <div class="stat"><div class="label">Fast trades</div><div class="value" id="v09Trades">0</div></div>
      <div class="stat"><div class="label">AI sidecar</div><div class="value" id="v09Sidecar">OFF</div></div>
      <div class="stat"><div class="label">Sidecar model</div><div class="value purple" id="v09SidecarModel">amazon/chronos-2</div></div>
    </div>
    <div class="small" id="v09FastNote">Waiting for fast-path telemetry…</div>
  `;

  // Put the latency panel near the top without changing the legacy V0.8 file.
  const referencePanel = main.children[2] || null;
  main.insertBefore(panel, referencePanel);

  const $v = (id) => document.getElementById(id);
  const fmt = (x, n = 1) => Number(x || 0).toFixed(n);

  async function refreshFastPath() {
    try {
      const r = await fetch('/api/fast-path', { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const x = await r.json();
      const side = x.sidecar || {};
      const running = !!x.running;
      const keepingUp = Number(x.ewma_cycle_ms || 0) <= Number(x.target_interval_ms || 0);

      $v('v09FastStatus').textContent = running ? (keepingUp ? 'LIVE' : 'LAGGING') : (x.enabled ? 'STARTING' : 'OFF');
      $v('v09FastStatus').className = 'value ' + (running && keepingUp ? 'good' : running ? 'yellow' : '');
      $v('v09TargetRate').textContent = `${fmt(x.target_hz, 1)} Hz / ${x.target_interval_ms || 0}ms`;
      $v('v09CycleMs').textContent = `${fmt(x.ewma_cycle_ms, 2)}ms avg`;
      $v('v09Cycles').textContent = x.valid_cycles ?? 0;
      $v('v09Trades').textContent = x.trades ?? 0;
      $v('v09Sidecar').textContent = side.enabled ? (side.error ? 'ERROR' : side.latest ? 'LIVE' : 'READY') : 'OFF';
      $v('v09Sidecar').className = 'value ' + (side.enabled && !side.error ? 'good' : side.error ? 'bad' : '');
      $v('v09SidecarModel').textContent = side.configured_model || 'amazon/chronos-2';

      const notes = [];
      if (x.last_skip_reason) notes.push(`skip: ${x.last_skip_reason}`);
      if (x.last_error) notes.push(`fast error: ${x.last_error}`);
      if (side.error) notes.push(`sidecar: ${side.error}`);
      if (!side.enabled) notes.push('Chronos sidecar is optional and currently disabled');
      if (running && keepingUp) notes.push('in-memory decision path is meeting its configured latency budget');
      $v('v09FastNote').textContent = notes.join(' • ') || 'Fast path healthy.';

      const badge = document.getElementById('fastBadge');
      if (badge) {
        badge.textContent = running ? `FAST ${fmt(x.target_hz, 0)}HZ` : 'FAST OFF';
        badge.className = 'pill ' + (running && keepingUp ? 'good' : running ? 'yellow' : '');
      }
    } catch (err) {
      $v('v09FastStatus').textContent = 'ERROR';
      $v('v09FastStatus').className = 'value bad';
      $v('v09FastNote').textContent = String(err);
    }
  }

  refreshFastPath();
  setInterval(refreshFastPath, 1000);
})();
