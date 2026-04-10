document.addEventListener('DOMContentLoaded', async () => {

  const data = await DataLoader.load();
  if (!data) {
    document.getElementById('app-loading').textContent = 'Failed to load data.';
    return;
  }

  document.getElementById('app-loading').style.display = 'none';
  document.getElementById('app-content').style.display = 'block';

  /* ─── Render everything ─── */
  Renderers.renderSignalHeader(data);
  Renderers.renderSignalTable(data);
  Renderers.renderDetailTitle(data);
  Renderers.renderMetrics(data);
  Renderers.renderPipeline(data);
  Renderers.renderSpecs(data);
  Renderers.renderBacktest(data);
  Renderers.renderTrading(data);

  if (data.equity_curve) {
    Charts.renderEquityCurve('chart-equity', data.equity_curve);
  }

  /* ─── Tab switching ─── */
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const t = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.panel').forEach(p => {
        const show = p.id === 'panel-' + t;
        p.classList.toggle('active', show);
        if (show) {
          p.querySelectorAll('.reveal:not(.vis)').forEach(r => r.classList.add('vis'));
          if (t === 'perf') {
            const c = Chart.getChart('chart-equity');
            if (c) c.resize();
          }
        }
      });
    });
  });

  /* ─── Scroll reveal ─── */
  if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) { e.target.classList.add('vis'); obs.unobserve(e.target); }
      });
    }, { threshold: 0.1 });
    document.querySelectorAll('.reveal').forEach(el => obs.observe(el));
  } else {
    document.querySelectorAll('.reveal').forEach(el => el.classList.add('vis'));
  }
});
