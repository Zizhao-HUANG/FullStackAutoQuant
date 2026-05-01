const Renderers = (() => {

  function pct(v, d = 2) { return (v * 100).toFixed(d) + '%'; }

  /* ─── Detail hero: giant date (left) + Day N (right) ─── */
  function renderDetailHero(data) {
    const el = document.getElementById('detail-hero');
    if (!el) return;

    const s = data.latest_signals || {};
    const date = s.date || '';
    const dayN = data.system_health ? 'Day ' + data.system_health.total_inference_days : '';

    el.innerHTML = `
      <div class="hero-row">
        <div class="signal-date">${date}</div>
        <div class="hero-dayn">${dayN}</div>
      </div>
      <div class="detail-subtitle">End to End Deep Learning Quantitative Trading System</div>
    `;
  }

  /* ─── Signal header: title + summary stats ─── */
  function renderSignalHeader(data) {
    const el = document.getElementById('signal-header');
    if (!el || !data.latest_signals) return;

    const s = data.latest_signals;
    const stats = s.statistics;

    el.innerHTML = `
      <div class="signal-title">Today's Signals</div>
      <div class="signal-summary">
        <div class="signal-stat">
          <div class="signal-stat-value">${stats.total_count}</div>
          <div class="signal-stat-label">Universe</div>
        </div>
        <div class="signal-stat">
          <div class="signal-stat-value accent">${stats.positive_count}</div>
          <div class="signal-stat-label">Positive</div>
        </div>
        <div class="signal-stat">
          <div class="signal-stat-value">${pct(stats.mean_confidence, 1)}</div>
          <div class="signal-stat-label">Avg Confidence</div>
        </div>
      </div>
    `;
  }

  /* ─── Signal table: top 30 ─── */
  function renderSignalTable(data) {
    const el = document.getElementById('signal-table');
    if (!el || !data.latest_signals) return;

    const signals = data.latest_signals.top_k;

    el.innerHTML = `
      <thead>
        <tr>
          <th>Rank</th>
          <th>Instrument</th>
          <th class="r">Score</th>
          <th class="r">Confidence</th>
        </tr>
      </thead>
      <tbody>
        ${signals.map(s => {
          const scoreClass = s.score >= 0 ? 'score-pos' : 'score-neg';
          const confClass = s.confidence >= 0.975 ? 'conf-high' : '';
          return `
          <tr>
            <td class="rank-cell">${s.rank}</td>
            <td class="ticker">${s.instrument}</td>
            <td class="r ${scoreClass}">${s.score >= 0 ? '+' : ''}${s.score.toFixed(5)}</td>
            <td class="r ${confClass}">${(s.confidence * 100).toFixed(1)}%</td>
          </tr>`;
        }).join('')}
      </tbody>
    `;
  }

  /* ─── Metrics grid ─── */
  function renderMetrics(data) {
    const el = document.getElementById('metrics-grid');
    if (!el || !data.performance) return;

    const p = data.performance;

    // Cumulative return from equity curve (changes daily)
    const cumRet = Number(p.cumulative_return || 0);
    const cumRetStr = (cumRet >= 0 ? '+' : '') + (cumRet * 100).toFixed(2) + '%';

    // Annualized return (extrapolated from cumulative return over trading period)
    const annRet = Number(p.annualized_return || 0);
    const annRetStr = (annRet >= 0 ? '+' : '') + (annRet * 100).toFixed(2) + '%';

    // Sharpe — now computed from universe-wide returns (realistic range)
    const sharpe = Number(p.topk_sharpe_ratio || 0).toFixed(2);

    // Today's top-K alpha score (changes every day)
    const alpha = Number(p.latest_topk_alpha || 0);
    const alphaStr = (alpha >= 0 ? '+' : '') + (alpha * 100).toFixed(2) + '%';

    // Win rate — universe-based (no longer always 100%)
    const winRate = pct(p.topk_win_rate, 0);

    // Max drawdown — universe-based (no longer always 0%)
    const maxDd = Number(p.topk_max_drawdown || 0);
    const maxDdStr = (maxDd * 100).toFixed(2) + '%';

    // Today's confidence (per-day value, not all-time average)
    const conf = pct(p.latest_confidence || p.avg_confidence, 1);

    const items = [
      [cumRetStr, 'Cumulative'],
      [annRetStr, 'Annualized'],
      [sharpe, 'Sharpe'],
      [alphaStr, "Today's Alpha"],
      [winRate, 'Win Rate'],
      [maxDdStr, 'Max Drawdown'],
      [conf, 'Confidence']
    ];

    el.innerHTML = items.map(([v, l]) => `
      <div class="metric-block">
        <div class="metric-block-value">${v}</div>
        <div class="metric-block-label">${l}</div>
      </div>
    `).join('');
  }

  /* ─── Pipeline ─── */
  function renderPipeline(data) {
    const el = document.getElementById('pipeline-content');
    if (!el) return;

    const stages = [
      { n: '01', t: 'Data', items: ['Tushare API', 'Qlib binary export', 'Alpha158 + 2 custom factors', '22 features \u00d7 72 timesteps'] },
      { n: '02', t: 'Model', items: ['Causal TCN (3 blocks)', 'Local self attention', 'GRU aggregation', '246K parameters'] },
      { n: '03', t: 'Inference', items: ['MC Dropout (16 passes)', 'Confidence scoring', 'Cross sectional ranking', 'Risk evaluation'] },
      { n: '04', t: 'Execution', items: ['TopK rebalancing', 'Waterfill sizing', 'Multi account orders', 'GM Trade API'] }
    ];

    el.innerHTML = `
      <div class="pipeline">
        ${stages.map(s => `
          <div class="pipe-stage">
            <div class="pipe-num">${s.n}</div>
            <div class="pipe-title">${s.t}</div>
            <div class="pipe-items">${s.items.join('<br>')}</div>
          </div>
        `).join('')}
      </div>
    `;
  }

  /* ─── Specs ─── */
  function renderSpecs(data) {
    const el = document.getElementById('spec-content');
    if (!el || !data.system_info) return;

    const m = data.system_info.model;
    const t = data.system_info.training;
    const specs = [
      ['Architecture', m.name.replace(/_/g, ' ')],
      ['Parameters', m.total_parameters.toLocaleString()],
      ['Input', m.input_features + ' \u00d7 ' + m.input_timesteps],
      ['Embedding', 'd_model = ' + m.d_model],
      ['Attention', m.attention_heads + ' heads, window ' + m.window_size],
      ['GRU Hidden', String(m.gru_hidden_size)],
      ['Loss', t.loss_function],
      ['Optimizer', t.optimizer + ', lr ' + t.learning_rate],
      ['Precision', t.precision],
      ['MC Samples', String(data.system_info.inference.mc_dropout_samples)],
    ];

    el.innerHTML = `
      <div class="spec-grid">
        ${specs.map(([k, v]) => `
          <div class="spec-row">
            <span class="spec-k">${k}</span>
            <span class="spec-v">${v}</span>
          </div>
        `).join('')}
      </div>
    `;
  }

  /* ─── Backtest ─── */
  function renderBacktest(data) {
    const el = document.getElementById('backtest-content');
    if (!el) return;

    el.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th class="r">With Cost</th>
            <th class="r">Without Cost</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Annualized Excess Return</td>
            <td class="r strong">16.72%</td>
            <td class="r">21.38%</td>
          </tr>
          <tr>
            <td>Max Drawdown</td>
            <td class="r strong">\u22124.60%</td>
            <td class="r">\u22124.41%</td>
          </tr>
          <tr>
            <td>Information Ratio</td>
            <td class="r strong">1.96</td>
            <td class="r">2.51</td>
          </tr>
        </tbody>
      </table>
      <p style="font-family:var(--font-mono);font-size:11px;color:var(--grey-400);margin-top:16px;line-height:1.8;letter-spacing:0.02em;">
        CSI300 universe. TopkDropoutStrategy (Top 50, Drop 5). Period 2024.01 to 2025.07.
        Transaction costs: open 0.05%, close 0.15%. Excess return vs CSI300 benchmark.
      </p>
    `;
  }

  /* ─── Trading ─── */
  function renderTrading(data) {
    const el = document.getElementById('trading-content');
    if (!el || !data.trading) return;

    const t = data.trading;
    const targets = t.latest_targets || {};
    const tList = targets.targets || [];
    const orders = (t.latest_orders || {}).orders || [];
    const risk = t.latest_risk_state || {};

    // Capital + summary stats header
    const totalNotional = tList.reduce((acc, tgt) => acc + (tgt.target_shares || 0) * (tgt.ref_price || 0), 0);
    const buyOrders = orders.filter(o => o.side === 'BUY');
    const sellOrders = orders.filter(o => o.side === 'SELL');

    let html = `
      <div class="trading-header">
        <div class="trading-capital">
          <div class="trading-capital-value">\u00a53,000,000</div>
          <div class="trading-capital-label">Capital</div>
        </div>
        <div class="trading-stats">
          <div class="trading-stat">
            <div class="trading-stat-value">${targets.target_count || tList.length}</div>
            <div class="trading-stat-label">Positions</div>
          </div>
          <div class="trading-stat">
            <div class="trading-stat-value">\u00a5${totalNotional > 0 ? Math.round(totalNotional).toLocaleString() : '0'}</div>
            <div class="trading-stat-label">Allocated</div>
          </div>
          <div class="trading-stat">
            <div class="trading-stat-value" style="color:var(--positive)">${buyOrders.length}</div>
            <div class="trading-stat-label">Buy</div>
          </div>
          <div class="trading-stat">
            <div class="trading-stat-value" style="color:var(--negative)">${sellOrders.length}</div>
            <div class="trading-stat-label">Sell</div>
          </div>
        </div>
      </div>
    `;

    // Position targets table
    if (tList.length > 0) {
      html += `
        <table class="order-table">
          <thead>
            <tr>
              <th>Instrument</th>
              <th class="r">Weight</th>
              <th class="r">Shares</th>
              <th class="r">Ref Price</th>
              <th class="r">Notional</th>
            </tr>
          </thead>
          <tbody>
            ${tList.map(tgt => {
              const notional = (tgt.target_shares || 0) * (tgt.ref_price || 0);
              return `
              <tr>
                <td class="sym">${tgt.instrument}</td>
                <td class="r">${(tgt.weight * 100).toFixed(2)}%</td>
                <td class="r" style="font-weight:800">${(tgt.target_shares || 0).toLocaleString()}</td>
                <td class="r">${(tgt.ref_price || 0).toFixed(2)}</td>
                <td class="r" style="font-weight:800">\u00a5${notional > 0 ? Math.round(notional).toLocaleString() : '0'}</td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      `;
    }

    // Latest orders table
    if (orders.length > 0) {
      html += `
        <table class="order-table" style="margin-top:24px;">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Side</th>
              <th class="r">Volume</th>
              <th class="r">Price</th>
              <th class="r">Notional</th>
              <th>Type</th>
            </tr>
          </thead>
          <tbody>
            ${orders.map(o => {
              const n = o.volume * o.price;
              return `
              <tr>
                <td class="sym">${o.symbol}</td>
                <td><span class="${o.side === 'BUY' ? 'side-buy' : 'side-sell'}">${o.side}</span></td>
                <td class="r" style="font-weight:800">${o.volume.toLocaleString()}</td>
                <td class="r">${o.price.toFixed(2)}</td>
                <td class="r">\u00a5${Math.round(n).toLocaleString()}</td>
                <td>${o.type || 'limit'}</td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      `;
    }

    // Risk state
    html += `
      <div class="risk-row">
        <div class="risk-item">
          <div class="risk-item-label">Allow Buy</div>
          <div class="risk-item-value ${risk.allow_buy ? 'risk-ok' : ''}">${risk.allow_buy ? 'Yes' : 'No'}</div>
        </div>
        <div class="risk-item">
          <div class="risk-item-label">Day Drawdown</div>
          <div class="risk-item-value">${((risk.day_drawdown || 0) * 100).toFixed(2)}%</div>
        </div>
        <div class="risk-item">
          <div class="risk-item-label">5D Drawdown</div>
          <div class="risk-item-value">${((risk.rolling5d_drawdown || 0) * 100).toFixed(2)}%</div>
        </div>
        <div class="risk-item">
          <div class="risk-item-label">Limit Up Filtered</div>
          <div class="risk-item-value">${(risk.limit_up_symbols || []).length > 0 ? risk.limit_up_symbols.join(', ') : 'None'}</div>
        </div>
      </div>
    `;

    el.innerHTML = html;
  }

  return {
    renderDetailHero,
    renderSignalHeader,
    renderSignalTable,
    renderMetrics,
    renderPipeline,
    renderSpecs,
    renderBacktest,
    renderTrading
  };
})();
