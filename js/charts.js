/**
 * charts.js — Minimal chart rendering, monochrome
 */
const Charts = (() => {

  const BLACK  = '#141414';
  const GREY_4 = '#8a8a8a';
  const GREY_2 = '#c4c4c4';
  const GREY_1 = '#e8e8e8';
  const FILL   = 'rgba(0,0,0,0.04)';

  const FONT_MONO = "'JetBrains Mono', 'SF Mono', monospace";

  function renderEquityCurve(canvasId, equityData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !equityData || equityData.length === 0) return null;

    const labels  = equityData.map(d => d.date);
    const returns = equityData.map(d => d.cumulative_return * 100);

    // CSI300 benchmark data (may be null/missing for some points)
    const hasBenchmark = equityData.some(d => d.benchmark_return != null);
    const benchmarkReturns = equityData.map(d =>
      d.benchmark_return != null ? d.benchmark_return * 100 : null
    );

    const datasets = [
      {
        label: 'Portfolio',
        data: returns,
        borderColor: BLACK,
        backgroundColor: FILL,
        borderWidth: 2,
        fill: true,
        tension: 0.35,
        pointRadius: 5,
        pointBackgroundColor: BLACK,
        pointBorderColor: '#ffffff',
        pointBorderWidth: 2,
        pointHoverRadius: 7
      }
    ];

    if (hasBenchmark) {
      datasets.push({
        label: 'CSI300',
        data: benchmarkReturns,
        borderColor: GREY_4,
        borderDash: [6, 4],
        borderWidth: 1.5,
        fill: false,
        tension: 0.35,
        pointRadius: 3,
        pointBackgroundColor: GREY_4,
        pointBorderColor: '#ffffff',
        pointBorderWidth: 1,
        pointHoverRadius: 5,
        spanGaps: true
      });
    }

    return new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeOutQuart' },
        plugins: {
          legend: {
            display: hasBenchmark,
            position: 'top',
            align: 'end',
            labels: {
              font: { family: FONT_MONO, size: 11, weight: '600' },
              color: GREY_4,
              boxWidth: 24,
              boxHeight: 2,
              padding: 16,
              usePointStyle: false
            }
          },
          tooltip: {
            backgroundColor: BLACK,
            titleFont: { family: FONT_MONO, size: 11, weight: '600' },
            bodyFont: { family: FONT_MONO, size: 12 },
            padding: { x: 12, y: 8 },
            cornerRadius: 0,
            displayColors: false,
            callbacks: {
              title: (items) => items[0].label,
              label: (item) => {
                const d = equityData[item.dataIndex];
                if (item.datasetIndex === 1) {
                  // CSI300 tooltip line
                  if (d.benchmark_return == null) return null;
                  const bSign = d.benchmark_return >= 0 ? '+' : '';
                  return `CSI300      ${bSign}${(d.benchmark_return * 100).toFixed(2)}%`;
                }
                // Portfolio tooltip lines
                const cumSign = d.cumulative_return >= 0 ? '+' : '';
                const daySign = d.daily_return >= 0 ? '+' : '';
                const lines = [
                  `Cumulative  ${cumSign}${(d.cumulative_return * 100).toFixed(2)}%`,
                  `Daily       ${daySign}${(d.daily_return * 100).toFixed(2)}%`,
                ];
                if (d.nav != null) {
                  lines.push(`NAV          ¥${d.nav.toLocaleString()}`);
                } else {
                  lines.push(`NAV          ${d.equity.toFixed(4)}`);
                }
                if (d.benchmark_return != null) {
                  const bSign = d.benchmark_return >= 0 ? '+' : '';
                  lines.push(`CSI300      ${bSign}${(d.benchmark_return * 100).toFixed(2)}%`);
                }
                if (d.topk_alpha != null) {
                  lines.push(`Top-K Alpha +${(d.topk_alpha * 100).toFixed(2)}%`);
                }
                if (d.position_count != null) {
                  lines.push(`Positions    ${d.position_count}`);
                }
                return lines;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: FONT_MONO, size: 11 }, color: GREY_4 },
            border: { color: GREY_2 }
          },
          y: {
            grid: { color: GREY_1, drawBorder: false },
            ticks: {
              font: { family: FONT_MONO, size: 11 },
              color: GREY_4,
              callback: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
            },
            border: { display: false }
          }
        }
      }
    });
  }

  return { renderEquityCurve };
})();

