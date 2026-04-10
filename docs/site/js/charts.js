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

    return new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
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
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
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
                return [
                  `Cumulative  +${(d.cumulative_return * 100).toFixed(2)}%`,
                  `Daily       +${(d.daily_return * 100).toFixed(2)}%`,
                  `NAV          ${d.equity.toFixed(4)}`
                ];
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
              callback: (v) => `+${v.toFixed(0)}%`
            },
            border: { display: false }
          }
        }
      }
    });
  }

  return { renderEquityCurve };
})();
