/**
 * data-loader.js — Fetch and cache dashboard JSON data
 */
const DataLoader = (() => {
  const BASE = './data';
  let _cache = null;

  async function load() {
    if (_cache) return _cache;
    try {
      const res = await fetch(`${BASE}/dashboard_data.json`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      _cache = await res.json();
      return _cache;
    } catch (err) {
      console.error('[DataLoader] Failed to load dashboard data:', err);
      return null;
    }
  }

  function get() { return _cache; }

  return { load, get };
})();
