/**
 * framework/templates/shared-utils.js
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Shared JavaScript utilities for all benchb0t templates.
 * Consolidated from duplicated code in dashboard.html, builder.html, history.html, analytics.html.
 *
 * Include this before other scripts:
 *   <script src="/static/shared-utils.js"></script>
 */

/**
 * HTML-escape a string for safe insertion into DOM.
 * Escapes: &, <, >, and " (for attribute values).
 * Used consistently across all templates.
 */
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/**
 * Get trimmed value from a form input element by ID.
 * Returns empty string if element not found.
 * Commonly used in form submissions (builder.html, dashboard.html).
 */
function val(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : '';
}

/**
 * Safely set innerHTML, escaping all content.
 * Alternative to: el.innerHTML = esc(text)
 */
function setText(el, text) {
  if (!el) return;
  el.textContent = text;
}

/**
 * Safely set an HTML attribute, escaping the value.
 */
function setAttr(el, name, value) {
  if (!el) return;
  el.setAttribute(name, String(value ?? ''));
}

/**
 * Check if an element has a given CSS class.
 */
function hasClass(el, cls) {
  return el ? el.classList.contains(cls) : false;
}

/**
 * Add CSS class(es) to an element.
 */
function addClass(el, cls) {
  if (!el) return;
  if (Array.isArray(cls)) {
    cls.forEach(c => el.classList.add(c));
  } else {
    el.classList.add(cls);
  }
}

/**
 * Remove CSS class(es) from an element.
 */
function removeClass(el, cls) {
  if (!el) return;
  if (Array.isArray(cls)) {
    cls.forEach(c => el.classList.remove(c));
  } else {
    el.classList.remove(cls);
  }
}

/**
 * Toggle CSS class(es) on an element.
 */
function toggleClass(el, cls, force) {
  if (!el) return;
  if (Array.isArray(cls)) {
    cls.forEach(c => el.classList.toggle(c, force));
  } else {
    el.classList.toggle(cls, force);
  }
}

/**
 * Generic fetch + JSON parsing with error handling.
 * Returns [data, error] tuple.
 */
async function fetchJson(url, options = {}) {
  try {
    const resp = await fetch(url, options);
    if (!resp.ok) {
      return [null, `HTTP ${resp.status}: ${resp.statusText}`];
    }
    const data = await resp.json();
    return [data, null];
  } catch (e) {
    return [null, String(e.message || e)];
  }
}

/**
 * Show a temporary loading spinner / feedback message.
 * Usage: showLoader(el, 'Loading...'); await someWork(); hideLoader(el);
 */
function showLoader(el, message = 'Loading...') {
  if (!el) return;
  el.setAttribute('disabled', 'disabled');
  el.setAttribute('data-loading-msg', message);
  el.textContent = message;
  el.style.opacity = '0.6';
}

function hideLoader(el, originalText = '') {
  if (!el) return;
  el.removeAttribute('disabled');
  el.textContent = originalText || el.getAttribute('data-loading-msg') || 'Done';
  el.style.opacity = '1';
}

/**
 * Format a number as a fixed-point string.
 */
function fmt(num, decimals = 2) {
  return Number(num).toFixed(decimals);
}

/**
 * Artifact helpers
 * ~~~~~~~~~~~~~~~~
 * Shared across history + dashboard for rendering anomaly reports and
 * prominent bundle downloads.
 */

/**
 * Find the first artifact of a given kind in an artifact list.
 */
function findArtifact(artifacts, kind) {
  if (!Array.isArray(artifacts)) return null;
  return artifacts.find(a => (a || {}).kind === kind) || null;
}

/**
 * Map an anomaly severity to a CSS class suffix.
 */
function severityClass(severity) {
  const s = String(severity || 'none').toLowerCase();
  if (s === 'high') return 'sev-high';
  if (s === 'medium') return 'sev-medium';
  if (s === 'low') return 'sev-low';
  return 'sev-none';
}

/**
 * Render a prominent anomaly panel inline in a run detail view.
 * Shows a colored severity badge, the item count, a per-item list, and the
 * optional LLM narrative summary. Returns '' when no anomaly report exists.
 */
function renderAnomalyPanel(artifacts) {
  const anomaly = findArtifact(artifacts, 'anomalies');
  if (!anomaly) return '';
  const summary = anomaly.anomaly_summary || {};
  const items = Array.isArray(anomaly.anomaly_items) ? anomaly.anomaly_items : [];
  const severity = summary.severity || 'none';
  const count = summary.count != null ? summary.count : items.length;
  const sevLabel = severity.toUpperCase();

  const itemsHtml = items.length ? `
    <ul class="anomaly-items">
      ${items.slice(0, 12).map(it => `
        <li class="anomaly-item ${severityClass(it.severity)}">
          <span class="anomaly-item-kind">${esc((it.kind || 'item').replace(/_/g, ' '))}</span>
          <span class="anomaly-item-sev">${esc(String(it.severity || 'low').toUpperCase())}</span>
          <span class="anomaly-item-detail">${esc(it.detail || '')}</span>
        </li>
      `).join('')}
      ${items.length > 12 ? `<li class="anomaly-item-more">… ${items.length - 12} more</li>` : ''}
    </ul>
  ` : `<div class="anomaly-empty">No notable events detected.</div>`;

  const llmHtml = anomaly.anomaly_llm_summary
    ? `<div class="anomaly-llm">${esc(anomaly.anomaly_llm_summary)}</div>`
    : '';

  return `
    <div class="anomaly-panel ${severityClass(severity)}">
      <div class="anomaly-head">
        <span class="anomaly-sev-badge ${severityClass(severity)}">${esc(sevLabel)}</span>
        <span class="anomaly-title">Anomaly report</span>
        <span class="anomaly-count">${count} item${count === 1 ? '' : 's'}</span>
        <a class="anomaly-raw" href="${esc(anomaly.url || '#')}" target="_blank" rel="noopener">raw JSON</a>
      </div>
      ${itemsHtml}
      ${llmHtml}
    </div>
  `;
}

/**
 * Render a prominent "Download bundle" call-to-action for a run.
 * Returns '' when no result bundle artifact exists.
 */
function renderBundleButton(artifacts) {
  const bundle = findArtifact(artifacts, 'result_bundle');
  if (!bundle) return '';
  return `
    <a class="bundle-cta" href="${esc(bundle.url || '#')}" download="${esc(bundle.name || 'result-bundle.zip')}">
      <span class="bundle-cta-icon">⬇</span>
      <span class="bundle-cta-text">Download bundle</span>
      <span class="bundle-cta-size">${esc(formatBytes(bundle.size_bytes || 0))}</span>
    </a>
  `;
}

/**
 * Debounce a function call.
 * Returns a debounced version that won't fire until `delayMs` has passed since last call.
 */
function debounce(fn, delayMs = 300) {
  let timeoutId = null;
  return function debounced(...args) {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delayMs);
  };
}

/**
 * ProviderManager
 * ~~~~~~~~~~~~~~~
 * Reusable saved-provider selector with manual fallback.
 *
 * Usage:
 *   const pm = new ProviderManager('my-provider-box', {
 *     onChange: (provider) => { ... },
 *     showSave: true,
 *   });
 *   await pm.load();
 *
 * Renders a dropdown of saved providers, a "manual" option, and a compact
 * manual-entry panel. When showSave is true the user can persist the manual
 * provider to the server so it appears in the dropdown everywhere.
 */
class ProviderManager {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`ProviderManager: container #${containerId} not found`);
    }
    this.options = {
      showSave: true,
      saveLabel: 'SAVE PROVIDER',
      manualLabel: '— manual —',
      placeholderUrl: 'http://localhost:11434/v1',
      placeholderModel: 'llama3',
      placeholderKey: '(empty for local)',
      onChange: null,
      ...options,
    };
    this.providers = [];
    this.selected = { base_url: '', model: '', api_key: '', label: '', source: 'manual', enabled: true };
    this.selectId = containerId + '-select';
    this.panelId = containerId + '-panel';
    this.statusId = containerId + '-status';
    this.urlId = containerId + '-url';
    this.modelId = containerId + '-model';
    this.keyId = containerId + '-key';
    this.labelId = containerId + '-label';
    this._renderSkeleton();
  }

  _renderSkeleton() {
    const o = this.options;
    this.container.innerHTML = `
      <div class="pm-bar">
        <span class="pm-label">Provider</span>
        <select class="pm-select" id="${this.selectId}"></select>
        <button class="pm-gear" id="${this.container.id}-gear" title="Configure provider">⚙</button>
      </div>
      <div class="pm-panel" id="${this.panelId}">
        <div class="pm-field">
          <label>Base URL</label>
          <input type="text" id="${this.urlId}" placeholder="${esc(o.placeholderUrl)}">
        </div>
        <div class="pm-field">
          <label>Model</label>
          <input type="text" id="${this.modelId}" placeholder="${esc(o.placeholderModel)}">
        </div>
        <div class="pm-field">
          <label>API Key</label>
          <input type="password" id="${this.keyId}" placeholder="${esc(o.placeholderKey)}">
        </div>
        <div class="pm-field" id="${this.labelId}-wrap" style="display:none">
          <label>Label</label>
          <input type="text" id="${this.labelId}" placeholder="display name">
        </div>
        <div class="pm-actions">
          <button class="pm-use" id="${this.container.id}-use">USE THIS PROVIDER</button>
          ${o.showSave ? `<button class="pm-save" id="${this.container.id}-save">${esc(o.saveLabel)}</button>` : ''}
        </div>
        <div class="pm-status" id="${this.statusId}"></div>
      </div>
    `;

    this.container.querySelector(`#${this.container.id}-gear`).onclick = () => this.togglePanel();
    this.container.querySelector(`#${this.container.id}-use`).onclick = () => this.applyManual();
    if (o.showSave) {
      this.container.querySelector(`#${this.container.id}-save`).onclick = () => this.saveCurrentProvider();
    }
    this.container.querySelector(`#${this.selectId}`).onchange = (e) => this.onSelect(e.target.value);
  }

  async load() {
    let providers = [];
    try {
      const resp = await fetch('/api/providers');
      providers = await resp.json();
    } catch (e) {
      providers = [];
    }
    if (!Array.isArray(providers) || !providers.length) {
      try {
        const creds = await fetch('/api/credentials').then(r => r.json());
        if (Array.isArray(creds.providers) && creds.providers.length) {
          providers = creds.providers.map((p, i) => ({
            id: `legacy-${p.model || i}`,
            label: p.label || p.model || 'legacy',
            base_url: p.base_url,
            model: p.model,
            api_key: p.api_key || '',
            source: 'legacy:creds',
            enabled: true,
          }));
        } else if (creds.base_url && creds.model) {
          providers = [{
            id: `legacy-${creds.model}`,
            label: creds.model,
            base_url: creds.base_url,
            model: creds.model,
            api_key: creds.api_key || '',
            source: 'legacy:creds',
            enabled: true,
          }];
        }
      } catch (e) {
        providers = [];
      }
    }
    this.providers = providers.filter(p => p.base_url && p.model && p.enabled !== false);
    this.render();
    if (this.options.onChange) this.options.onChange(this.getSelected());
  }

  render() {
    const sel = document.getElementById(this.selectId);
    if (!sel) return;
    const selectedKey = `${this.selected.base_url}|${this.selected.model}`;
    let html = '';
    this.providers.forEach(p => {
      const key = `${p.base_url}|${p.model}`;
      const host = String(p.base_url).replace(/^https?:\/\//, '');
      const label = esc(p.label || p.model || 'provider');
      html += `<option value="${esc(key)}" ${key === selectedKey ? 'selected' : ''}>${label} @ ${esc(host)}</option>`;
    });
    html += `<option value="__manual__" ${selectedKey === '|' || !this.providers.length ? 'selected' : ''}>${esc(this.options.manualLabel)}</option>`;
    sel.innerHTML = html;
    this._syncInputs();
  }

  _syncInputs() {
    const setVal = (id, v) => {
      const el = document.getElementById(id);
      if (el) el.value = v || '';
    };
    setVal(this.urlId, this.selected.base_url);
    setVal(this.modelId, this.selected.model);
    setVal(this.keyId, this.selected.api_key);
    setVal(this.labelId, this.selected.label);
  }

  onSelect(value) {
    if (value === '__manual__') {
      this.openPanel();
      return;
    }
    const [base_url, ...rest] = value.split('|');
    const model = rest.join('|');
    const p = this.providers.find(x => x.base_url === base_url && x.model === model);
    if (p) {
      this.selected = { ...p };
      this.closePanel();
      if (this.options.onChange) this.options.onChange(this.getSelected());
    }
  }

  setSelected(provider) {
    this.selected = {
      base_url: provider.base_url || '',
      model: provider.model || '',
      api_key: provider.api_key || '',
      label: provider.label || provider.model || '',
      source: provider.source || 'manual',
      enabled: provider.enabled !== false,
    };
    this.render();
  }

  getSelected() {
    return { ...this.selected };
  }

  applyManual() {
    const url = document.getElementById(this.urlId).value.trim();
    const model = document.getElementById(this.modelId).value.trim();
    const api_key = document.getElementById(this.keyId).value.trim();
    const status = document.getElementById(this.statusId);
    if (!url || !model) {
      if (status) status.textContent = 'URL and model required';
      return;
    }
    const labelWrap = document.getElementById(this.labelId + '-wrap');
    const labelInput = document.getElementById(this.labelId);
    const label = (labelWrap && labelWrap.style.display !== 'none' && labelInput)
      ? (labelInput.value.trim() || model)
      : model;
    this.selected = {
      base_url: url,
      model,
      api_key,
      label,
      source: 'manual',
      enabled: true,
    };
    this.render();
    this.closePanel();
    if (status) status.textContent = '';
    if (this.options.onChange) this.options.onChange(this.getSelected());
  }

  async saveCurrentProvider() {
    const status = document.getElementById(this.statusId);
    const panel = document.getElementById(this.panelId);
    if (panel && panel.classList.contains('open')) {
      this.applyManual();
    }
    const p = this.getSelected();
    if (!p.base_url || !p.model) {
      if (status) status.textContent = 'Fill URL and model first';
      return;
    }
    if (status) status.textContent = 'Saving…';
    try {
      const resp = await fetch('/api/onboarding/providers/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          providers: [{
            id: p.id || `p${Date.now()}`,
            label: p.label || p.model,
            base_url: p.base_url,
            model: p.model,
            api_key: p.api_key,
            source: p.source || 'manual',
            enabled: true,
          }],
        }),
      });
      if (!resp.ok) throw new Error('save failed');
      if (status) status.textContent = 'Saved ✔';
      await this.load();
      this.setSelected(p);
    } catch (e) {
      if (status) status.textContent = 'Save failed';
    }
  }

  openPanel() {
    const panel = document.getElementById(this.panelId);
    if (panel) panel.classList.add('open');
  }

  closePanel() {
    const panel = document.getElementById(this.panelId);
    if (panel) panel.classList.remove('open');
  }

  togglePanel() {
    const panel = document.getElementById(this.panelId);
    if (panel) panel.classList.toggle('open');
  }
}
