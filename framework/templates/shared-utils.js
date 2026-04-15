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
