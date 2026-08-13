/*
 * redwing-fp.js - the device fingerprint collector.
 *
 * The client half of core/fingerprint.py. Runs in the institution's OWN app or web session,
 * which is the only place fingerprinting is possible: a card authorization reaches an issuer as
 * an ISO 8583 message with no browser attached, so there is nothing to collect there.
 *
 * WHAT IT COLLECTS, and the split matters more than the list. Components are tagged ANCHOR
 * (hardware, slow to change, decides identity) or DRIFT (software surface, changes constantly,
 * used only to re-link a device whose anchor moved). Hashing them all together is the standard
 * mistake: it yields a new identity every time a browser updates.
 *
 * WHAT IT DELIBERATELY DOES NOT DO:
 *   - No cookies, no localStorage, no persistent client-side id. The identity is derived
 *     server-side from what the hardware reports, so clearing storage does not reset it and,
 *     equally, nothing is written to the user's device.
 *   - No IP geolocation lookups from the client.
 *   - Never blocks or delays the page. Every probe is wrapped and falls back to a null marker,
 *     because a fingerprint that breaks a checkout is worse than no fingerprint.
 *
 * A BLOCKED PROBE REPORTS "blocked", NOT A GUESS. core/fingerprint.py counts null markers as
 * ZERO entropy, which is what stops a privacy-hardened browser (all of which return identical
 * blocked values) from being treated as a confident identity and read as a shared fraud device.
 *
 *   RedWingFP.collect().then(c => fetch('/telemetry/fingerprint', {
 *     method: 'POST', headers: {'Content-Type': 'application/json'},
 *     body: JSON.stringify({subject_ref: txnId, components: c})
 *   }));
 */
(function (global) {
  'use strict';

  var NULL = 'blocked';

  function safe(fn) {
    try {
      var v = fn();
      return (v === undefined || v === null || v === '') ? NULL : v;
    } catch (e) {
      return NULL;
    }
  }

  // Small, fast, non-cryptographic. The server re-hashes with SHA-256; this only needs to
  // compress a long string into a stable token without shipping the whole thing.
  function fnv1a(str) {
    var h = 0x811c9dc5;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return ('00000000' + h.toString(16)).slice(-8);
  }

  // ---- ANCHOR: hardware and platform -------------------------------------

  function webgl() {
    return safe(function () {
      var c = document.createElement('canvas');
      var gl = c.getContext('webgl') || c.getContext('experimental-webgl');
      if (!gl) return NULL;
      var dbg = gl.getExtension('WEBGL_debug_renderer_info');
      var renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : NULL;
      var vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : NULL;
      // Parameter surface is DRIFT: driver updates move these while the GPU stays the same.
      var params = [
        gl.getParameter(gl.MAX_TEXTURE_SIZE),
        gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
        gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
        gl.getParameter(gl.MAX_VARYING_VECTORS),
        (gl.getSupportedExtensions() || []).join(',')
      ].join('|');
      return { renderer: renderer, vendor: vendor, params_hash: fnv1a(params) };
    });
  }

  // Audio DSP characteristics. A hardware/stack property rather than a software one, which is
  // why it sits in the ANCHOR set: the same machine produces the same output across browser
  // versions. Offline so nothing is played.
  function audioDsp() {
    return new Promise(function (resolve) {
      try {
        var OC = global.OfflineAudioContext || global.webkitOfflineAudioContext;
        if (!OC) return resolve(NULL);
        var ctx = new OC(1, 44100, 44100);
        var osc = ctx.createOscillator();
        osc.type = 'triangle';
        osc.frequency.value = 10000;
        var comp = ctx.createDynamicsCompressor();
        comp.threshold.value = -50; comp.knee.value = 40; comp.ratio.value = 12;
        comp.attack.value = 0; comp.release.value = 0.25;
        osc.connect(comp); comp.connect(ctx.destination);
        osc.start(0);
        var done = false;
        var timer = setTimeout(function () { if (!done) { done = true; resolve(NULL); } }, 1200);
        ctx.oncomplete = function (e) {
          if (done) return;
          done = true; clearTimeout(timer);
          try {
            var buf = e.renderedBuffer.getChannelData(0);
            var acc = 0;
            for (var i = 4500; i < 5000; i++) acc += Math.abs(buf[i]);
            resolve(fnv1a(acc.toString()));
          } catch (err) { resolve(NULL); }
        };
        ctx.startRendering();
      } catch (e) { resolve(NULL); }
    });
  }

  // ---- DRIFT: software surface -------------------------------------------

  function canvasHash() {
    return safe(function () {
      var c = document.createElement('canvas');
      c.width = 280; c.height = 60;
      var ctx = c.getContext('2d');
      if (!ctx) return NULL;
      ctx.textBaseline = 'top';
      ctx.font = "14px 'Arial'";
      ctx.fillStyle = '#f60'; ctx.fillRect(125, 1, 62, 20);
      ctx.fillStyle = '#069'; ctx.fillText('RedWing ⊕ fp 1.0', 2, 15);
      ctx.fillStyle = 'rgba(102,204,0,0.7)'; ctx.fillText('RedWing ⊕ fp 1.0', 4, 17);
      var data = c.toDataURL();
      // A blocked/noised canvas returns a constant or a fresh random each call. Either way it is
      // not identifying, and the server must see a null marker rather than a plausible hash.
      if (!data || data.length < 128) return NULL;
      return fnv1a(data);
    });
  }

  function fontHash() {
    return safe(function () {
      var probes = ['Arial', 'Verdana', 'Times New Roman', 'Courier New', 'Georgia',
        'Palatino', 'Garamond', 'Comic Sans MS', 'Trebuchet MS', 'Impact',
        'Tahoma', 'Geneva', 'Helvetica Neue', 'Menlo', 'Consolas', 'Segoe UI',
        'Roboto', 'Ubuntu', 'Cantarell', 'DejaVu Sans'];
      var base = ['monospace', 'sans-serif', 'serif'];
      var span = document.createElement('span');
      span.style.cssText = 'position:absolute;left:-9999px;font-size:72px;';
      span.textContent = 'mmmmmmmmmmlli';
      document.body.appendChild(span);
      var widths = {};
      base.forEach(function (b) {
        span.style.fontFamily = b;
        widths[b] = [span.offsetWidth, span.offsetHeight];
      });
      var found = [];
      probes.forEach(function (f) {
        for (var i = 0; i < base.length; i++) {
          span.style.fontFamily = "'" + f + "'," + base[i];
          if (span.offsetWidth !== widths[base[i]][0] ||
              span.offsetHeight !== widths[base[i]][1]) { found.push(f); break; }
        }
      });
      document.body.removeChild(span);
      return found.length ? fnv1a(found.join(',')) : NULL;
    });
  }

  // ---- automation and integrity ------------------------------------------
  //
  // Not identity. Actionable even when the fingerprint is too weak to identify anyone, which is
  // the common case for a scripted client: bots often run hardened, low-entropy browsers.

  function automation() {
    var out = {};
    out.webdriver = safe(function () { return navigator.webdriver === true; }) === true;
    out.headless = safe(function () {
      // Headless Chrome historically ships no plugins and a UA marker; neither alone is proof.
      var ua = (navigator.userAgent || '').toLowerCase();
      if (ua.indexOf('headless') !== -1) return true;
      return (navigator.plugins && navigator.plugins.length === 0) &&
             (navigator.languages && navigator.languages.length === 0);
    }) === true;
    out.cdp_detected = safe(function () {
      // A DevTools-Protocol driver leaves a stack-serialisation tell on Error.
      var e = new Error(); var hit = false;
      Object.defineProperty(e, 'stack', { get: function () { hit = true; return ''; } });
      // eslint-disable-next-line no-console
      console.debug(e);
      return hit;
    }) === true;
    out.selenium_markers = safe(function () {
      for (var k in global) {
        if (k.indexOf('selenium') === 0 || k.indexOf('_Selenium') === 0 ||
            k.indexOf('callSelenium') === 0) return true;
      }
      return !!(document.documentElement.getAttribute('selenium') ||
                document.documentElement.getAttribute('webdriver') ||
                document.documentElement.getAttribute('driver'));
    }) === true;
    out.phantom_markers = safe(function () {
      return !!(global.callPhantom || global._phantom || global.__nightmare);
    }) === true;
    out.playwright_markers = safe(function () {
      return !!(global.__playwright || global.__pw_manual || global.__PW_inspect);
    }) === true;
    return out;
  }

  // ---- collect ------------------------------------------------------------

  function collect() {
    var gl = webgl();
    var glOk = gl && gl !== NULL;
    return audioDsp().then(function (dsp) {
      var c = {
        fp_version: '1.0',

        // ANCHOR
        gpu_renderer: glOk ? gl.renderer : NULL,
        gpu_vendor: glOk ? gl.vendor : NULL,
        cpu_cores: safe(function () { return navigator.hardwareConcurrency; }),
        device_memory_gb: safe(function () { return navigator.deviceMemory; }),
        screen_w: safe(function () { return screen.width; }),
        screen_h: safe(function () { return screen.height; }),
        color_depth: safe(function () { return screen.colorDepth; }),
        platform: safe(function () {
          return (navigator.userAgentData && navigator.userAgentData.platform) ||
                 navigator.platform;
        }),
        audio_dsp_hash: dsp,
        touch_points: safe(function () { return navigator.maxTouchPoints; }),

        // DRIFT
        browser_family: safe(function () {
          var b = navigator.userAgentData && navigator.userAgentData.brands;
          if (b && b.length) return b[b.length - 1].brand;
          var ua = navigator.userAgent || '';
          return /Firefox/.test(ua) ? 'Firefox' : /Edg/.test(ua) ? 'Edge'
               : /Chrome/.test(ua) ? 'Chrome' : /Safari/.test(ua) ? 'Safari' : NULL;
        }),
        browser_major: safe(function () {
          var m = (navigator.userAgent || '').match(/(?:Chrome|Firefox|Version)\/(\d+)/);
          return m ? m[1] : NULL;
        }),
        timezone: safe(function () {
          return Intl.DateTimeFormat().resolvedOptions().timeZone;
        }),
        language: safe(function () { return navigator.language; }),
        font_hash: fontHash(),
        canvas_hash: canvasHash(),
        webgl_params_hash: glOk ? gl.params_hash : NULL
      };
      var a = automation();
      for (var k in a) { if (a[k]) c[k] = true; }
      return c;
    });
  }

  global.RedWingFP = { collect: collect, version: '1.0' };
})(typeof window !== 'undefined' ? window : this);
