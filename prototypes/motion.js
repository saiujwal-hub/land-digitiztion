/* ============================================================
   Shared motion layer for the Land Records prototypes.
   Restrained, government-system tone: entrances only, nothing
   loops after its reveal. All effects are opt-in via data
   attributes / classes in the markup, everything is gated on
   `.motion-ready` (set pre-paint only when the user has no
   reduced-motion preference), and final states are the default
   when JS never runs.
   Hooks:
     [data-countup]          number sweeps up, ~620ms ease-out
     [data-stagger]          children fade + translateY in, 40ms/row
     .bar-fill etc.          width fills from 0, 70ms stagger per row
     .chart-bars rect        SVG columns grow from baseline
     .chart-hbars rect       SVG stacked bar grows from the left
     .chart-line             stroke-dashoffset draw-in (dashed paths fade)
     .chart-area             area fill fades in after the line
     .chart-dots circle      endpoint dots pop in, staggered
     .chart-donut circle     donut slices sweep from 0°, in order
   Triggers fire once per page via IntersectionObserver.
   ============================================================ */
(function () {
  "use strict";

  window.__motionReady = true; // tells the inline pre-paint snippet the layer loaded

  var root = document.documentElement;
  var mq = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (mq.matches || !root.classList.contains("motion-ready")) {
    root.classList.remove("motion-ready");
    return;
  }

  /* ---------- injected stylesheet (all states gated on .motion-ready) ---------- */
  var css = [
    ".motion-ready [data-stagger]:not(.in) > * { opacity: 0; }",
    ".motion-ready [data-stagger].in > * { animation: protoRowIn .45s cubic-bezier(.22,.61,.36,1) backwards; }",
    "@keyframes protoRowIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }",
    ".motion-ready .chart-bars:not(.in) rect { transform: scaleY(0); transform-box: fill-box; transform-origin: bottom center; }",
    ".motion-ready .chart-bars.in rect { transform: scaleY(1); transform-box: fill-box; transform-origin: bottom center; transition: transform .55s cubic-bezier(.22,.61,.36,1); }",
    ".motion-ready .chart-hbars:not(.in) rect { transform: scaleX(0); transform-box: fill-box; transform-origin: left center; }",
    ".motion-ready .chart-hbars.in rect { transform: scaleX(1); transform-box: fill-box; transform-origin: left center; transition: transform .55s cubic-bezier(.22,.61,.36,1); }",
    ".motion-ready .chart-dots:not(.in) circle { opacity: 0; transform: scale(0); transform-box: fill-box; transform-origin: center; }",
    ".motion-ready .chart-dots.in circle { opacity: 1; transform: scale(1); transform-box: fill-box; transform-origin: center; transition: opacity .3s ease-out, transform .3s cubic-bezier(.22,.61,.36,1); }"
  ].join("\n");
  var style = document.createElement("style");
  style.id = "proto-motion-css";
  style.textContent = css;
  document.head.appendChild(style);

  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
  var MAX_DELAY = 400; // cap so long tables don't wait forever

  function observe(el, fn, threshold) {
    if (!("IntersectionObserver" in window)) { fn(); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          io.unobserve(entry.target);
          fn();
        }
      });
    }, { threshold: threshold == null ? 0.2 : threshold });
    io.observe(el);
  }

  /* ---------- 1. KPI count-up ---------- */
  function runCountUp(el, delay) {
    var original = el.getAttribute("data-countup-original");
    if (original == null) {
      original = el.textContent;
      el.setAttribute("data-countup-original", original);
    }
    var tokens = original.match(/\d[\d,]*(?:\.\d+)?/g);
    if (!tokens) return;
    var parts = original.split(/\d[\d,]*(?:\.\d+)?/g); // text between the numbers
    var targets = tokens.map(function (s) { return parseFloat(s.replace(/,/g, ""), 10); });
    var grouped = tokens.map(function (s) { return s.indexOf(",") !== -1; });
    var decimals = tokens.map(function (s) {
      var i = s.indexOf(".");
      return i === -1 ? 0 : s.length - i - 1;
    });
    var duration = 620;
    var start = null;
    function frame(ts) {
      if (start == null) start = ts;
      var t = Math.min((ts - start) / duration, 1);
      var e = easeOut(t);
      var out = "";
      for (var i = 0; i < targets.length; i++) {
        var v = targets[i] * e;
        var s = decimals[i] ? v.toFixed(decimals[i]) : Math.round(v).toString();
        if (grouped[i]) s = Number(s).toLocaleString("en-IN");
        out += parts[i] + s;
      }
      out += parts[parts.length - 1] || "";
      el.textContent = out;
      if (t < 1) requestAnimationFrame(frame);
      else el.textContent = original;
    }
    setTimeout(function () { requestAnimationFrame(frame); }, delay || 0);
  }

  /* ---------- 2. progress / meter bars ---------- */
  function prepBars() {
    var bars = document.querySelectorAll(
      ".bar-fill, .progress-list .fill, .prow .fill, .brow .fill, .meter-row .fill, .acc-row .fill, .meter i, .fields .bar i, .duty-row .fill"
    );
    var groups = new Map();
    bars.forEach(function (bar) {
      var target = bar.style.width || getComputedStyle(bar).width;
      if (!target || target === "0px") return;
      var host = bar.closest("[data-bar-group], .panel, .panel-card, .card, .sheet, section") || bar.parentElement;
      if (!groups.has(host)) groups.set(host, []);
      groups.get(host).push({ el: bar, target: target });
    });
    groups.forEach(function (list, host) {
      list.forEach(function (item, i) {
        item.el.style.transition = "none";
        item.el.style.width = "0%";
        observe(host, function () {
          item.el.style.transition = "width .5s cubic-bezier(.22,.61,.36,1)";
          item.el.style.transitionDelay = Math.min(i * 70, MAX_DELAY) + "ms";
          requestAnimationFrame(function () { item.el.style.width = item.target; });
        }, 0.25);
      });
    });
  }

  /* ---------- 3. staggered row/card reveals ---------- */
  function prepStagger() {
    document.querySelectorAll("[data-stagger]").forEach(function (group) {
      observe(group, function () {
        var items = group.children;
        for (var i = 0; i < items.length; i++) {
          items[i].style.animationDelay = Math.min(i * 40, MAX_DELAY) + "ms";
        }
        group.classList.add("in");
      }, 0.15);
    });
  }

  /* ---------- 4. SVG column / stacked-bar charts + endpoint dots ---------- */
  function prepColumnCharts() {
    document.querySelectorAll(".chart-bars, .chart-hbars, .chart-dots").forEach(function (g) {
      observe(g, function () {
        var marks = g.querySelectorAll("rect, circle");
        for (var i = 0; i < marks.length; i++) {
          marks[i].style.transitionDelay = Math.min(i * 60, MAX_DELAY) + "ms";
        }
        g.classList.add("in");
      }, 0.3);
    });
  }

  /* ---------- 5. line charts (dash draw-in; dashed strokes fade) ---------- */
  function prepLineCharts() {
    document.querySelectorAll("svg").forEach(function (svg) {
      if (!svg.querySelector(".chart-line, .chart-area")) return;
      observe(svg, function () {
        svg.querySelectorAll(".chart-line").forEach(function (path, i) {
          if (path.getAttribute("stroke-dasharray")) { // dashed stroke: a draw-in would fight the dash pattern
            path.style.opacity = "0";
            path.style.transition = "opacity .5s ease-out " + Math.min(i * 180, MAX_DELAY) + "ms";
            requestAnimationFrame(function () { path.style.opacity = "1"; });
            return;
          }
          var len = 0;
          try { len = path.getTotalLength(); } catch (e) { return; }
          path.style.strokeDasharray = len + " " + len;
          path.style.strokeDashoffset = len;
          path.getBoundingClientRect(); // commit the hidden state before transitioning
          path.style.transition = "stroke-dashoffset .65s cubic-bezier(.22,.61,.36,1)";
          path.style.strokeDashoffset = "0";
          setTimeout(function () { path.style.strokeDasharray = "none"; }, 900);
        });
        svg.querySelectorAll(".chart-area").forEach(function (area, i) {
          area.style.opacity = "0";
          area.style.transition = "opacity .5s ease-out " + (250 + Math.min(i * 150, MAX_DELAY)) + "ms";
          requestAnimationFrame(function () { area.style.opacity = "1"; });
        });
      }, 0.3);
    });
  }

  /* ---------- 6. donut sweep from 0° ---------- */
  function prepDonuts() {
    document.querySelectorAll(".chart-donut").forEach(function (g) {
      var slices = [];
      g.querySelectorAll("circle").forEach(function (c) {
        var dashattr = c.getAttribute("stroke-dasharray");
        if (!dashattr) return;
        var nums = dashattr.split(/[\s,]+/).map(parseFloat);
        var rAttr = parseFloat(c.getAttribute("r"), 10);
        var rotMatch = (c.getAttribute("transform") || "rotate(0)").match(/rotate\(\s*(-?[\d.]+)/);
        var rot = rotMatch ? parseFloat(rotMatch[1], 10) : 0;
        var C = 2 * Math.PI * (isNaN(rAttr) ? 1 : rAttr);
        slices.push({
          el: c,
          dash: nums[0],
          C: C,
          from: (rot + 90 + 360) % 360, // degrees clockwise from 12 o'clock
          span: Math.max((nums[0] / C) * 360, 0.0001)
        });
      });
      if (!slices.length) return;
      var finals = slices.map(function (s) { return s.el.getAttribute("stroke-dasharray"); });
      observe(g, function () {
        var duration = 700, start = null;
        function frame(ts) {
          if (start == null) start = ts;
          var t = Math.min((ts - start) / duration, 1);
          var sweep = easeOut(t) * 360;
          slices.forEach(function (s) {
            var p = Math.max(0, Math.min(1, (sweep - s.from) / s.span));
            var d = s.dash * p;
            s.el.setAttribute("stroke-dasharray", d + " " + (s.C - d));
          });
          if (t < 1) requestAnimationFrame(frame);
          else slices.forEach(function (s, i) { s.el.setAttribute("stroke-dasharray", finals[i]); });
        }
        requestAnimationFrame(frame);
      }, 0.4);
    });
  }

  /* ---------- boot ---------- */
  function boot() {
    var counters = document.querySelectorAll("[data-countup]");
    counters.forEach(function (el, i) {
      var rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight) {
        runCountUp(el, 120 + Math.min(i * 70, 280));
      } else {
        observe(el, function () { runCountUp(el, 0); }, 0.4);
      }
    });
    prepBars();
    prepStagger();
    prepColumnCharts();
    prepLineCharts();
    prepDonuts();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
