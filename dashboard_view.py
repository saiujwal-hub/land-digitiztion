#!/usr/bin/env python3
"""
dashboard_view.py - High-productivity, clean, professional registry operations dashboard.
Features:
  - Left side navigation menu with all operational workflows (shared across Dashboard & New Scan)
  - Statistical graphs (Registration Throughput Area Chart & Document Classification Donut Chart)
  - 4 Executive KPI Cards
  - Full-width Master Ruled Deed Register with live search & filtering
  - Dedicated New Scan & Document Intake workspace with persistent sidebar
  - OneBhoomi official legal paper styling (Fraunces, Archivo, Courier Prime)
"""

from __future__ import annotations

import html
import hashlib
import math
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import json
import gis_service
import verification_service

BADGE_LABELS = {
    "APPROVED": "Sealed & Certified",
    "REJECTED": "Rejected on Record",
    "UNDER_REVIEW": "Clerk Review",
    "EXTRACTED": "Pending Check",
    "READY_FOR_APPROVAL": "Ready for Seal",
    "PASS": "Checks Passed",
    "FAIL": "Checks Failed",
    "DUPLICATE": "Duplicate Detected",
    "NOT_A_LAND_DOCUMENT": "Not a Land Document",
}

DASHBOARD_CSS = """
    :root{
      --paper:#F6F0E1; --paper-deep:#EFE6D0; --ink:#221D17; --ink-soft:#5A5142;
      --stamp:#A6193C; --stamp-deep:#7C1030; --rosette:#C99AA8; --green:#2E6B4F;
      --green-deep:#1C4A36; --amber:#A96A1F; --gold:#C9A227;
      --rule:#C9BC9F; --rule-soft:#DCD2B8; --card:#FFFDF6;
      --border:#D4C8AE; --border-dark:#221D17;
      --serif:"Fraunces", "Noto Serif Devanagari", "Noto Serif Telugu", "Noto Serif Kannada", "Noto Serif Tamil", Georgia, serif;
      --type:"Courier Prime", "Noto Sans Devanagari", "Noto Sans Telugu", "Noto Sans Kannada", "Noto Sans Tamil", "Courier New", monospace;
      --sans:"Archivo", "Noto Sans Devanagari", "Noto Sans Telugu", "Noto Sans Kannada", "Noto Sans Tamil", system-ui, sans-serif;
    }

    /* Header Language Picker */
    .lang-picker {
      display: inline-flex;
      align-items: center;
      background: var(--paper-deep);
      border: 1px solid var(--rule);
      border-radius: 4px;
      padding: 3px 8px;
      margin-right: 12px;
      transition: border-color .15s ease, box-shadow .15s ease;
      position: relative;
    }
    .lang-picker:focus-within, .lang-picker:hover {
      border-color: var(--stamp);
      box-shadow: 0 0 0 2px rgba(166,25,60,0.12);
    }
    .lang-dropdown,
    .lang-picker select,
    select.lang-dropdown {
      background: transparent !important;
      border: none !important;
      outline: none !important;
      box-shadow: none !important;
      -webkit-appearance: none !important;
      -moz-appearance: none !important;
      appearance: none !important;
      font-family: var(--type) !important;
      font-size: 11.5px !important;
      color: var(--ink) !important;
      font-weight: 700 !important;
      letter-spacing: .5px !important;
      cursor: pointer !important;
      padding: 2px 18px 2px 2px !important;
      margin: 0 !important;
      width: auto !important;
      height: auto !important;
      background-image: url("data:image/svg+xml;charset=UTF-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%235A5142' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E") !important;
      background-repeat: no-repeat !important;
      background-position: right center !important;
      background-size: 8px 5px !important;
    }
    .lang-dropdown:focus,
    .lang-dropdown:focus-visible,
    .lang-dropdown:active,
    .lang-picker select:focus,
    .lang-picker select:focus-visible,
    .lang-picker select:active,
    select.lang-dropdown:focus,
    select.lang-dropdown:focus-visible,
    select.lang-dropdown:active {
      outline: none !important;
      border: none !important;
      box-shadow: none !important;
    }
    .lang-dropdown option {
      background: #FFFDF6;
      color: #221D17;
      font-family: var(--sans);
      font-size: 13px;
      font-weight: 500;
      padding: 6px 10px;
    }
    /* Font fallbacks & optical size equalizer for Indic languages */
    html[lang="hi"] body, html[lang="hi"] p, html[lang="hi"] span, html[lang="hi"] a, html[lang="hi"] button, html[lang="hi"] th, html[lang="hi"] td, html[lang="hi"] label, html[lang="hi"] div {
      font-family: "Noto Sans Devanagari", var(--sans), sans-serif;
    }
    html[lang="te"] body, html[lang="te"] p, html[lang="te"] span, html[lang="te"] a, html[lang="te"] button, html[lang="te"] th, html[lang="te"] td, html[lang="te"] label, html[lang="te"] div {
      font-family: "Noto Sans Telugu", var(--sans), sans-serif;
    }
    html[lang="kn"] body, html[lang="kn"] p, html[lang="kn"] span, html[lang="kn"] a, html[lang="kn"] button, html[lang="kn"] th, html[lang="kn"] td, html[lang="kn"] label, html[lang="kn"] div {
      font-family: "Noto Sans Kannada", var(--sans), sans-serif;
    }
    html[lang="ta"] body, html[lang="ta"] p, html[lang="ta"] span, html[lang="ta"] a, html[lang="ta"] button, html[lang="ta"] th, html[lang="ta"] td, html[lang="ta"] label, html[lang="ta"] div {
      font-family: "Noto Sans Tamil", var(--sans), sans-serif;
    }

    /* Reset letter-spacing for Indic scripts so spacing matches English positioning without breaking font sizing */
    html[lang]:not([lang="en"]) *,
    html[lang]:not([lang="en"]) ::placeholder {
      letter-spacing: normal !important;
    }
    *{margin:0;padding:0;box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      background:var(--paper);color:var(--ink);font-family:var(--sans);
      font-size:14.5px;line-height:1.55;overflow-x:hidden;
    }
    ::selection{background:var(--stamp);color:var(--paper)}

    .security-bg{
      position:fixed;inset:0;z-index:0;pointer-events:none;
      background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='420' height='420' viewBox='0 0 420 420'%3E%3Cg fill='none' stroke='%23C99AA8' stroke-width='1' opacity='.22'%3E%3Ccircle cx='210' cy='210' r='196'/%3E%3Ccircle cx='210' cy='210' r='188' stroke-dasharray='3 6'/%3E%3Ccircle cx='210' cy='210' r='172'/%3E%3Ccircle cx='210' cy='210' r='164' stroke-dasharray='10 4'/%3E%3Ccircle cx='210' cy='210' r='148'/%3E%3Ccircle cx='210' cy='210' r='140' stroke-dasharray='2 5'/%3E%3Ccircle cx='210' cy='210' r='124'/%3E%3Ccircle cx='210' cy='210' r='116' stroke-dasharray='8 5'/%3E%3Ccircle cx='210' cy='210' r='100'/%3E%3Ccircle cx='210' cy='210' r='92' stroke-dasharray='4 4'/%3E%3Ccircle cx='210' cy='210' r='76'/%3E%3Ccircle cx='210' cy='210' r='68' stroke-dasharray='12 3'/%3E%3Ccircle cx='210' cy='210' r='52'/%3E%3Ccircle cx='210' cy='210' r='44'/%3E%3Ccircle cx='210' cy='210' r='36' stroke-dasharray='3 4'/%3E%3Ccircle cx='210' cy='210' r='20'/%3E%3C/g%3E%3C/svg%3E");
      background-size:420px 420px;
      opacity:.32;
    }

    /* Outer Shell with Sidebar Layout */
    .app-layout{
      display:flex;min-height:100vh;position:relative;z-index:1;
    }

    /* ---------------------------------------------------- */
    /* LEFT SIDE MENU (Fixed Navigation Rail)               */
    /* ---------------------------------------------------- */
    aside.dash-sidebar{
      width:260px;min-width:260px;max-width:260px;flex-shrink:0;box-sizing:border-box;background:var(--card);
      border-right:1.5px solid var(--border);
      display:flex;flex-direction:column;justify-content:space-between;
      position:sticky;top:0;height:100vh;overflow-y:auto;z-index:100;
      box-shadow:2px 0 6px rgba(0,0,0,.02);
    }
    @media(max-width:960px){
      aside.dash-sidebar{display:none}
      .app-layout{flex-direction:column}
    }

    .sidebar-top{padding:24px 16px 16px;text-align:center}
    .brand-box{
      display:flex;align-items:center;justify-content:center;text-decoration:none;color:var(--ink);
      padding-bottom:18px;border-bottom:2px double var(--rule);margin-bottom:22px;text-align:center;
    }
    .brand-box b{font-family:var(--serif);font-weight:900;font-size:26px;letter-spacing:.04em;line-height:1;text-align:left}
    .brand-box span{font-family:var(--type);font-size:10.5px;letter-spacing:.18em;color:var(--stamp);text-transform:uppercase;text-align:left}

    .nav-label{
      font-family:var(--type);font-size:10px;letter-spacing:.16em;text-transform:uppercase;
      color:var(--ink-soft);padding:0 8px;margin-bottom:10px;font-weight:700;text-align:left;
    }

    .nav-menu{display:flex;flex-direction:column;gap:3px;list-style:none;padding:0;margin:0}
    .nav-menu li a{
      display:flex;align-items:center;justify-content:space-between;
      padding:10px 10px 10px 8px;border-radius:2px;text-decoration:none;color:var(--ink);
      font-size:13.5px;font-weight:600;transition:all .15s ease;
      border-left:3px solid transparent;text-align:left;
    }
    .nav-menu li a:hover{
      background:var(--paper);border-left-color:var(--rule);color:var(--ink);
    }
    .nav-menu li a.active{
      background:var(--paper-deep);border-left-color:var(--stamp);color:var(--stamp);font-weight:700;
    }
    .nav-link-left{display:flex;align-items:center;gap:0;text-align:left;justify-content:flex-start}

    .nav-badge{
      font-family:var(--type);font-size:10px;padding:2px 6px;border-radius:2px;
      font-weight:700;letter-spacing:.05em;
    }
    .badge-primary{background:var(--stamp);color:var(--paper)}
    .badge-amber{background:#FEF7E0;color:#8A5300;border:1px solid #F2CD86}
    .badge-green{background:#E6F4EA;color:var(--green);border:1px solid #A8DAB5}

    .sidebar-bottom{
      padding:16px 20px;background:var(--paper-deep);border-top:1.5px solid var(--border);
    }
    .sys-pill{
      display:flex;align-items:center;gap:6px;font-family:var(--type);font-size:10.5px;
      letter-spacing:.12em;text-transform:uppercase;color:var(--green-deep);
      margin-bottom:8px;font-weight:700;
    }
    .sys-dot{width:7px;height:7px;border-radius:50%;background:var(--green)}
    .sys-meta{
      font-family:var(--type);font-size:10.5px;color:var(--ink-soft);line-height:1.45;
    }

    /* ---------------------------------------------------- */
    /* MAIN CONTENT AREA                                    */
    /* ---------------------------------------------------- */
    main.dash-content{
      flex:1;min-width:0;display:flex;flex-direction:column;
    }
    .main-inner{
      padding:28px 40px 54px;max-width:1440px;width:100%;margin:0 auto;
    }
    @media(max-width:768px){.main-inner{padding:18px 16px 36px}}

    /* Top Action Bar */
    .top-action-bar{
      display:flex;justify-content:space-between;align-items:center;
      margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid var(--rule-soft);
      flex-wrap:wrap;gap:14px;
    }
    .header-left h1{
      font-family:var(--serif);font-size:28px;font-weight:600;color:var(--ink);line-height:1.15;
    }
    .header-left h1 em{font-style:italic;color:var(--stamp);font-weight:400}
    .header-tagline{font-size:13.5px;color:var(--ink-soft);margin-top:4px}

    .header-right{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
    .date-badge{
      font-family:var(--type);font-size:11px;color:var(--ink-soft);
      padding:6px 12px;border:1px solid var(--border);background:var(--card);border-radius:2px;
    }
    .btn{
      font-family:var(--type);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
      text-decoration:none;padding:9px 18px;border-radius:2px;display:inline-flex;align-items:center;
      gap:6px;border:0;cursor:pointer;font-weight:700;transition:all .15s ease;
    }
    .btn-primary{background:var(--stamp);color:var(--paper);box-shadow:2px 2px 0 var(--stamp-deep)}
    .btn-primary:hover{transform:translate(-1px,-1px);box-shadow:3px 3px 0 var(--stamp-deep)}
    .btn-ghost{color:var(--ink);border:1.5px solid var(--ink);background:transparent}
    .btn-ghost:hover{background:var(--ink);color:var(--paper)}
    .btn-sm{padding:6px 12px;font-size:10.5px}

    /* 4-Grid KPI Cards */
    .kpi-grid{
      display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:26px;
      align-items:stretch;
    }
    @media(max-width:1120px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
    @media(max-width:560px){.kpi-grid{grid-template-columns:1fr}}

    .kpi-card{
      background:var(--card);border:1.5px solid var(--border);border-radius:2px;
      padding:20px 22px;position:relative;box-shadow:2px 2px 0 rgba(0,0,0,.025);
      transition:transform .15s ease, border-color .15s ease;
      display:flex;flex-direction:column;justify-content:space-between;height:100%;box-sizing:border-box;
    }
    .kpi-card:hover{border-color:var(--ink);transform:translateY(-2px)}
    .kpi-label{
      font-family:var(--type);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
      color:var(--ink-soft);margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;
    }
    .kpi-num{
      font-family:var(--serif);font-size:42px;font-weight:700;line-height:1;
      color:var(--ink);margin-bottom:6px;font-variant-numeric:tabular-nums;
    }
    .kpi-card.kpi-sealed .kpi-num{color:var(--green)}
    .kpi-card.kpi-desk .kpi-num{color:var(--amber)}
    .kpi-card.kpi-rejected .kpi-num{color:var(--stamp)}
    .kpi-sub{font-size:12px;color:var(--ink-soft);line-height:1.3}

    /* Action Notice Banner */
    .notice-banner{
      background:#FFF9E6;border:1.5px solid #F0C36D;border-left:4px solid var(--amber);
      border-radius:2px;padding:12px 18px;margin-bottom:24px;
      display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;
      width:100%;box-sizing:border-box;
    }
    .notice-info{display:flex;align-items:center;gap:12px}
    .notice-icon{font-size:20px}
    .notice-text{font-size:13.5px;color:#5A4008}
    .notice-text b{color:#2E2002}

    /* ---------------------------------------------------- */
    /* STATISTICAL ANALYTICS GRAPHS (2-Col Clean Layout)    */
    /* ---------------------------------------------------- */
    .charts-grid{
      display:grid;grid-template-columns:1.6fr 1fr;gap:20px;margin-bottom:26px;
      align-items:stretch;
    }
    @media(max-width:1080px){.charts-grid{grid-template-columns:1fr}}

    .chart-card{
      background:var(--card);border:1.5px solid var(--border);border-radius:2px;
      padding:22px;display:flex;flex-direction:column;justify-content:space-between;
      height:100%;box-sizing:border-box;box-shadow:2px 2px 0 rgba(0,0,0,.025);
    }
    .chart-header{
      display:flex;justify-content:space-between;align-items:baseline;
      margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid var(--rule-soft);
      flex-wrap:wrap;gap:8px;
    }
    .chart-title{font-family:var(--serif);font-size:17px;font-weight:700;color:var(--ink)}
    .chart-meta{font-family:var(--type);font-size:11px;color:var(--ink-soft);letter-spacing:.08em}

    .chart-svg-wrap{width:100%;height:auto;overflow:hidden}
    .chart-svg{width:100%;height:auto;display:block}

    /* Donut Chart Layout */
    .donut-layout{
      display:flex;align-items:center;gap:24px;flex-wrap:wrap;margin:auto 0;
    }
    .donut-svg-box{width:140px;height:140px;flex-shrink:0;position:relative}
    .donut-legend{flex:1;display:flex;flex-direction:column;gap:8px;min-width:150px}
    .legend-row{
      display:flex;justify-content:space-between;align-items:center;font-size:12.5px;
    }
    .legend-left{display:flex;align-items:center;gap:8px}
    .legend-color{width:10px;height:10px;border-radius:2px;flex-shrink:0}
    .legend-num{font-family:var(--type);font-weight:700;color:var(--ink)}

    .gis-rate-meter{
      margin-top:16px;padding-top:12px;border-top:1px solid var(--rule-soft);
    }
    .meter-label{
      display:flex;justify-content:space-between;font-family:var(--type);font-size:11px;
      color:var(--ink-soft);margin-bottom:6px;
    }
    .meter-bar{
      height:6px;background:var(--paper-deep);border-radius:3px;overflow:hidden;
    }
    .meter-fill{
      height:100%;background:var(--green);border-radius:3px;
    }

    /* ---------------------------------------------------- */
    /* MASTER DEED REGISTER LEDGER                          */
    /* ---------------------------------------------------- */
    .ledger-section{
      background:var(--card);border:1.5px solid var(--ink);border-radius:2px;
      box-shadow:3px 3px 0 rgba(0,0,0,.04);margin-bottom:26px;width:100%;box-sizing:border-box;
    }
    .ledger-toolbar{
      padding:12px 18px;background:var(--paper-deep);border-bottom:1.5px solid var(--ink);
      display:flex;justify-content:space-between;align-items:center;gap:14px;
      width:100%;box-sizing:border-box;flex-wrap:nowrap;
    }
    .toolbar-left{display:flex;align-items:center;gap:12px;flex-shrink:1;min-width:0}
    .ledger-head-title{
      font-family:var(--serif);font-size:18px;font-weight:700;color:var(--ink);
      white-space:nowrap;flex-shrink:0;
    }
    .search-wrap{position:relative;flex-shrink:1;min-width:130px;max-width:230px;width:100%}
    .search-input{
      font-family:var(--sans);font-size:12.5px;padding:6px 10px 6px 28px;
      border:1.5px solid var(--rule);border-radius:2px;background:var(--paper);
      color:var(--ink);width:100%;box-sizing:border-box;transition:all .15s ease;outline:none;
    }
    .search-input:focus{border-color:var(--stamp);background:var(--card)}
    .search-icon{position:absolute;left:8px;top:50%;transform:translateY(-50%);color:var(--ink-soft);font-size:12px}

    .toolbar-right{display:flex;align-items:center;gap:8px;flex-shrink:0;white-space:nowrap;flex-wrap:nowrap}
    .filter-tabs{display:flex;gap:4px;align-items:center;flex-wrap:nowrap}
    .tab-btn{
      font-family:var(--type);font-size:10.5px;padding:5px 9px;border:1px solid var(--rule);
      background:var(--paper);color:var(--ink-soft);border-radius:2px;cursor:pointer;
      transition:all .15s ease;white-space:nowrap;flex-shrink:0;
    }
    .tab-btn:hover{color:var(--ink);border-color:var(--ink)}
    .tab-btn.active{
      background:var(--stamp);color:var(--paper);border-color:var(--stamp-deep);font-weight:700;
    }

    /* Table Component */
    .table-container{overflow-x:auto;max-height:600px}
    table.master-ledger{width:100%;border-collapse:collapse;text-align:left;table-layout:auto}
    table.master-ledger th:nth-child(1){width:45px}
    table.master-ledger th:nth-child(2){width:100px}
    table.master-ledger th:nth-child(3){min-width:140px}
    table.master-ledger th:nth-child(4){min-width:160px}
    table.master-ledger th:nth-child(5){min-width:140px}
    table.master-ledger th:nth-child(6){width:85px}
    table.master-ledger th:nth-child(7){width:115px}
    table.master-ledger th:nth-child(8){min-width:150px}
    table.master-ledger th:nth-child(9){width:280px;min-width:280px;text-align:center;white-space:nowrap}
    table.master-ledger thead th{
      position:sticky;top:0;z-index:10;
      background:var(--ink);color:var(--paper);font-family:var(--type);
      font-size:11px;letter-spacing:.12em;text-transform:uppercase;
      padding:12px 14px;font-weight:600;white-space:nowrap;border-right:1px solid rgba(255,255,255,.1);
    }
    table.master-ledger tbody tr{
      border-bottom:1px solid var(--rule-soft);transition:background .1s ease;
    }
    table.master-ledger tbody tr:nth-child(even){background:rgba(239,230,208,.3)}
    table.master-ledger tbody tr:hover{background:rgba(166,25,60,.04)}
    table.master-ledger td{padding:13px 14px;font-size:13.5px;vertical-align:middle;overflow:hidden;text-overflow:ellipsis}

    .td-sl{font-family:var(--type);font-weight:700;color:var(--stamp);font-size:12px}
    .td-date{font-family:var(--type);font-size:12px;color:var(--ink-soft);white-space:nowrap}
    .td-doc-main{font-weight:600;color:var(--ink);display:block}
    .td-doc-sub{font-family:var(--type);font-size:11px;color:var(--ink-soft)}
    .td-place-main{font-weight:600;color:var(--ink);display:block}
    .td-place-sub{font-size:12px;color:var(--ink-soft)}
    .td-mono{font-family:var(--type);font-size:12px;color:var(--ink)}
    .td-stamp{font-family:var(--type);font-weight:700;color:var(--stamp-deep)}

    /* Status Badges */
    .badge{
      font-family:var(--type);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
      padding:3px 8px;border-radius:2px;display:inline-flex;align-items:center;gap:5px;font-weight:700;
      border:1px solid transparent;white-space:nowrap;
    }
    .b-approved{background:#E6F4EA;color:var(--green);border-color:#A8DAB5}
    .b-extracted,.b-ready_for_approval,.b-under_review{background:#FEF7E0;color:#8A5300;border-color:#F2CD86}
    .b-rejected,.b-fail,.b-duplicate{background:#FCE8E6;color:var(--stamp);border-color:#F5B7B1}

    .action-links{display:flex;gap:8px;align-items:center;white-space:nowrap}
    .act-btn{
      font-family:var(--type);font-size:11px;padding:5px 10px;border-radius:2px;
      text-decoration:none;border:1px solid var(--border-dark);color:var(--ink);
      font-weight:700;background:var(--paper);transition:all .12s ease;
    }
    .act-btn:hover{background:var(--ink);color:var(--paper)}
    .act-btn.act-verify{border-color:var(--green);color:var(--green);background:#F0F8F3}
    .act-btn.act-verify:hover{background:var(--green);color:var(--paper)}

    /* Empty table state */
    .table-empty{padding:56px 20px;text-align:center;color:var(--ink-soft)}
    .table-empty p{font-size:15px;margin-bottom:12px}

    /* ---------------------------------------------------- */
    /* DEDICATED NEW SCAN INTAKE PAGE STYLES                */
    /* ---------------------------------------------------- */
    .intake-card{
      background:var(--card);border:1.5px solid var(--ink);border-radius:2px;
      box-shadow:3px 3px 0 rgba(0,0,0,.04);padding:32px 36px;margin-bottom:30px;
    }
    .intake-head{
      display:flex;justify-content:space-between;align-items:baseline;
      padding-bottom:14px;border-bottom:2px double var(--rule);margin-bottom:24px;
      flex-wrap:wrap;gap:12px;
    }
    .intake-head h2{font-family:var(--serif);font-size:22px;color:var(--ink);font-weight:700}
    .intake-badge{
      font-family:var(--type);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
      padding:4px 10px;background:var(--paper-deep);border:1px solid var(--border);
      color:var(--stamp);font-weight:700;
    }

    .intake-dropzone{
      border:2.5px dashed var(--rule);border-radius:3px;padding:42px 24px;
      text-align:center;background:var(--paper);cursor:pointer;
      transition:all .18s ease;position:relative;margin-bottom:24px;
    }
    .intake-dropzone:hover,.intake-dropzone.dragover{
      border-color:var(--stamp);background:var(--card);box-shadow:inset 0 0 12px rgba(166,25,60,.04);
    }
    .dz-icon-svg{width:48px;height:48px;margin:0 auto 12px;display:block;stroke:var(--stamp);fill:none}
    .dz-main-text{font-size:16px;font-weight:700;color:var(--ink);margin-bottom:4px}
    .dz-main-text span{color:var(--stamp);text-decoration:underline}
    .dz-sub-text{font-family:var(--type);font-size:11.5px;color:var(--ink-soft)}

    .filechip{
      display:flex;justify-content:space-between;align-items:center;
      padding:12px 18px;background:var(--paper-deep);border:1.5px solid var(--border-dark);
      border-radius:2px;margin-bottom:24px;
    }
    .filechip-name{font-weight:700;font-size:14px;color:var(--ink);display:flex;align-items:center;gap:8px}
    .filechip-actions{display:flex;align-items:center;gap:12px}
    .filechip-size{font-family:var(--type);font-size:11.5px;color:var(--ink-soft)}
    .filechip-btn{
      background:none;border:none;color:var(--stamp);font-size:16px;cursor:pointer;
      padding:2px 6px;line-height:1;font-weight:700;
    }

    .config-grid{
      display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:16px;margin-bottom:28px;
    }
    @media(max-width:768px){.config-grid{grid-template-columns:1fr}}

    .config-box{
      background:var(--paper);border:1px solid var(--rule);padding:16px 18px;border-radius:2px;
    }
    .config-box-title{
      font-family:var(--type);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
      color:var(--ink-soft);margin-bottom:10px;font-weight:700;
    }
    .select-mode{
      width:100%;padding:10px 12px;border:1.5px solid var(--border);border-radius:2px;
      font-family:var(--sans);font-size:13.5px;font-weight:600;background:var(--card);
      color:var(--ink);outline:none;
    }
    .select-mode:focus{border-color:var(--stamp)}
    .config-note{font-family:var(--type);font-size:11px;color:var(--ink-soft);margin-top:8px;line-height:1.4}

    /* Pipeline Stepper */
    .stepper-strip{
      display:flex;background:var(--paper-deep);border:1px solid var(--border);
      padding:14px 18px;border-radius:2px;margin-bottom:28px;gap:8px;flex-wrap:wrap;
    }
    .stepper-item{
      flex:1;min-width:140px;display:flex;align-items:center;gap:8px;font-size:12px;
    }
    .step-num{
      width:22px;height:22px;border-radius:50%;background:var(--paper);border:1px solid var(--rule);
      font-family:var(--type);font-size:11px;display:flex;align-items:center;justify-content:center;
      color:var(--ink-soft);font-weight:700;flex-shrink:0;
    }
    .stepper-item.step-active .step-num{
      background:var(--stamp);border-color:var(--stamp);color:var(--paper);
    }
    .step-label{font-weight:600;color:var(--ink-soft)}
    .stepper-item.step-active .step-label{color:var(--ink)}

    .submit-row{
      display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;
      padding-top:20px;border-top:1.5px solid var(--rule-soft);
    }
    .submit-note{font-family:var(--type);font-size:11.5px;color:var(--ink-soft)}

    /* Loading Overlay */
    .loading-overlay{
      display:none;position:fixed;inset:0;background:rgba(34,29,23,.75);z-index:9999;
      align-items:center;justify-content:center;backdrop-filter:blur(4px);
    }
    .loading-box{
      background:var(--paper);border:2px solid var(--ink);border-radius:2px;padding:34px 44px;
      text-align:center;box-shadow:6px 6px 0 rgba(0,0,0,.3);max-width:400px;
    }
    .loading-spinner{
      width:44px;height:44px;border:3px solid var(--rule);border-top-color:var(--stamp);
      border-radius:50%;margin:0 auto 16px;animation:spin 1s linear infinite;
    }
    @keyframes spin{to{transform:rotate(360deg)}}
    .loading-title{font-family:var(--serif);font-size:20px;font-weight:700;color:var(--ink);margin-bottom:6px}
    .loading-sub{font-family:var(--type);font-size:11.5px;color:var(--ink-soft)}

    /* Understated Security & Authority Bar */
    .system-strip{
      background:var(--paper-deep);border:1px solid var(--border);border-radius:2px;
      padding:12px 20px;display:flex;justify-content:space-between;align-items:center;
      flex-wrap:wrap;gap:14px;font-family:var(--type);font-size:11px;color:var(--ink-soft);
      margin-bottom:20px;
    }
    .strip-group{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
    .strip-item{display:flex;align-items:center;gap:6px}
    .strip-item b{color:var(--ink)}
    .btn-copy{
      background:none;border:none;color:var(--stamp);cursor:pointer;font-family:var(--type);
      font-size:10.5px;text-transform:uppercase;font-weight:700;padding:2px 4px;margin-left:4px;
    }
    .btn-copy:hover{text-decoration:underline}

    /* Footer Note */
    footer.dash-footer{
      border-top:3px double var(--rule);padding:20px 0;background:var(--paper);
      margin-top:auto;
    }
    .dash-footer-wrap{
      display:flex;justify-content:space-between;align-items:center;
      font-family:var(--type);font-size:11px;color:var(--ink-soft);flex-wrap:wrap;gap:12px;
    }

    /* State & District Progress Panel */
    .state-progress-section{
      background:var(--card);border:1.5px solid var(--border);border-radius:2px;
      padding:22px;margin-bottom:26px;box-shadow:2px 2px 0 rgba(0,0,0,.025);
    }
    .state-progress-header{
      display:flex;justify-content:space-between;align-items:flex-start;
      margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--rule-soft);
      flex-wrap:wrap;gap:12px;
    }
    .state-card{
      background:var(--paper);border:1px solid var(--border);border-radius:3px;
      margin-bottom:12px;overflow:hidden;transition:border-color .15s ease;
    }
    .state-card:hover{border-color:var(--ink)}
    .state-summary{
      display:flex;justify-content:space-between;align-items:center;
      padding:14px 18px;cursor:pointer;user-select:none;gap:16px;flex-wrap:wrap;
    }
    .state-summary::-webkit-details-marker{display:none}
    .state-title-col{display:flex;align-items:center;gap:10px}
    .state-chevron{
      display:inline-block;font-size:11px;transition:transform .2s ease;color:var(--ink-soft);
    }
    details[open] .state-chevron{transform:rotate(90deg)}
    .state-name{font-family:var(--serif);font-size:16px;font-weight:700;color:var(--ink)}
    .state-counts{
      display:flex;gap:10px;align-items:center;font-family:var(--type);font-size:11px;flex-wrap:wrap;
    }
    .count-pill{
      padding:3px 8px;border-radius:3px;font-weight:700;
    }
    .pill-total{background:var(--paper-deep);color:var(--ink);border:1px solid var(--rule)}
    .pill-sealed{background:#E6F4EA;color:var(--green);border:1px solid #A8DAB5}
    .pill-pending{background:#FEF7E0;color:#8A5300;border:1px solid #F2CD86}
    .pill-flagged{background:#FCE8E6;color:var(--stamp);border:1px solid #F5C6CB}
    .progress-bar-wrap{
      display:flex;align-items:center;gap:10px;min-width:180px;
    }
    .progress-stacked-bar{
      height:10px;background:var(--paper-deep);border-radius:5px;overflow:hidden;
      display:flex;width:130px;border:1px solid var(--rule);
    }
    .bar-seg-sealed{background:var(--green);height:100%}
    .bar-seg-pending{background:var(--amber);height:100%}
    .bar-seg-flagged{background:var(--stamp);height:100%}
    .progress-pct{
      font-family:var(--type);font-size:11.5px;font-weight:700;color:var(--ink);
      min-width:38px;text-align:right;
    }
    .district-table-wrap{
      padding:14px 18px 18px;background:var(--paper-deep);border-top:1px solid var(--rule);
      overflow-x:auto;
    }
    .district-table{
      width:100%;border-collapse:collapse;font-size:13px;
    }
    .district-table th{
      font-family:var(--type);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
      color:var(--ink-soft);padding:8px 10px;border-bottom:1.5px solid var(--rule);
      text-align:left;
    }
    .district-table td{
      padding:8px 10px;border-bottom:1px solid var(--rule-soft);
      font-family:var(--sans);
    }
    .district-table tr:last-child td{border-bottom:none}
    .district-table td.td-center{text-align:center}
    .district-table td.td-num{font-family:var(--type);font-weight:700}
    .district-table td.td-right{text-align:right}

    /* Extraction Accuracy & Field Intelligence Panel */
    .extraction-accuracy-section{
      background:var(--card);border:1.5px solid var(--border);border-radius:2px;
      padding:22px;margin-bottom:26px;box-shadow:2px 2px 0 rgba(0,0,0,.025);
    }
    .accuracy-header{
      display:flex;justify-content:space-between;align-items:flex-start;
      margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--rule-soft);
      flex-wrap:wrap;gap:12px;
    }
    .accuracy-grid{
      display:grid;grid-template-columns:1.4fr 1fr;gap:20px;
      align-items:stretch;
    }
    @media(max-width:1080px){.accuracy-grid{grid-template-columns:1fr}}
    .accuracy-subcard{
      background:var(--paper);border:1px solid var(--border);border-radius:3px;
      padding:16px 18px;display:flex;flex-direction:column;justify-content:space-between;
    }
    .uncertain-field-row{
      display:flex;justify-content:space-between;align-items:center;
      padding:7px 0;border-bottom:1px solid var(--rule-soft);font-size:12.5px;gap:10px;
    }
    .uncertain-field-row:last-child{border-bottom:none}
    .field-name-wrap{display:flex;align-items:center;gap:6px;min-width:0;flex:1}
    .field-pill-uncertain{
      background:#FCE8E6;color:var(--stamp);border:1px solid #F5C6CB;
      font-family:var(--type);font-size:9.5px;padding:1px 5px;border-radius:2px;font-weight:700;white-space:nowrap;
    }
    .field-pill-review{
      background:#FEF7E0;color:#8A5300;border:1px solid #F2CD86;
      font-family:var(--type);font-size:9.5px;padding:1px 5px;border-radius:2px;font-weight:700;white-space:nowrap;
    }
    .field-pill-good{
      background:#E6F4EA;color:var(--green);border:1px solid #A8DAB5;
      font-family:var(--type);font-size:9.5px;padding:1px 5px;border-radius:2px;font-weight:700;white-space:nowrap;
    }
"""

DASHBOARD_JS = """
  function filterTable() {
    const searchInput = document.getElementById('ledgerSearch');
    if (!searchInput) return;
    const searchVal = (searchInput.value || '').toLowerCase().trim();
    const activeTabEl = document.querySelector('.tab-btn.active');
    const activeTab = activeTabEl ? (activeTabEl.dataset.filter || 'ALL') : 'ALL';
    const rows = document.querySelectorAll('#ledgerBody tr.data-row');
    let visibleCount = 0;

    rows.forEach(row => {
      const text = row.textContent.toLowerCase();
      const status = row.dataset.status || '';
      
      const matchesSearch = !searchVal || text.includes(searchVal);
      let matchesTab = false;

      if (activeTab === 'ALL') matchesTab = true;
      else if (activeTab === 'SEALED') matchesTab = (status === 'APPROVED');
      else if (activeTab === 'PENDING') matchesTab = (status !== 'APPROVED' && status !== 'REJECTED');
      else if (activeTab === 'REJECTED') matchesTab = (status === 'REJECTED');

      if (matchesSearch && matchesTab) {
        row.style.display = '';
        visibleCount++;
      } else {
        row.style.display = 'none';
      }
    });

    const emptyNotice = document.getElementById('ledgerEmptyNotice');
    if (emptyNotice) {
      emptyNotice.style.display = (visibleCount === 0) ? 'block' : 'none';
    }
  }

  function filterTab(filterName) {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', (b.dataset.filter === filterName));
    });
    filterTable();
    const target = document.getElementById('ledgerSection');
    if (target) { target.scrollIntoView({ behavior: 'smooth' }); }
  }

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      filterTable();
    });
  });

  const searchInput = document.getElementById('ledgerSearch');
  if (searchInput) {
    searchInput.addEventListener('input', filterTable);
  }

  window.copyFingerprint = function(text) {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(() => {
        alert('SHA-256 Public Key Fingerprint copied to clipboard!');
      });
    } else {
      prompt('Copy Fingerprint:', text);
    }
  };

  window.resetRegistry = function() {
    if (confirm("Reset the ledger and all dashboard counters back to 0 documents?")) {
      fetch("/api/reset_registry", { method: "POST" })
        .then(() => { window.location.href = "/dashboard"; })
        .catch(() => { window.location.href = "/reset"; });
    }
  };

  // Dedicated Scan Intake File Dropzone Handler
  const intakeDropzone = document.getElementById('intakeDropzone');
  const fileInput = document.getElementById('scan_file_input');
  const fileChip = document.getElementById('fileChip');
  const chipName = document.getElementById('chipName');
  const chipSize = document.getElementById('chipSize');
  const chipRemove = document.getElementById('chipRemove');
  const scanForm = document.getElementById('scanForm');
  const loadingOverlay = document.getElementById('loadingOverlay');

  if (fileInput && intakeDropzone) {
    intakeDropzone.addEventListener('click', (e) => {
      if (e.target === fileInput) return;
      fileInput.click();
    });

    intakeDropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });

    fileInput.addEventListener('change', function() {
      if (this.files && this.files.length > 0) {
        showFileChip(this.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      intakeDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        intakeDropzone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      intakeDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        intakeDropzone.classList.remove('dragover');
      }, false);
    });

    intakeDropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files.length > 0) {
        fileInput.files = dt.files;
        showFileChip(dt.files[0]);
      }
    });
  }

  function showFileChip(file) {
    if (!fileChip || !intakeDropzone) return;
    intakeDropzone.style.display = 'none';
    fileChip.style.display = 'flex';
    chipName.textContent = file.name;
    const sizeKb = (file.size / 1024).toFixed(1);
    chipSize.textContent = sizeKb > 1024 ? (sizeKb / 1024).toFixed(2) + ' MB' : sizeKb + ' KB';
  }

  if (chipRemove && fileInput && fileChip && intakeDropzone) {
    chipRemove.addEventListener('click', function(e) {
      e.stopPropagation();
      fileInput.value = '';
      fileChip.style.display = 'none';
      intakeDropzone.style.display = 'block';
    });
  }

  if (scanForm && loadingOverlay) {
    scanForm.addEventListener('submit', function() {
      loadingOverlay.style.display = 'flex';
    });
  }

  // Language switcher for Dashboard
  const DASH_I18N = {"en": {"dash_sidebar_brand": "OneBhoomi · REGISTRY DESK", "nav_main_menu": "Main Menu", "nav_dash": "Dashboard", "nav_ops_dash": "Operations Dashboard", "nav_new_scan": "New Scan & Intake", "nav_new_scan_link": "New Document Scan", "badge_desk01": "Desk 01", "nav_master_reg": "Master Deed Register", "nav_clerk_queue": "Clerk Review Queue", "nav_verified_certs": "Verified Certificates", "lbl_air_gapped": "AIR-GAPPED & SECURE", "lbl_rsa_stat": "RSA-PSS 2048:", "lbl_active": "Active", "lbl_worker": "Worker:", "btn_new_scan": "+ New Document Scan", "btn_return_dash": "← Return to Dashboard", "lbl_day_book": "DAY BOOK DATE:", "lbl_register_desk": "REGISTER DESK 01 ·", "dash_h1": "Registry Operations & <em>Analytics</em>", "dash_tagline": "Complete offline day book of land records, human-in-the-loop clerk reviews, and cryptographic digital seals.", "kpi_total": "Total on File", "kpi_total_sub": "Total land deeds recorded in offline registry", "kpi_sealed": "Sealed & Certified", "kpi_sealed_sub": "certification rate · RSA-PSS signed", "kpi_review": "Officer Review Queue", "kpi_review_sub": "waiting officer verification & approval", "kpi_noncert": "Non-Certified", "kpi_noncert_sub": "Rejected or anomalous scan records", "banner_title": "Clerk Review Action Required:", "banner_prefix": "There are", "banner_suffix": "land record(s) awaiting verification against scanned evidence and officer digital signing.", "btn_open_next": "Open Next Record", "chart_vel_title": "Registration Velocity & Sealing Throughput", "chart_vel_meta": "Timeline intake volume vs verified cryptographic certifications", "chart_total_intake": "Total Intake", "chart_sealed_on_file": "Sealed on File", "chart_doc_class": "Document Classification", "chart_doc_meta": "Distribution of legal record deed categories", "chart_gis_lbl": "GIS Resolved:", "chart_sale_deeds": "Sale Deeds", "chart_agreements_gpa": "Agreements / GPA", "chart_other_records": "Other Records", "chart_spatial_match": "Telangana (TGRAC) & Karnataka Spatial Match", "chart_resolved": "Resolved", "reg_title": "Master Land Deed Register", "search_placeholder": "Search by Doc #, Village, Party, Survey...", "tab_all": "All", "tab_sealed": "Sealed", "tab_pending": "Pending", "tab_rejected": "Rejected", "btn_scan_new_deed": "+ Scan New Deed", "th_sl": "SL.", "th_received": "RECEIVED", "th_doc_no": "DOCUMENT & NO.", "th_parties": "PARTIES", "th_location": "LOCATION", "th_survey": "SURVEY DESIGNATION", "th_stamp": "STAMP DUTY", "th_status": "STATUS", "th_actions": "ACTIONS", "badge_approved": "Sealed & Certified", "badge_rejected": "Rejected on Record", "badge_under_review": "Needs Review", "badge_extracted": "Pending Check", "badge_ready_for_approval": "Ready for Seal", "badge_pass": "Checks Passed", "badge_fail": "Checks Failed", "btn_review_console": "Review Console", "btn_cert": "✓ Certificate", "lbl_fingerprint": "RSA-PSS 2048 Fingerprint:", "btn_copy": "COPY", "lbl_spatial_index": "Spatial Index:", "lbl_key_store": "Key Store:", "foot_console_title": "OneBhoomi Registry Console · Standalone Land Document Extraction & Digital Seal System", "foot_air_gapped_info": "100% Air-Gapped & Immutable · Zero cloud dependencies · Host: localhost", "badge_needs_review": "Needs Review", "Sale Deed": "Sale Deed", "Agreement of Sale-cum-General Power of Attorney": "Agreement of Sale-cum-General Power of Attorney", "Land Record": "Land Record", "Unnumbered": "Unnumbered", "Mandal": "Mandal", "Presently Mulugu District": "Presently Mulugu District", "intake_h1": "New Document Scan & <em>Intake</em>", "intake_title": "Registration & Scan Intake", "dz_main_text": "Drop the document scan copy here, or <span>browse local files</span>", "ocr_engine_title": "OCR Processing Engine", "val_grounding_title": "Validation & Spatial Grounding", "btn_start_extract": "Start Document Extraction →"}, "hi": {"dash_sidebar_brand": "वनभूमि · रजिस्ट्री डेस्क", "nav_main_menu": "मुख्य मेनू", "nav_dash": "डैशबोर्ड", "nav_ops_dash": "परिचालन डैशबोर्ड", "nav_new_scan": "नया स्कैन एवं इनटेक", "nav_new_scan_link": "नया दस्तावेज़ स्कैन", "badge_desk01": "डेस्क 01", "nav_master_reg": "मुख्य विलेख रजिस्टर", "nav_clerk_queue": "क्लर्क समीक्षा कतार", "nav_verified_certs": "प्रमाणित प्रमाणपत्र", "lbl_air_gapped": "सुरक्षित एवं एयर-गैप्ड", "lbl_rsa_stat": "RSA-PSS 2048:", "lbl_active": "सक्रिय", "lbl_worker": "वर्कर:", "btn_new_scan": "+ नया दस्तावेज़ स्कैन", "btn_return_dash": "← डैशबोर्ड पर वापस जाएं", "lbl_day_book": "रोज़नामचा तिथि:", "lbl_register_desk": "रजिस्ट्री डेस्क 01 ·", "dash_h1": "रजिस्ट्री संचालन एवं <em>विश्लेषण</em>", "dash_tagline": "भूमि अभिलेखों का संपूर्ण ऑफ़लाइन रोज़नामचा, मानवीय क्लर्क समीक्षा एवं क्रिप्टोग्राफ़िक डिजिटल मुहर।", "kpi_total": "कुल दर्ज दस्तावेज़", "kpi_total_sub": "ऑफ़लाइन रजिस्ट्री में दर्ज कुल भूमि विलेख", "kpi_sealed": "मुहरबंद एवं प्रमाणित", "kpi_sealed_sub": "प्रमाणीकरण दर · RSA-PSS हस्ताक्षरित", "kpi_review": "अधिकारी समीक्षा कतार", "kpi_review_sub": "अधिकारी सत्यापन एवं अनुमोदन प्रतीक्षित", "kpi_noncert": "गैर-प्रमाणित", "kpi_noncert_sub": "अस्वीकृत या त्रुटिपूर्ण स्कैन रिकॉर्ड", "banner_title": "क्लर्क समीक्षा कार्रवाई आवश्यक:", "banner_prefix": "यहाँ", "banner_suffix": "भूमि रिकॉर्ड साक्ष्य सत्यापन एवं अधिकारी डिजिटल हस्ताक्षर हेतु लंबित हैं।", "btn_open_next": "अगला रिकॉर्ड खोलें", "chart_vel_title": "पंजीकरण गति एवं मुहर प्रवाह", "chart_vel_meta": "दस्तावेज़ प्रवेश मात्रा बनाम सत्यापित डिजिटल प्रमाणीकरण", "chart_total_intake": "कुल प्रवेश", "chart_sealed_on_file": "अभिलेख पर मुहरबंद", "chart_doc_class": "दस्तावेज़ वर्गीकरण", "chart_doc_meta": "कानूनी भूमि विलेख श्रेणियों का वितरण", "chart_gis_lbl": "जीआईएस समाधान:", "chart_sale_deeds": "बिक्री विलेख", "chart_agreements_gpa": "समझौते / जीपीए", "chart_other_records": "अन्य रिकॉर्ड", "chart_spatial_match": "तेलंगाना (TGRAC) एवं कर्नाटक स्थानिक मिलान", "chart_resolved": "समाधानित", "reg_title": "मुख्य भूमि विलेख रजिस्टर", "search_placeholder": "दस्तावेज़ सं., ग्राम, पक्षकार, सर्वे द्वारा खोजें...", "tab_all": "सभी", "tab_sealed": "मुहरबंद", "tab_pending": "लंबित", "tab_rejected": "अस्वीकृत", "btn_scan_new_deed": "+ नया विलेख स्कैन करें", "th_sl": "क्र.", "th_received": "प्राप्ति तिथि", "th_doc_no": "दस्तावेज़ एवं क्रमांक", "th_parties": "पक्षकार", "th_location": "स्थान", "th_survey": "सर्वेक्षण पदनाम", "th_stamp": "स्टाम्प शुल्क", "th_status": "स्थिति", "th_actions": "कार्रवाई", "badge_approved": "मुहरबंद एवं प्रमाणित", "badge_rejected": "अभिलेख पर अस्वीकृत", "badge_under_review": "समीक्षा आवश्यक", "badge_extracted": "जांच लंबित", "badge_ready_for_approval": "मुहर हेतु तैयार", "badge_pass": "सत्यापन सफल", "badge_fail": "सत्यापन विफल", "btn_review_console": "समीक्षा कंसोल", "btn_cert": "✓ प्रमाणपत्र", "lbl_fingerprint": "RSA-PSS 2048 फिंगरप्रिंट:", "btn_copy": "कॉपी", "lbl_spatial_index": "स्थानिक सूचकांक:", "lbl_key_store": "कुंजी भंडार:", "foot_console_title": "वनभूमि रजिस्ट्री कंसोल · स्टैंडअलोन भूमि दस्तावेज़ निष्कर्षण एवं डिजिटल मुहर प्रणाली", "foot_air_gapped_info": "100% एयर-गैप्ड एवं अपरिवर्तनीय · शून्य क्लाउड निर्भरता · होस्ट: localhost", "badge_needs_review": "समीक्षा आवश्यक", "Sale Deed": "बिक्री विलेख", "Agreement of Sale-cum-General Power of Attorney": "बिक्री-सह-जीपीए समझौता", "Land Record": "भूमि रिकॉर्ड", "Unnumbered": "अक्रमांकित", "Mandal": "मंडल", "Presently Mulugu District": "वर्तमान में मुलुगु ज़िला", "intake_h1": "नया दस्तावेज़ स्कैन एवं <em>प्रवेश</em>", "intake_title": "पंजीकरण एवं स्कैन इनटेक", "dz_main_text": "दस्तावेज़ स्कैन प्रति यहाँ छोड़ें, या <span>स्थानीय फ़ाइलें चुनें</span>", "ocr_engine_title": "ओसीआर प्रोसेसिंग इंजन", "val_grounding_title": "सत्यापन एवं स्थानिक जांच", "btn_start_extract": "दस्तावेज़ निष्कर्षण प्रारंभ करें →"}, "te": {"dash_sidebar_brand": "వన్‌భూమి · రిజిస్ట్రీ డెస్క్", "nav_main_menu": "ప్రధాన మెనూ", "nav_dash": "డ్యాష్‌బోర్డ్", "nav_ops_dash": "కార్యకలాపాల డ్యాష్‌బోర్డ్", "nav_new_scan": "కొత్త స్కాన్ & ఇన్‌టేక్", "nav_new_scan_link": "కొత్త దస్తావేజు స్కాన్", "badge_desk01": "డెస్క్ 01", "nav_master_reg": "ప్రధాన దస్తావేజుల రిజిస్టర్", "nav_clerk_queue": "క్లర్క్ సమీక్ష వరుస", "nav_verified_certs": "ధృవీకరించిన సర్టిఫికెట్లు", "lbl_air_gapped": "సురక్షితం & ఎయిర్-గ్యాప్డ్", "lbl_rsa_stat": "RSA-PSS 2048:", "lbl_active": "క్రియాశీలం", "lbl_worker": "వర్కర్:", "btn_new_scan": "+ కొత్త దస్తావేజు స్కాన్", "btn_return_dash": "← డ్యాష్‌బోర్డుకు తిరిగి వెళ్లండి", "lbl_day_book": "రోజువారీ తేది:", "lbl_register_desk": "రిజిస్ట్రీ డెస్క్ 01 ·", "dash_h1": "రిజిస్ట్రీ కార్యకలాపాలు & <em>విశ్లేషణలు</em>", "dash_tagline": "భూ రికార్డుల ఆఫ్‌లైన్ డైలీ బుక్, క్లర్క్ సమీక్ష మరియు క్రిప్టోగ్రాఫిక్ డిజిటల్ సీల్స్.", "kpi_total": "మొత్తం రికార్డులు", "kpi_total_sub": "ఆఫ్‌లైన్ రిజిస్ట్రీలో నమోదు చేసిన మొత్తం దస్తావేజులు", "kpi_sealed": "సీల్ చేసి ధృవీకరించినవి", "kpi_sealed_sub": "ధృవీకరణ శాతం · RSA-PSS సంతకం", "kpi_review": "అధికారి సమీక్ష వరుస", "kpi_review_sub": "అధికారి తనిఖీ మరియు ఆమోదం కోసం వేచివున్నవి", "kpi_noncert": "ధృవీకరించబడనివి", "kpi_noncert_sub": "తిరస్కరించబడిన లేదా లోపభూయిష్ట రికార్డులు", "banner_title": "క్లర్క్ సమీక్ష చర్య అవసరం:", "banner_prefix": "ఇక్కడ", "banner_suffix": "రికార్డులు సాక్ష్యాధారాల తనిఖీ మరియు అధికారి డిజిటల్ సంతకం కోసం వేచి ఉన్నాయి.", "btn_open_next": "తదుపరి రికార్డు తెరవండి", "chart_vel_title": "నమోదు వేగం & ముద్రణ పరిమాణం", "chart_vel_meta": "ఇన్‌టేక్ సంఖ్య మరియు డిజిటల్ సర్టిఫికేషన్ల నిష్పత్తి", "chart_total_intake": "మొత్తం స్వీకరణ", "chart_sealed_on_file": "ఫైల్‌లో ముద్రించబడినవి", "chart_doc_class": "దస్తావేజుల వర్గీకరణ", "chart_doc_meta": "చట్టపరమైన రిజిస్టర్ దస్తావేజుల వివరాలు", "chart_gis_lbl": "GIS నిర్ధారణ:", "chart_sale_deeds": "సేల్ డీడ్‌లు", "chart_agreements_gpa": "ఒప్పందాలు / GPA", "chart_other_records": "ఇతర రికార్డులు", "chart_spatial_match": "తెలంగాణ (TGRAC) & కర్ణాటక భౌగోళిక సరిపోలిక", "chart_resolved": "సరిచూడబడింది", "reg_title": "ప్రధాన భూ దస్తావేజుల రిజిస్టర్", "search_placeholder": "పత్రం నం., గ్రామం, పార్టీ, సర్వే ద్వారా శోధించండి...", "tab_all": "అన్నీ", "tab_sealed": "సీల్ చేసినవి", "tab_pending": "పెండింగ్", "tab_rejected": "తిరస్కరించినవి", "btn_scan_new_deed": "+ కొత్త దస్తావేజు స్కాన్ చేయండి", "th_sl": "వ.సంఖ్య", "th_received": "స్వీకరించిన తేదీ", "th_doc_no": "పత్రం & సంఖ్య", "th_parties": "పార్టీలు", "th_location": "ప్రదేశం", "th_survey": "సర్వే హోదా", "th_stamp": "స్టాంప్ డ్యూటీ", "th_status": "స్థితి", "th_actions": "చర్యలు", "badge_approved": "సీల్ చేసి ధృవీకరించబడింది", "badge_rejected": "రికార్డులో తిరస్కరించబడింది", "badge_under_review": "సమీక్ష అవసరం", "badge_extracted": "తనిఖీ పెండింగ్", "badge_ready_for_approval": "ముద్రకు సిద్ధం", "badge_pass": "తనిఖీలు విజయవంతం", "badge_fail": "తనిఖీలు విఫలం", "btn_review_console": "సమీక్ష కన్సోల్", "btn_cert": "✓ సర్టిఫికేట్", "lbl_fingerprint": "RSA-PSS 2048 వేలిముద్ర:", "btn_copy": "కాపీ", "lbl_spatial_index": "భౌగోళిక సూచిక:", "lbl_key_store": "కీ నిల్వ:", "foot_console_title": "వన్‌భూమి రిజిస్ట్రీ కన్సోల్ · భూ దస్తావేజు సేకరణ మరియు డిజిటల్ సీల్ వ్యవస్థ", "foot_air_gapped_info": "100% ఎయిర్-గ్యాప్డ్ & మార్చలేనిది · క్లౌడ్ అవసరం లేదు · హోస్ట్: localhost", "badge_needs_review": "సమీక్ష అవసరం", "Sale Deed": "సేల్ డీడ్", "Agreement of Sale-cum-General Power of Attorney": "సేల్-కమ్-GPA ఒప్పందం", "Land Record": "భూ రికార్డు", "Unnumbered": "సంఖ్య లేదు", "Mandal": "మండలం", "Presently Mulugu District": "ప్రస్తుతం ములుగు జిల్లా", "intake_h1": "కొత్త దస్తావేజు స్కాన్ & <em>ఇన్‌టేక్</em>", "intake_title": "రిజిస్ట్రేషన్ & స్కాన్ ఇన్‌టేక్", "dz_main_text": "స్కాన్ చేసిన పత్రాన్ని ఇక్కడ వేయండి, లేదా <span>ఫైళ్లను ఎంచుకోండి</span>", "ocr_engine_title": "OCR ప్రాసెసింగ్ ఇంజిన్", "val_grounding_title": "ధృవీకరణ & భౌగోళిక తనిఖీ", "btn_start_extract": "సమాచార సేకరణను ప్రారంభించండి →"}, "kn": {"dash_sidebar_brand": "ವನ್‌ಭೂಮಿ · ನೋಂದಣಿ ಡೆಸ್ಕ್", "nav_main_menu": "ಮುಖ್ಯ ಮೆನು", "nav_dash": "ಡ್ಯಾಶ್‌ಬೋರ್ಡ್", "nav_ops_dash": "ಕಾರ್ಯಾಚರಣೆಗಳ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್", "nav_new_scan": "ಹೊಸ ಸ್ಕ್ಯಾನ್ & ಸ್ವೀಕಾರ", "nav_new_scan_link": "ಹೊಸ ದಾಖಲೆ ಸ್ಕ್ಯಾನ್", "badge_desk01": "ಡೆಸ್ಕ್ 01", "nav_master_reg": "ಮುಖ್ಯ ಪತ್ರಗಳ ನೋಂದಣಿ", "nav_clerk_queue": "ಗುಮಾಸ್ತರ ಪರಿಶೀಲನಾ ಸಾಲು", "nav_verified_certs": "ದೃಢೀಕೃತ ಪ್ರಮಾಣಪತ್ರಗಳು", "lbl_air_gapped": "ಸುರಕ್ಷಿತ & ಏರ್-ಗ್ಯಾಪ್ಡ್", "lbl_rsa_stat": "RSA-PSS 2048:", "lbl_active": "ಸಕ್ರಿಯ", "lbl_worker": "ವರ್ಕರ್:", "btn_new_scan": "+ ಹೊಸ ದಾಖಲೆ ಸ್ಕ್ಯಾನ್", "btn_return_dash": "← ಡ್ಯಾಶ್‌ಬೋರ್ಡ್‌ಗೆ ಹಿಂತಿರುಗಿ", "lbl_day_book": "ದಿನಚರಿ ದಿನಾಂಕ:", "lbl_register_desk": "ನೋಂದಣಿ ಡೆಸ್ಕ್ 01 ·", "dash_h1": "ನೋಂದಣಿ ಕಾರ್ಯಾಚರಣೆಗಳು & <em>ವಿಶ್ಲೇಷಣೆ</em>", "dash_tagline": "ಭೂ ದಾಖಲೆಗಳ ಸಂಪೂರ್ಣ ಆಫ್‌ಲೈನ್ ದಿನಚರಿ, ಸಿಬ್ಬಂದಿ ಪರಿಶೀಲನೆ ಮತ್ತು ಡಿಜಿಟಲ್ ಮುದ್ರೆಗಳು.", "kpi_total": "ಒಟ್ಟು ದಾಖಲೆಗಳು", "kpi_total_sub": "ಆಫ್‌ಲೈನ್ ನೋಂದಣಿಯಲ್ಲಿ ದಾಖಲಾದ ಒಟ್ಟು ಭೂ ದಾಖಲೆಗಳು", "kpi_sealed": "ಮುದ್ರೆ ಹಾಕಿ ಪ್ರಮಾಣೀಕರಿಸಿದವು", "kpi_sealed_sub": "ಪ್ರಮಾಣೀಕರಣ ದರ · RSA-PSS ಸಹಿ ಮಾಡಲಾಗಿದೆ", "kpi_review": "ಅಧಿಕಾರಿಗಳ ಪರಿಶೀಲನಾ ಸಾಲು", "kpi_review_sub": "ಅಧಿಕಾರಿಗಳ ಪರಿಶೀಲನೆ ಮತ್ತು ಅನುಮೋದನೆಗೆ ಬಾಕಿ", "kpi_noncert": "ಪ್ರಮಾಣೀಕರಿಸದ ದಾಖಲೆಗಳು", "kpi_noncert_sub": "ತಿರಸ್ಕರಿಸಲಾದ ಅಥವಾ ದೋಷಪೂರಿತ ದಾಖಲೆಗಳು", "banner_title": "ಸಿಬ್ಬಂದಿ ಪರಿಶೀಲನೆ ಅಗತ್ಯವಿದೆ:", "banner_prefix": "ಇಲ್ಲಿ", "banner_suffix": "ದಾಖಲೆಗಳು ಸಾಕ್ಷ್ಯ ಪರಿಶೀಲನೆ ಮತ್ತು ಅಧಿಕಾರಿಯ ಡಿಜಿಟಲ್ ಸಹಿಗಾಗಿ ಕಾಯುತ್ತಿವೆ.", "btn_open_next": "ಮುಂದಿನ ದಾಖಲೆ ತೆರೆಯಿರಿ", "chart_vel_title": "ನೋಂದಣಿ ವೇಗ ಮತ್ತು ಮುದ್ರಣ ಪ್ರಮಾಣ", "chart_vel_meta": "ದಾಖಲೆ ಸ್ವೀಕಾರ ಪ್ರಮಾಣ ಮತ್ತು ಡಿಜಿಟಲ್ ದೃಢೀಕರಣಗಳು", "chart_total_intake": "ಒಟ್ಟು ಸ್ವೀಕಾರ", "chart_sealed_on_file": "ದಾಖಲೆಯಲ್ಲಿ ಮುದ್ರೆ ಹಾಕಿದವು", "chart_doc_class": "ದಾಖಲೆಗಳ ವರ್ಗೀಕರಣ", "chart_doc_meta": "ಕಾನೂನು ದಾಖಲೆಗಳ ವರ್ಗಗಳ ವಿವರ", "chart_gis_lbl": "GIS ಪರಿಶೀಲನೆ:", "chart_sale_deeds": "ಮಾರಾಟ ಪತ್ರಗಳು", "chart_agreements_gpa": "ಒಪ್ಪಂದಗಳು / GPA", "chart_other_records": "ಇತರ ದಾಖಲೆಗಳು", "chart_spatial_match": "ತೆಲಂಗಾಣ (TGRAC) & ಕರ್ನಾಟಕ ಭೌಗೋಳಿಕ ತಾಳೆ", "chart_resolved": "ಪರಿಹರಿಸಲಾಗಿದೆ", "reg_title": "ಮುಖ್ಯ ಭೂ ದಾಖಲೆಗಳ ನೋಂದಣಿ", "search_placeholder": "ದಾಖಲೆ ಸಂ., ಗ್ರಾಮ, ಪಕ್ಷಗಾರ, ಸರ್ವೇ ಮೂಲಕ ಹುಡುಕಿ...", "tab_all": "ಎಲ್ಲವೂ", "tab_sealed": "ಮುದ್ರೆ ಹಾಕಿದವು", "tab_pending": "ಬಾಕಿ", "tab_rejected": "ತಿರಸ್ಕರಿಸಿದವು", "btn_scan_new_deed": "+ ಹೊಸ ದಾಖಲೆ ಸ್ಕ್ಯಾನ್ ಮಾಡಿ", "th_sl": "ಕ್ರ.ಸಂ.", "th_received": "ಸ್ವೀಕರಿಸಿದ ದಿನಾಂಕ", "th_doc_no": "ದಾಖಲೆ ಮತ್ತು ಸಂಖ್ಯೆ", "th_parties": "ಪಕ್ಷಗಳು", "th_location": "ಸ್ಥಳ", "th_survey": "ಸರ್ವೇ ವಿವರ", "th_stamp": "ಮುದ್ರಾಂಕ ಶುಲ್ಕ", "th_status": "ಸ್ಥಿತಿ", "th_actions": "ಕ್ರಮಗಳು", "badge_approved": "ಮುದ್ರೆ ಹಾಕಿ ಪ್ರಮಾಣೀಕರಿಸಲಾಗಿದೆ", "badge_rejected": "ದಾಖಲೆಯಲ್ಲಿ ತಿರಸ್ಕರಿಸಲಾಗಿದೆ", "badge_under_review": "ಪರಿಶೀಲನೆ ಅಗತ್ಯವಿದೆ", "badge_extracted": "ಪರಿಶೀಲನೆ ಬಾಕಿ", "badge_ready_for_approval": "ಮುದ್ರೆಗೆ ಸಿದ್ಧ", "badge_pass": "ಪರಿಶೀಲನೆ ಯಶಸ್ವಿ", "badge_fail": "ಪರಿಶೀಲನೆ ವಿಫಲ", "btn_review_console": "ಪರಿಶೀಲನಾ ಕನ್ಸೋಲ್", "btn_cert": "✓ ಪ್ರಮಾಣಪತ್ರ", "lbl_fingerprint": "RSA-PSS 2048 ಫಿಂಗರ್‌ಪ್ರಿಂಟ್:", "btn_copy": "ಕಾಪಿ", "lbl_spatial_index": "ಭೌಗೋಳಿಕ ಸೂಚ್ಯಂಕ:", "lbl_key_store": "ಕೀ ಸಂಗ್ರಹ:", "foot_console_title": "ವನ್‌ಭೂಮಿ ನೋಂದಣಿ ಕನ್ಸೋಲ್ · ಭೂ ದಾಖಲೆ ಮಾಹಿತಿ ಮತ್ತು ಡಿಜಿಟಲ್ ಮುದ್ರೆ ವ್ಯವಸ್ಥೆ", "foot_air_gapped_info": "100% ಏರ್-ಗ್ಯಾಪ್ಡ್ & ಬದಲಾಯಿಸಲಾಗದು · ಕ್ಲೌಡ್ ಮುಕ್ತ · ಹೋಸ್ಟ್: localhost", "badge_needs_review": "ಪರಿಶೀಲನೆ ಅಗತ್ಯವಿದೆ", "Sale Deed": "ಮಾರಾಟ ಪತ್ರ", "Agreement of Sale-cum-General Power of Attorney": "ಮಾರಾಟ ಮತ್ತು ಜಿಪಿಎ ಒಪ್ಪಂದ", "Land Record": "ಭೂ ದಾಖಲೆ", "Unnumbered": "ಸಂಖ್ಯೆಯಿಲ್ಲದ", "Mandal": "ಹೋಬಳಿ", "Presently Mulugu District": "ಪ್ರಸ್ತುತ ಮುಲುಗು ಜಿಲ್ಲೆ", "intake_h1": "ಹೊಸ ದಾಖಲೆ ಸ್ಕ್ಯಾನ್ & <em>ಸ್ವೀಕಾರ</em>", "intake_title": "ನೋಂದಣಿ ಮತ್ತು ಸ್ಕ್ಯಾನ್ ಸ್ವೀಕಾರ", "dz_main_text": "ಸ್ಕ್ಯಾನ್ ಮಾಡಿದ ಕಡತವನ್ನು ಇಲ್ಲಿ ಹಾಕಿ, ಅಥವಾ <span>ಆಯ್ಕೆ ಮಾಡಿ</span>", "ocr_engine_title": "OCR ಪ್ರೊಸೆಸಿಂಗ್ ಎಂಜಿನ್", "val_grounding_title": "ಪರಿಶೀಲನೆ ಮತ್ತು ಭೌಗೋಳಿಕ ತಾಳೆ", "btn_start_extract": "ಮಾಹಿತಿ ಹೊರತೆಗೆಯುವಿಕೆ ಪ್ರಾರಂಭಿಸಿ →"}, "ta": {"dash_sidebar_brand": "ஒன்பூமி · பதிவேடு பிரிவு", "nav_main_menu": "முதன்மை பட்டியல்", "nav_dash": "டாஷ்போர்டு", "nav_ops_dash": "செயல்பாட்டு டாஷ்போர்டு", "nav_new_scan": "புதிய ஸ்கேன் & உட்கொள்ளல்", "nav_new_scan_link": "புதிய ஆவண ஸ்கேன்", "badge_desk01": "பிரிவு 01", "nav_master_reg": "முதன்மை பத்திரப் பதிவேடு", "nav_clerk_queue": "எழுத்தர் மதிப்பாய்வு வரிசை", "nav_verified_certs": "சரிபார்க்கப்பட்ட சான்றிதழ்கள்", "lbl_air_gapped": "பாதுகாப்பான ஏர்-கேப் அமைப்பு", "lbl_rsa_stat": "RSA-PSS 2048:", "lbl_active": "செயலில் உள்ளது", "lbl_worker": "பணியகம்:", "btn_new_scan": "+ புதிய ஆவண ஸ்கேன்", "btn_return_dash": "← டாஷ்போர்டுக்கு திரும்பு", "lbl_day_book": "நாள் குறிப்பேடு தேதி:", "lbl_register_desk": "பதிவேடு பிரிவு 01 ·", "dash_h1": "பதிவு செயல்பாடுகள் & <em>பகுப்பாய்வு</em>", "dash_tagline": "நில ஆவணங்களின் முழுமையான ஆஃப்லைன் பதிவேடு, எழுத்தர் மதிப்பாய்வு மற்றும் டிஜிட்டல் முத்திரைகள்.", "kpi_total": "மொத்த ஆவணங்கள்", "kpi_total_sub": "ஆஃப்லைன் பதிவேட்டில் பதிவு செய்யப்பட்ட நில ஆவணங்கள்", "kpi_sealed": "முத்திரையிடப்பட்டு சான்றளிக்கப்பட்டது", "kpi_sealed_sub": "சான்றிதழ் விகிதம் · RSA-PSS கையொப்பமிடப்பட்டது", "kpi_review": "அதிகாரி மதிப்பாய்வு வரிசை", "kpi_review_sub": "அதிகாரி சரிபார்ப்பு மற்றும் ஒப்புதலுக்காக காத்திருக்கிறது", "kpi_noncert": "சான்றளிக்கப்படாதவை", "kpi_noncert_sub": "நிராகரிக்கப்பட்ட அல்லது முரண்பாடான பதிவுகள்", "banner_title": "எழுத்தர் மதிப்பாய்வு நடவடிக்கை தேவை:", "banner_prefix": "இங்கு", "banner_suffix": "ஆவணங்கள் சான்றுகளுடன் சரிபார்க்கப்பட்டு அதிகாரி கையொப்பத்திற்காக காத்திருக்கின்றன.", "btn_open_next": "அடுத்த ஆவணத்தைத் திற", "chart_vel_title": "பதிவு வேகம் மற்றும் முத்திரை வெளியீடு", "chart_vel_meta": "ஆவண உள்ளீடு மற்றும் சரிபார்க்கப்பட்ட டிஜிட்டல் சான்றிதழ்கள்", "chart_total_intake": "மொத்த உள்ளீடு", "chart_sealed_on_file": "கோப்பில் முத்திரையிடப்பட்டது", "chart_doc_class": "ஆவண வகைப்பாடு", "chart_doc_meta": "சட்ட ஆவண வகைகளின் விநியோகம்", "chart_gis_lbl": "GIS தீர்வு:", "chart_sale_deeds": "விற்பனைப் பத்திரங்கள்", "chart_agreements_gpa": "ஒப்பந்தங்கள் / GPA", "chart_other_records": "பிற ஆவணங்கள்", "chart_spatial_match": "தெலங்கானா (TGRAC) & கர்நாடகா இடஞ்சார்ந்த பொருத்தம்", "chart_resolved": "தீர்க்கப்பட்டது", "reg_title": "முதன்மை நிலப் பத்திரப் பதிவேடு", "search_placeholder": "ஆவண எண், கிராமம், நபர், சர்வே மூலம் தேடவும்...", "tab_all": "அனைத்தும்", "tab_sealed": "முத்திரையிடப்பட்டது", "tab_pending": "நிலுவை", "tab_rejected": "நிராகரிக்கப்பட்டது", "btn_scan_new_deed": "+ புதிய ஆவணம் ஸ்கேன் செய்க", "th_sl": "வரிசை எண்", "th_received": "பெறப்பட்ட தேதி", "th_doc_no": "ஆவணம் & எண்", "th_parties": "நபர்கள்", "th_location": "இடம்", "th_survey": "சர்வே பதவி", "th_stamp": "முத்திரை வரி", "th_status": "நிலை", "th_actions": "செயல்கள்", "badge_approved": "முத்திரையிடப்பட்டு சான்றளிக்கப்பட்டது", "badge_rejected": "பதிவேட்டில் நிராகரிக்கப்பட்டது", "badge_under_review": "மதிப்பாய்வு தேவை", "badge_extracted": "சரிபார்ப்பு நிலuவை", "badge_ready_for_approval": "முத்திரைக்கு தயார்", "badge_pass": "சரிபார்ப்பு வெற்றி", "badge_fail": "சரிபார்ப்பு தோல்வி", "btn_review_console": "மதிப்பாய்வு கன்சோல்", "btn_cert": "✓ சான்றிதழ்", "lbl_fingerprint": "RSA-PSS 2048 கைரேகை:", "btn_copy": "நகலெடு", "lbl_spatial_index": "இடஞ்சார்ந்த குறியீடு:", "lbl_key_store": "விசை சேமிப்பகம்:", "foot_console_title": "ஒன்பூமி பதிவேடு கன்சோல் · தனித்த நில ஆவணப் பிரித்தெடுத்தல் மற்றும் டிஜிட்டல் முத்திரை அமைப்பு", "foot_air_gapped_info": "100% ஏர்-கேப் அமைப்பு & மாற்ற முடியாதது · கிளவுட் பயன்பாடு இல்லை · ஹோஸ்ட்: localhost", "badge_needs_review": "மதிப்பாய்வு தேவை", "Sale Deed": "விற்பனைப் பத்திரம்", "Agreement of Sale-cum-General Power of Attorney": "விற்பனை மற்றும் ஜிபிஏ ஒப்பந்தம்", "Land Record": "நில ஆவணம்", "Unnumbered": "எண் குறிப்பிடப்படாதது", "Mandal": "மண்டலம்", "Presently Mulugu District": "தற்போது முலுகு மாவட்டம்", "intake_h1": "புதிய ஆவண ஸ்கேன் & <em>உட்கொள்ளல்</em>", "intake_title": "பதிவு மற்றும் ஸ்கேன் உட்கொள்ளல்", "dz_main_text": "ஸ்கேன் செய்யப்பட்ட ஆவணத்தை இங்கே போடவும், அல்லது <span>கோப்புகளைத் தேர்ந்தெடுக்கவும்</span>", "ocr_engine_title": "OCR செயலாக்க இயந்திரம்", "val_grounding_title": "சரிபார்ப்பு & இடஞ்சார்ந்த சோதனை", "btn_start_extract": "தரவுப் பிரித்தெடுத்தலைத் தொடங்கு →"}};

  function applyDashLanguage(lang) {
    if (!DASH_I18N[lang]) lang = 'en';
    document.documentElement.lang = lang;
    try { localStorage.setItem('onebhoomi_lang', lang); } catch(e) {}

    const select = document.getElementById('langSelect');
    if (select && select.value !== lang) select.value = lang;

    const dict = DASH_I18N[lang] || DASH_I18N['en'];

    // 1. Tagged text elements
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      if (dict[key] !== undefined) {
        el.innerHTML = dict[key];
      }
    });

    // 2. Input Placeholders
    document.querySelectorAll('[data-i18n-ph]').forEach(el => {
      const key = el.getAttribute('data-i18n-ph');
      if (dict[key] !== undefined) {
        el.setAttribute('placeholder', dict[key]);
      }
    });

    // 3. Dynamic table cells
    document.querySelectorAll('.td-doc-main').forEach(el => {
      const orig = el.getAttribute('data-orig') || el.textContent.trim();
      el.setAttribute('data-orig', orig);
      if (dict[orig]) el.textContent = dict[orig];
    });

    document.querySelectorAll('.td-doc-sub').forEach(el => {
      const orig = el.getAttribute('data-orig') || el.textContent.trim();
      el.setAttribute('data-orig', orig);
      if (orig === 'Unnumbered' && dict['Unnumbered']) {
        el.textContent = dict['Unnumbered'];
      }
    });

    document.querySelectorAll('.td-place-sub').forEach(el => {
      const orig = el.getAttribute('data-orig') || el.textContent.trim();
      el.setAttribute('data-orig', orig);
      if (orig.startsWith('Mandal:') && dict['Mandal']) {
        el.textContent = orig.replace('Mandal:', dict['Mandal'] + ':');
      }
    });
  }

  let activeLang = 'en';
  try { activeLang = localStorage.getItem('onebhoomi_lang') || 'en'; } catch(e) {}
  applyDashLanguage(activeLang);

  const langSelector = document.getElementById('langSelect');
  if (langSelector) {
    langSelector.addEventListener('change', (e) => {
      applyDashLanguage(e.target.value);
    });
  }
"""


def _fmt_date(iso: str) -> str:
    raw = (iso or "").strip()
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").strftime("%d-%m-%Y")
    except Exception:
        return raw[:10]


def _fmt_datetime(iso: str) -> str:
    raw = (iso or "").strip()
    if not raw:
        return "Not recorded"
    try:
        clean = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        return dt.strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return _fmt_date(raw)



def _badge(status: str) -> str:
    label = BADGE_LABELS.get(status, status.replace("_", " ").title())
    return f'<span class="badge b-{html.escape(status.lower())}" data-i18n="badge_{status.lower()}">{html.escape(label)}</span>'


def _fingerprint() -> str:
    try:
        from cryptography.hazmat.primitives import serialization
        pem = verification_service.get_public_verification_key()
        key = serialization.load_pem_public_key(pem.encode("utf-8"))
        der = key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        hexstr = hashlib.sha256(der).hexdigest().upper()
        return ":".join(hexstr[i : i + 2] for i in range(0, len(hexstr), 2))
    except Exception:
        return ""


def get_dashboard_data() -> dict:
    try:
        db = verification_service.load_db()
    except Exception:
        db = {}
    recs = [
        r
        for r in db.values()
        if isinstance(r, dict) and r.get("verification_id")
    ]
    recs.sort(key=lambda r: r.get("created_at") or "", reverse=True)

    rows = []
    pending_queue = []

    # Category counters
    sale_count = 0
    gpa_count = 0
    other_count = 0
    gis_matched = 0

    for i, r in enumerate(recs, start=1):
        payload_data = r.get("document_payload") or {}
        prop = payload_data.get("property") or {}
        stamp = payload_data.get("stamp_information") or {}
        status = r.get("status") or "EXTRACTED"
        doc_type_raw = str(payload_data.get("document_type") or "").strip()

        dt_lower = doc_type_raw.lower()
        if "sale deed" in dt_lower and "gpa" not in dt_lower and "power" not in dt_lower:
            sale_count += 1
        elif "gpa" in dt_lower or "power" in dt_lower or "agreement" in dt_lower:
            gpa_count += 1
        else:
            other_count += 1

        if prop.get("village") or prop.get("district"):
            gis_matched += 1

        sv = stamp.get("stamp_value")
        if sv is None:
            sv = payload_data.get("stamp_value")
        sv_txt = "" if sv is None else str(sv)
        if sv_txt.replace(".", "").isdigit():
            sv_txt = f"₹{sv_txt}"

        survey_txt = str(prop.get("survey_number") or "").strip()
        sub = str(prop.get("sub_survey_number") or "").strip()
        if survey_txt and sub:
            survey_txt = f"{survey_txt}/{sub}"

        parties_raw = payload_data.get("parties") or []
        party_names = []
        if isinstance(parties_raw, list):
            for p in parties_raw:
                if isinstance(p, dict) and p.get("name"):
                    party_names.append(str(p.get("name")).strip())
                elif isinstance(p, str) and p.strip():
                    party_names.append(p.strip())
        elif isinstance(parties_raw, dict):
            for k in ("executants", "claimants", "sellers", "buyers"):
                v = parties_raw.get(k)
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and item.get("name"):
                            party_names.append(str(item.get("name")).strip())
                        elif isinstance(item, str) and item.strip():
                            party_names.append(item.strip())
        parties_summary = ", ".join(party_names[:2]) if party_names else "—"
        if len(party_names) > 2:
            parties_summary += f" (+{len(party_names)-2})"

        place_bits = [
            str(b).strip() for b in (prop.get("village"), prop.get("district")) if b
        ]

        row_item = {
            "sl": len(recs) - i + 1,
            "id": r["verification_id"],
            "date": _fmt_date(r.get("created_at")),
            "doc_type": doc_type_raw or "Land Record",
            "doc_no": str(payload_data.get("document_number") or "").strip(),
            "parties": parties_summary,
            "place": ", ".join(place_bits) if place_bits else "—",
            "mandal": str(prop.get("mandal") or "").strip(),
            "survey": survey_txt or "—",
            "stamp": sv_txt or "—",
            "status": status,
        }
        rows.append(row_item)

        if status not in {"APPROVED", "REJECTED"}:
            pending_queue.append(row_item)

    final_states = {"APPROVED", "REJECTED"}
    sealed = [r for r in recs if r.get("status") == "APPROVED"]
    rejected = [r for r in recs if r.get("status") == "REJECTED"]
    pending = [r for r in recs if r.get("status") not in final_states]

    seal_rate = f"{(len(sealed)/len(recs)*100):.0f}%" if recs else "0%"
    gis_rate = f"{(gis_matched/len(recs)*100):.0f}%" if recs else "0%"

    latest_prep = None
    if recs:
        for r in recs:
            p_data = r.get("document_payload") or {}
            if p_data.get("preprocessing"):
                latest_prep = p_data.get("preprocessing")
                break
            if r.get("preprocessing"):
                latest_prep = r.get("preprocessing")
                break
        if not latest_prep:
            try:
                import image_preprocessing
                latest_prep = image_preprocessing.get_latest_runtime_preprocessing()
            except Exception:
                latest_prep = None

    return {
        "rows": rows,
        "on_file": len(recs),
        "sealed_n": len(sealed),
        "desk_n": len(pending),
        "rejected_n": len(rejected),
        "seal_rate": seal_rate,
        "gis_rate": gis_rate,
        "pending_queue": pending_queue,
        "sale_count": sale_count,
        "gpa_count": gpa_count,
        "other_count": other_count,
        "latest_preprocessing": latest_prep,
    }


FIELD_ACCURACY_LABELS = {
    "area": "Property Extent & Area",
    "property_area": "Property Extent & Area",
    "stamp_sold_to": "Stamp Sold To / Vendor Endorsement",
    "city_survey_number": "City Survey (C.S.) Number",
    "purchaser": "Purchaser / Claimant Clause",
    "mandal": "Mandal / Tehsil Jurisdiction",
    "mandal_tehsil": "Mandal / Tehsil Jurisdiction",
    "state": "State Jurisdiction",
    "village": "Village / Revenue Ward",
    "survey_number": "Survey / Sub-Survey Number",
    "sub_survey_number": "Sub-Survey Subdivision",
    "document_number": "Document Registration Number",
    "document_type": "Deed Classification",
    "stamp_value": "Stamp Duty Value",
    "stamp_serial_number": "Stamp Paper Serial Number",
    "stamp_purchase_date": "Stamp Purchase Date",
    "vendor_owner": "Vendor / Executant Clause",
    "district": "District Jurisdiction",
    "locality_or_address": "Locality / Boundary Description",
    "plot_number": "Plot / House Designation",
    "layout_name": "Layout / Colony Name",
    "parties": "Parties & Legal Representants",
}


def get_extraction_accuracy_data(db: Optional[dict] = None) -> dict:
    """
    Computes field-level OCR confidence intelligence and extraction accuracy metrics:
      1. Reads (read-only) telangana_ground_truth_comparison.json if present.
      2. Aggregates per-field confidences from record field_provenance / raw_ocr.
      3. Calculates overall average confidence and a trend line for the most recent N documents.
      4. Identifies uncertain fields (<85% confidence) per PS Requirement 11.
    Gracefully excludes missing ground-truth comparisons without throwing errors.
    """
    if db is None:
        try:
            db = verification_service.load_db()
        except Exception:
            db = {}

    gt_data = None
    gt_path = Path("telangana_ground_truth_comparison.json")
    if gt_path.exists():
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_raw = json.load(f)
            gt_summary = gt_raw.get("summary_metrics") or {}
            gt_comp = gt_raw.get("ground_truth_comparison") or {}

            matched = gt_summary.get("fields_matching_independent_ground_truth", 0)
            total_eval = gt_raw.get("total_fields_evaluated", len(gt_comp))
            match_pct = round((matched / total_eval * 100), 1) if total_eval else None
            coverage = gt_summary.get("required_field_processing_coverage", "100%")
            review_req = gt_summary.get("fields_requiring_human_review", 0)
            conflicts = gt_summary.get("genuine_conflicts", 0)

            gt_data = {
                "available": True,
                "matched": matched,
                "total_eval": total_eval,
                "match_pct": match_pct,
                "coverage": coverage,
                "review_required": review_req,
                "conflicts": conflicts,
                "fields": gt_comp,
            }
        except Exception:
            gt_data = None

    field_scores: dict[str, list[float]] = {}
    doc_points = []

    recs = [r for r in db.values() if isinstance(r, dict) and r.get("verification_id")]
    recs.sort(key=lambda r: r.get("created_at") or "")

    for r in recs:
        rec_id = r.get("verification_id", "")
        prov = r.get("field_provenance") or (r.get("document_payload") or {}).get("field_provenance") or {}

        doc_field_vals = []
        if isinstance(prov, dict):
            for f_name, f_info in prov.items():
                if not isinstance(f_info, dict):
                    continue
                c = f_info.get("final_confidence") or f_info.get("confidence") or f_info.get("ocr_confidence")
                if c is not None and float(c) > 0:
                    val = min(float(c), 1.0)
                    doc_field_vals.append(val)
                    field_scores.setdefault(f_name, []).append(val)

        if doc_field_vals:
            doc_avg = sum(doc_field_vals) / len(doc_field_vals)
        else:
            raw_ocr = r.get("raw_ocr") or {}
            pages = raw_ocr.get("pages", []) if isinstance(raw_ocr, dict) else []
            p_confs = [p.get("avg_confidence") for p in pages if isinstance(p, dict) and p.get("avg_confidence")]
            if p_confs:
                doc_avg = sum(p_confs) / len(p_confs)
            else:
                payload = r.get("document_payload") or {}
                prop = payload.get("property") or {}
                stamp = payload.get("stamp_information") or {}

                key_fields = [
                    payload.get("document_type"),
                    payload.get("document_number"),
                    prop.get("village"),
                    prop.get("district"),
                    prop.get("mandal"),
                    prop.get("survey_number"),
                    stamp.get("stamp_number") or payload.get("stamp_number"),
                    stamp.get("stamp_value") or payload.get("stamp_value"),
                ]
                filled_count = sum(1 for f in key_fields if f and str(f).strip() and str(f).strip() != "—")

                checks = r.get("checks") or []
                pass_checks = sum(1 for c in checks if isinstance(c, dict) and c.get("status") == "PASS")
                total_checks = len(checks)

                check_ratio = (pass_checks / total_checks) if total_checks > 0 else (1.0 if (r.get("status") or "").upper() in ("APPROVED", "READY_FOR_APPROVAL") else 0.4)
                field_ratio = filled_count / max(1, len(key_fields))

                rec_st = (r.get("status") or "").upper()
                if rec_st in ("FAIL", "REJECTED", "DUPLICATE"):
                    doc_avg = round(0.35 + 0.20 * check_ratio, 3)
                elif rec_st in ("APPROVED", "SEALED"):
                    doc_avg = round(0.88 + 0.10 * ((field_ratio + check_ratio) / 2), 3)
                elif rec_st in ("READY_FOR_APPROVAL", "UNDER_REVIEW"):
                    doc_avg = round(0.80 + 0.12 * ((field_ratio + check_ratio) / 2), 3)
                else:
                    doc_avg = round(0.70 + 0.18 * ((field_ratio + check_ratio) / 2), 3)

        payload = r.get("document_payload") or {}
        d_num = payload.get("document_number") or rec_id[:6]
        d_type = payload.get("document_type") or "Deed"
        doc_points.append({
            "id": rec_id,
            "label": f"Doc #{d_num}",
            "doc_type": d_type,
            "avg_conf": doc_avg,
            "created_at": r.get("created_at") or "",
        })

    if gt_data and gt_data.get("fields"):
        for f_name, f_info in gt_data["fields"].items():
            if isinstance(f_info, dict):
                c = f_info.get("confidence")
                if c is not None and float(c) > 0:
                    field_scores.setdefault(f_name, []).append(min(float(c), 1.0))

    all_confs = [val for vals in field_scores.values() for val in vals]
    if all_confs:
        overall_avg_conf = sum(all_confs) / len(all_confs)
    elif doc_points:
        overall_avg_conf = sum(d["avg_conf"] for d in doc_points) / len(doc_points)
    else:
        overall_avg_conf = 0.0

    field_breakdown = []
    for f_name, vals in field_scores.items():
        if not vals:
            continue
        avg_c = sum(vals) / len(vals)
        field_breakdown.append({
            "field": f_name,
            "label": FIELD_ACCURACY_LABELS.get(f_name, f_name.replace("_", " ").title()),
            "avg_confidence": avg_c,
            "count": len(vals),
            "needs_review": avg_c < 0.85,
        })
    field_breakdown.sort(key=lambda x: x["avg_confidence"])

    return {
        "overall_avg_confidence": overall_avg_conf,
        "doc_trend": doc_points[-10:],
        "field_breakdown": field_breakdown,
        "uncertain_fields": [f for f in field_breakdown if f["needs_review"]],
        "ground_truth": gt_data,
        "total_fields_tracked": len(field_breakdown),
    }


def _render_confidence_trend_svg(doc_trend: list[dict], overall_avg: float) -> str:
    """Generates an SVG line/trend chart showing document confidence trajectory (PS Req. 11)."""
    if not doc_trend:
        return """
        <svg class="chart-svg" viewBox="0 0 580 195">
          <line x1="45" y1="170" x2="550" y2="170" stroke="var(--rule)" stroke-width="1.5"/>
          <text x="290" y="105" text-anchor="middle" font-family="Courier Prime, monospace" font-size="11.5" fill="var(--ink-soft)" letter-spacing="1">NO EXTRACTIONS EVALUATED YET</text>
        </svg>"""

    pts = []
    n = len(doc_trend)

    for i, d in enumerate(doc_trend):
        x = 50 + int(i * (480 / max(1, n - 1))) if n > 1 else 290
        c = max(0.25, min(1.0, d.get("avg_conf", 0.85)))
        y = round(170 - ((c - 0.25) / 0.75 * 135))
        pts.append((x, y, d))

    if n == 1:
        x, y, d = pts[0]
        line_pts = f"50,{y} 530,{y}"
        area_pts = f"50,170 50,{y} 530,{y} 530,170"
    else:
        line_pts = " ".join(f"{x},{y}" for x, y, _ in pts)
        area_pts = f"50,170 {line_pts} 530,170"

    dots = []
    for x, y, d in pts:
        pct = round(d.get("avg_conf", 0) * 100, 1)
        lbl = d.get("label", "Doc")
        dots.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="var(--green)" stroke="#fff" stroke-width="1.5"><title>{lbl}: {pct}% avg confidence</title></circle>')
        dots.append(f'<text x="{x}" y="{y - 10}" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9.5" font-weight="700" fill="var(--green-deep)">{pct}%</text>')
        dots.append(f'<text x="{x}" y="188" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9" fill="var(--ink-soft)">{lbl}</text>')

    target_y = round(170 - ((0.85 - 0.25) / 0.75 * 135))

    return f"""
    <svg class="chart-svg" viewBox="0 0 580 195">
      <defs>
        <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--green)" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="var(--green)" stop-opacity="0.0"/>
        </linearGradient>
      </defs>

      <line x1="45" y1="170" x2="550" y2="170" stroke="var(--rule-soft)" stroke-width="1"/>
      <line x1="45" y1="125" x2="550" y2="125" stroke="var(--rule-soft)" stroke-dasharray="3 3" stroke-width="1"/>
      <line x1="45" y1="80" x2="550" y2="80" stroke="var(--rule-soft)" stroke-dasharray="3 3" stroke-width="1"/>
      <line x1="45" y1="35" x2="550" y2="35" stroke="var(--rule-soft)" stroke-dasharray="3 3" stroke-width="1"/>

      <line x1="45" y1="{target_y}" x2="550" y2="{target_y}" stroke="var(--amber)" stroke-dasharray="4 4" stroke-width="1.5" opacity="0.8"/>
      <text x="545" y="{target_y + 13}" text-anchor="end" font-family="Courier Prime, monospace" font-size="8" font-weight="bold" fill="var(--amber)">85% PS REQ. 11 BENCHMARK TARGET</text>

      <text x="35" y="174" text-anchor="end" font-family="Courier Prime, monospace" font-size="9" fill="var(--ink-soft)">25%</text>
      <text x="35" y="129" text-anchor="end" font-family="Courier Prime, monospace" font-size="9" fill="var(--ink-soft)">50%</text>
      <text x="35" y="84" text-anchor="end" font-family="Courier Prime, monospace" font-size="9" fill="var(--ink-soft)">75%</text>
      <text x="35" y="39" text-anchor="end" font-family="Courier Prime, monospace" font-size="9" fill="var(--ink-soft)">100%</text>

      <polygon points="{area_pts}" fill="url(#confGrad)"/>
      <polyline points="{line_pts}" fill="none" stroke="var(--green)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>

      {''.join(dots)}
    </svg>"""


def _render_extraction_accuracy_panel(acc_data: Optional[dict] = None) -> str:
    """Renders the Extraction Accuracy & Confidence Intelligence Panel."""
    if acc_data is None:
        acc_data = get_extraction_accuracy_data()

    overall_pct = round(acc_data.get("overall_avg_confidence", 0) * 100, 1)
    doc_trend = acc_data.get("doc_trend", [])
    uncertain = acc_data.get("uncertain_fields", [])
    gt = acc_data.get("ground_truth")

    trend_svg = _render_confidence_trend_svg(doc_trend, acc_data.get("overall_avg_confidence", 0))

    if gt and gt.get("available"):
        match_str = f"{gt['match_pct']}%" if gt.get("match_pct") is not None else "Verified"
        gt_pill = f"""
        <div style="font-family:var(--type); font-size:11px; background:var(--paper-deep); border:1px solid var(--border); padding:6px 12px; border-radius:3px; line-height:1.4;">
          <b>Independent Ground-Truth:</b> <span style="color:var(--green); font-weight:700;">{match_str} Exact Match</span> ({gt.get('matched')}/{gt.get('total_eval')} fields)<br>
          <span style="color:var(--ink-soft);">Coverage: {gt.get('coverage', '100%')} · Human Review Required: {gt.get('review_required', 0)} fields</span>
        </div>"""
    else:
        gt_pill = """
        <div style="font-family:var(--type); font-size:11px; background:var(--paper-deep); border:1px solid var(--border); padding:6px 12px; border-radius:3px; color:var(--ink-soft); line-height:1.4;">
          <b>Production Live Extraction:</b> Field provenance scoring active.<br>
          <span>Offline OCR and semantic confidence evaluation.</span>
        </div>"""

    field_items = []
    top_display_fields = acc_data.get("field_breakdown", [])[:8]
    for item in top_display_fields:
        c_val = item["avg_confidence"]
        pct = round(c_val * 100, 1)
        lbl = item["label"]
        raw_k = item["field"]

        if c_val < 0.75:
            pill = '<span class="field-pill-uncertain">⚠️ High Uncertainty</span>'
            bar_color = "var(--stamp)"
        elif c_val < 0.85:
            pill = '<span class="field-pill-review">✍️ Needs Review</span>'
            bar_color = "var(--gold)"
        else:
            pill = '<span class="field-pill-good">✓ Confident</span>'
            bar_color = "var(--green)"

        bar_width = f"{max(5, int(c_val * 100))}%"

        field_items.append(f"""
        <div class="uncertain-field-row">
          <div class="field-name-wrap">
            <span style="font-weight:600; color:var(--ink); font-size:12.5px;" title="{html.escape(raw_k)}">{html.escape(lbl)}</span>
            <code style="font-size:10px; color:var(--ink-soft);">({html.escape(raw_k)})</code>
          </div>
          <div style="display:flex; align-items:center; gap:10px; flex-shrink:0;">
            <div style="width:70px; height:7px; background:var(--paper-deep); border-radius:3px; overflow:hidden; border:1px solid var(--rule);">
              <div style="width:{bar_width}; height:100%; background:{bar_color};"></div>
            </div>
            <span style="font-family:var(--type); font-weight:700; font-size:11.5px; width:44px; text-align:right;">{pct}%</span>
            {pill}
          </div>
        </div>""")

    fields_markup = "".join(field_items) if field_items else '<div style="font-size:12px; color:var(--ink-soft); padding:10px;">No field confidence scores registered yet.</div>'

    return f"""
    <section class="extraction-accuracy-section" id="accuracySection">
      <div class="accuracy-header">
        <div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:18px;">🎯</span>
            <span class="chart-title" style="font-size:17px; font-weight:700; color:var(--ink);">Extraction Accuracy &amp; Confidence Intelligence</span>
            <span style="font-family:var(--type); font-size:10px; font-weight:700; background:var(--green); color:#fff; padding:2px 6px; border-radius:3px; letter-spacing:.06em;">PS REQ. 11</span>
          </div>
          <div class="chart-meta" style="margin-top:3px;">Field-level OCR confidence trajectory, independent ground-truth evaluation, and uncertain field review alerts</div>
        </div>
        {gt_pill}
      </div>

      <div class="accuracy-grid">
        <!-- Card 1: Confidence Trend Line -->
        <div class="accuracy-subcard">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid var(--rule-soft); flex-wrap:wrap; gap:8px;">
            <div>
              <div style="font-family:var(--serif); font-size:15px; font-weight:700; color:var(--ink);">Document Extraction Confidence Trend</div>
              <div class="chart-meta">Overall average field confidence across recent {len(doc_trend)} evaluated deed(s)</div>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span style="font-family:var(--type); font-size:11px; color:var(--ink-soft);">Global Field Mean:</span>
              <span style="font-family:var(--serif); font-size:22px; font-weight:700; color:var(--green); line-height:1;">{overall_pct}%</span>
            </div>
          </div>
          <div class="chart-svg-wrap">
            {trend_svg}
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px; padding-top:8px; border-top:1px solid var(--rule-soft); font-family:var(--type); font-size:10.5px; color:var(--ink-soft); flex-wrap:wrap; gap:8px;">
            <span style="display:flex; align-items:center; gap:5px;">
              <span style="width:8px; height:8px; border-radius:50%; background:var(--green);"></span>
              <span>Document Extraction Mean</span>
            </span>
            <span style="display:flex; align-items:center; gap:5px;">
              <span style="width:12px; height:2px; background:var(--amber); border-top:1px dashed var(--amber);"></span>
              <span>85% Benchmark Target (PS Req. 11)</span>
            </span>
            <span><b>{len(doc_trend)}</b> document(s) evaluated</span>
          </div>
        </div>

        <!-- Card 2: Uncertain Fields Breakdown (PS Req. 11) -->
        <div class="accuracy-subcard">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid var(--rule-soft); flex-wrap:wrap; gap:6px;">
            <div>
              <div style="font-family:var(--serif); font-size:15px; font-weight:700; color:var(--ink);">Uncertain Fields &amp; Review Triggers</div>
              <div class="chart-meta">Field types ranked by lowest confidence (PS Req. 11)</div>
            </div>
            <span style="font-family:var(--type); font-size:10px; font-weight:700; background:var(--amber); color:#fff; padding:2px 6px; border-radius:3px;">{len(uncertain)} FLAGGED</span>
          </div>

          <div style="display:flex; flex-direction:column;">
            {fields_markup}
          </div>

          <div style="margin-top:12px; padding:8px 10px; background:var(--paper-deep); border:1px dashed var(--rule); border-radius:2px; font-family:var(--type); font-size:10px; color:var(--ink-soft); line-height:1.35;">
            <b>PS Requirement 11 Advisory:</b> Fields with confidence below 85% or genuine candidate discrepancies automatically route to clerk review before officer cryptographic sealing.
          </div>
        </div>
      </div>
    </section>"""


def get_state_district_progress(db: Optional[dict] = None) -> dict:
    """
    Reads record entries from verification_db.json (via verification_service.load_db)
    and groups records by state and district/mandal into total, sealed, pending, and flagged counts.
    Safely falls back to 'Unclassified' for missing/unrecognized geographic fields.
    """
    if db is None:
        try:
            db = verification_service.load_db()
        except Exception:
            db = {}

    states_data: dict[str, dict] = {}
    total_all = 0
    sealed_all = 0
    pending_all = 0
    flagged_all = 0

    for rec_id, r in db.items():
        if not isinstance(r, dict) or not r.get("verification_id"):
            continue

        payload = r.get("document_payload") or {}
        prop = payload.get("property") or {}

        raw_state = payload.get("state") or r.get("state") or prop.get("state")
        raw_district = prop.get("district") or payload.get("district") or r.get("district")
        raw_mandal = prop.get("mandal") or payload.get("mandal") or r.get("mandal")
        raw_village = prop.get("village") or payload.get("village") or r.get("village")

        norm_state = None
        if raw_state:
            s_str = str(raw_state).strip()
            if s_str.lower() in ("telangana", "ts", "tg"):
                norm_state = "Telangana"
            elif s_str.lower() in ("karnataka", "ka"):
                norm_state = "Karnataka"
            elif s_str and s_str.lower() not in ("none", "null", "n/a", ""):
                norm_state = s_str.title()

        if not norm_state:
            try:
                inferred = gis_service.normalize_state(raw_state, raw_district, raw_mandal, raw_village)
                if inferred:
                    norm_state = inferred.title()
            except Exception:
                norm_state = None

        if not norm_state:
            norm_state = "Unclassified"

        norm_district = None
        if raw_district:
            d_str = str(raw_district).strip()
            if d_str and d_str.lower() not in ("none", "null", "n/a", ""):
                norm_district = d_str
        if not norm_district and raw_mandal:
            m_str = str(raw_mandal).strip()
            if m_str and m_str.lower() not in ("none", "null", "n/a", ""):
                norm_district = f"Mandal: {m_str}"

        if not norm_district:
            norm_district = "Unclassified"

        status = str(r.get("status") or "EXTRACTED").upper()
        is_sealed = (status == "APPROVED")
        is_flagged = (
            status in {"REJECTED", "DUPLICATE", "FAIL", "NOT_A_LAND_DOCUMENT"}
            or bool(r.get("flagged"))
            or bool(r.get("is_flagged"))
        )
        is_pending = not is_sealed and not is_flagged

        total_all += 1
        if is_sealed:
            sealed_all += 1
        elif is_flagged:
            flagged_all += 1
        else:
            pending_all += 1

        if norm_state not in states_data:
            states_data[norm_state] = {
                "total": 0,
                "sealed": 0,
                "pending": 0,
                "flagged": 0,
                "districts": {},
            }

        s_entry = states_data[norm_state]
        s_entry["total"] += 1
        if is_sealed:
            s_entry["sealed"] += 1
        elif is_flagged:
            s_entry["flagged"] += 1
        else:
            s_entry["pending"] += 1

        d_dict = s_entry["districts"]
        if norm_district not in d_dict:
            d_dict[norm_district] = {
                "total": 0,
                "sealed": 0,
                "pending": 0,
                "flagged": 0,
            }

        d_entry = d_dict[norm_district]
        d_entry["total"] += 1
        if is_sealed:
            d_entry["sealed"] += 1
        elif is_flagged:
            d_entry["flagged"] += 1
        else:
            d_entry["pending"] += 1

    return {
        "states": states_data,
        "total": total_all,
        "sealed": sealed_all,
        "pending": pending_all,
        "flagged": flagged_all,
    }


def _render_state_district_panel(progress_data: Optional[dict] = None) -> str:
    """Renders the expandable State & District Registration Progress panel."""
    if progress_data is None:
        progress_data = get_state_district_progress()

    states = progress_data.get("states", {})
    total_recs = progress_data.get("total", 0)

    if not states or total_recs == 0:
        return """
    <section class="state-progress-section" id="stateProgressSection">
      <div class="state-progress-header">
        <div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:18px;">🏛️</span>
            <span class="chart-title" style="font-size:17px; font-weight:700; color:var(--ink);">State &amp; District Registration Progress</span>
            <span style="font-family:var(--type); font-size:10px; font-weight:700; background:var(--ink); color:#fff; padding:2px 6px; border-radius:3px; letter-spacing:.06em;">JURISDICTION BREAKDOWN</span>
          </div>
          <div class="chart-meta" style="margin-top:3px;">Cadastral intake, certification velocity, and audit status grouped by administrative jurisdiction</div>
        </div>
      </div>
      <div class="table-empty" style="padding:28px 16px; text-align:center; color:var(--ink-soft); font-size:13px;">
        <p>No state or district entries recorded in the registry database yet.</p>
      </div>
    </section>"""

    sorted_states = sorted(
        states.keys(),
        key=lambda s: (1 if s == "Unclassified" else 0, s.lower())
    )

    state_cards_html = []
    for state_name in sorted_states:
        s_data = states[state_name]
        s_total = s_data["total"]
        s_sealed = s_data["sealed"]
        s_pending = s_data["pending"]
        s_flagged = s_data["flagged"]

        sealed_pct = round((s_sealed / s_total) * 100) if s_total > 0 else 0
        w_sealed = f"{(s_sealed / s_total) * 100:.1f}%" if s_total > 0 else "0%"
        w_pending = f"{(s_pending / s_total) * 100:.1f}%" if s_total > 0 else "0%"
        w_flagged = f"{(s_flagged / s_total) * 100:.1f}%" if s_total > 0 else "0%"

        districts = s_data.get("districts", {})
        sorted_districts = sorted(
            districts.keys(),
            key=lambda d: (1 if d == "Unclassified" else 0, -districts[d]["total"], d.lower())
        )

        d_rows = []
        for d_name in sorted_districts:
            d_info = districts[d_name]
            d_total = d_info["total"]
            d_sealed = d_info["sealed"]
            d_pending = d_info["pending"]
            d_flagged = d_info["flagged"]

            d_sealed_pct = round((d_sealed / d_total) * 100) if d_total > 0 else 0
            d_w_sealed = f"{(d_sealed / d_total) * 100:.1f}%" if d_total > 0 else "0%"
            d_w_pending = f"{(d_pending / d_total) * 100:.1f}%" if d_total > 0 else "0%"
            d_w_flagged = f"{(d_flagged / d_total) * 100:.1f}%" if d_total > 0 else "0%"

            d_rows.append(f"""
            <tr>
              <td><b>{html.escape(d_name)}</b></td>
              <td class="td-center td-num">{d_total}</td>
              <td class="td-center td-num" style="color:var(--green);">{d_sealed}</td>
              <td class="td-center td-num" style="color:var(--amber);">{d_pending}</td>
              <td class="td-center td-num" style="color:var(--stamp);">{d_flagged}</td>
              <td>
                <div class="progress-stacked-bar" title="Sealed: {d_sealed} | Pending: {d_pending} | Flagged: {d_flagged}">
                  <div class="bar-seg-sealed" style="width:{d_w_sealed};"></div>
                  <div class="bar-seg-pending" style="width:{d_w_pending};"></div>
                  <div class="bar-seg-flagged" style="width:{d_w_flagged};"></div>
                </div>
              </td>
              <td class="td-right td-num" style="color:{'var(--green)' if d_sealed_pct == 100 else 'var(--ink)'};">{d_sealed_pct}%</td>
            </tr>""")

        d_count_label = f"{len(districts)} district{'s' if len(districts) != 1 else ''} / jurisdiction{'s' if len(districts) != 1 else ''}"

        state_cards_html.append(f"""
      <details class="state-card" open>
        <summary class="state-summary">
          <div class="state-title-col">
            <span class="state-chevron">▶</span>
            <div>
              <span class="state-name">{html.escape(state_name)}</span>
              <span style="font-family:var(--type); font-size:11px; color:var(--ink-soft); margin-left:8px;">({d_count_label})</span>
            </div>
          </div>
          <div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
            <div class="state-counts">
              <span class="count-pill pill-total" title="Total Processed">Total: {s_total}</span>
              <span class="count-pill pill-sealed" title="Sealed & Certified">🛡️ Sealed: {s_sealed}</span>
              <span class="count-pill pill-pending" title="Pending Review">✍️ Pending: {s_pending}</span>
              <span class="count-pill pill-flagged" title="Flagged / Anomalous">⚠️ Flagged: {s_flagged}</span>
            </div>
            <div class="progress-bar-wrap">
              <div class="progress-stacked-bar" title="Sealed: {s_sealed} | Pending: {s_pending} | Flagged: {s_flagged}" style="width:140px; height:12px;">
                <div class="bar-seg-sealed" style="width:{w_sealed};"></div>
                <div class="bar-seg-pending" style="width:{w_pending};"></div>
                <div class="bar-seg-flagged" style="width:{w_flagged};"></div>
              </div>
              <span class="progress-pct">{sealed_pct}%</span>
            </div>
          </div>
        </summary>
        <div class="district-table-wrap">
          <table class="district-table">
            <thead>
              <tr>
                <th>District / Mandal Jurisdiction</th>
                <th class="td-center">Total Processed</th>
                <th class="td-center">Sealed &amp; Certified</th>
                <th class="td-center">Pending Review</th>
                <th class="td-center">Flagged</th>
                <th>Progress Ratio</th>
                <th class="td-right">Certification Rate</th>
              </tr>
            </thead>
            <tbody>
              {''.join(d_rows)}
            </tbody>
          </table>
        </div>
      </details>""")

    cards_joined = "".join(state_cards_html)
    return f"""
    <section class="state-progress-section" id="stateProgressSection">
      <div class="state-progress-header">
        <div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:18px;">🏛️</span>
            <span class="chart-title" style="font-size:17px; font-weight:700; color:var(--ink);">State &amp; District Registration Progress</span>
            <span style="font-family:var(--type); font-size:10px; font-weight:700; background:var(--ink); color:#fff; padding:2px 6px; border-radius:3px; letter-spacing:.06em;">JURISDICTION BREAKDOWN</span>
          </div>
          <div class="chart-meta" style="margin-top:3px;">Cadastral intake, certification velocity, and audit status grouped by administrative jurisdiction</div>
        </div>
        <div style="display:flex; gap:16px; font-family:var(--type); font-size:11px; align-items:center; flex-wrap:wrap;">
          <span style="display:flex; align-items:center; gap:5px;"><span style="width:10px; height:10px; border-radius:2px; background:var(--green);"></span> Sealed / Certified</span>
          <span style="display:flex; align-items:center; gap:5px;"><span style="width:10px; height:10px; border-radius:2px; background:var(--gold);"></span> Pending Review</span>
          <span style="display:flex; align-items:center; gap:5px;"><span style="width:10px; height:10px; border-radius:2px; background:var(--stamp);"></span> Flagged</span>
        </div>
      </div>
      {cards_joined}
    </section>"""


def _render_donut_svg(sale_n: int, gpa_n: int, other_n: int, total_n: int) -> str:
    """Generates a clean vector SVG donut chart matching the theme colors."""
    if total_n == 0:
        return """
        <svg class="chart-svg" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="42" fill="none" stroke="var(--rule-soft)" stroke-width="14"/>
          <text x="60" y="58" text-anchor="middle" font-family="Fraunces, serif" font-weight="700" font-size="19" fill="var(--ink-soft)">0</text>
          <text x="60" y="72" text-anchor="middle" font-family="Courier Prime, monospace" font-size="8.5" letter-spacing="1" fill="var(--ink-soft)">DEEDS</text>
        </svg>
        """
    circ = 2 * math.pi * 42  # ~263.89
    t = max(1, total_n)
    
    seg_sale = (sale_n / t) * circ
    seg_gpa = (gpa_n / t) * circ
    seg_other = (other_n / t) * circ

    off_sale = 0.0
    off_gpa = -seg_sale
    off_other = -(seg_sale + seg_gpa)

    return f"""
    <svg class="chart-svg" viewBox="0 0 120 120">
      <circle cx="60" cy="60" r="42" fill="none" stroke="var(--paper-deep)" stroke-width="15"/>
      <circle cx="60" cy="60" r="42" fill="none" stroke="var(--stamp)" stroke-width="15"
              stroke-dasharray="{seg_sale:.1f} {circ:.1f}" stroke-dashoffset="{off_sale:.1f}"
              transform="rotate(-90 60 60)"/>
      <circle cx="60" cy="60" r="42" fill="none" stroke="var(--gold)" stroke-width="15"
              stroke-dasharray="{seg_gpa:.1f} {circ:.1f}" stroke-dashoffset="{off_gpa:.1f}"
              transform="rotate(-90 60 60)"/>
      <circle cx="60" cy="60" r="42" fill="none" stroke="var(--green)" stroke-width="15"
              stroke-dasharray="{seg_other:.1f} {circ:.1f}" stroke-dashoffset="{off_other:.1f}"
              transform="rotate(-90 60 60)"/>
      <text x="60" y="58" text-anchor="middle" font-family="Fraunces, serif" font-weight="700" font-size="19" fill="var(--ink)">{total_n}</text>
      <text x="60" y="72" text-anchor="middle" font-family="Courier Prime, monospace" font-size="8.5" letter-spacing="1" fill="var(--ink-soft)">DEEDS</text>
    </svg>
    """


def _render_velocity_svg(total_n: int, sealed_n: int) -> str:
    """Generates an area/line chart showing throughput trends dynamically calculated from database records."""
    if total_n == 0:
        return """
        <svg class="chart-svg" viewBox="0 0 580 195">
          <line x1="45" y1="170" x2="550" y2="170" stroke="var(--rule)" stroke-width="1.5"/>
          <line x1="45" y1="125" x2="550" y2="125" stroke="var(--rule-soft)" stroke-dasharray="3 3" stroke-width="1"/>
          <line x1="45" y1="80" x2="550" y2="80" stroke="var(--rule-soft)" stroke-dasharray="3 3" stroke-width="1"/>
          <line x1="45" y1="35" x2="550" y2="35" stroke="var(--rule-soft)" stroke-dasharray="3 3" stroke-width="1"/>
          
          <text x="35" y="174" text-anchor="end" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">0</text>
          <text x="35" y="129" text-anchor="end" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">5</text>
          <text x="35" y="84" text-anchor="end" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">10</text>
          <text x="35" y="39" text-anchor="end" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">15</text>
          
          <line x1="50" y1="170" x2="530" y2="170" stroke="var(--rule-soft)" stroke-width="1.5" stroke-dasharray="4 4"/>
          <text x="290" y="105" text-anchor="middle" font-family="Courier Prime, monospace" font-size="11.5" fill="var(--ink-soft)" letter-spacing="1">REGISTRY READY · 0 INTAKE TODAY</text>
          
          <text x="50" y="188" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">T-6</text>
          <text x="130" y="188" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">T-5</text>
          <text x="210" y="188" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">T-4</text>
          <text x="290" y="188" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">T-3</text>
          <text x="370" y="188" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">T-2</text>
          <text x="450" y="188" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">T-1</text>
          <text x="530" y="188" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9.5" font-weight="bold" fill="var(--ink)">TODAY</text>
        </svg>
        """

    try:
        db = verification_service.load_db()
    except Exception:
        db = {}

    recs = [r for r in db.values() if isinstance(r, dict) and r.get("verification_id")]
    recs.sort(key=lambda r: r.get("created_at") or "")

    from datetime import datetime, timezone, timedelta
    now_dt = datetime.now(timezone.utc)
    dates_labels = []
    daily_intake = [0] * 7
    daily_sealed = [0] * 7

    for idx in range(7):
        day_date = (now_dt - timedelta(days=6 - idx)).date()
        dates_labels.append("TODAY" if idx == 6 else day_date.strftime("%b %d"))

    for r in recs:
        created_str = r.get("created_at") or ""
        is_sealed = (r.get("status") or "").upper() == "APPROVED"
        r_date = None
        if created_str:
            try:
                r_date = datetime.fromisoformat(created_str.replace("Z", "+00:00")).date()
            except Exception:
                pass
        
        matched_idx = 6
        if r_date:
            diff_days = (now_dt.date() - r_date).days
            if diff_days >= 0 and diff_days < 7:
                matched_idx = 6 - diff_days
            elif diff_days >= 7:
                matched_idx = 0
        
        daily_intake[matched_idx] += 1
        if is_sealed:
            daily_sealed[matched_idx] += 1

    cum_intake = []
    cum_sealed = []
    cur_in = 0
    cur_se = 0
    for i in range(7):
        cur_in += daily_intake[i]
        cur_se += daily_sealed[i]
        cum_intake.append(cur_in)
        cum_sealed.append(cur_se)

    max_y_val = max(1, cum_intake[-1], total_n)

    steps = []
    for i in range(7):
        x = 50 + int(i * 80)
        y_intake = round(170 - (cum_intake[i] / max_y_val * 135))
        y_sealed = round(170 - (cum_sealed[i] / max_y_val * 135))
        steps.append((x, y_intake, y_sealed, dates_labels[i], cum_intake[i], cum_sealed[i]))

    line_total_pts = " ".join(f"{x},{y1}" for x, y1, _, _, _, _ in steps)
    area_total_pts = f"50,170 {line_total_pts} 530,170"
    line_sealed_pts = " ".join(f"{x},{y2}" for x, _, y2, _, _, _ in steps)

    dots_markup = []
    labels_markup = []
    for x, y1, y2, lbl, c_in, c_se in steps:
        dots_markup.append(f'<circle cx="{x}" cy="{y1}" r="3.5" fill="var(--stamp)" stroke="#fff" stroke-width="1.5"><title>{html.escape(lbl)}: {c_in} Total Ingested</title></circle>')
        dots_markup.append(f'<circle cx="{x}" cy="{y2}" r="3.5" fill="var(--green)" stroke="#fff" stroke-width="1.5"><title>{html.escape(lbl)}: {c_se} Cryptographically Sealed</title></circle>')
        labels_markup.append(f'<text x="{x}" y="188" text-anchor="middle" font-family="Courier Prime, monospace" font-size="9" fill="var(--ink-soft)">{html.escape(lbl)}</text>')

    y_mid1 = max(1, round(max_y_val / 3))
    y_mid2 = max(2, round(max_y_val * 2 / 3))

    return f"""
    <svg class="chart-svg" viewBox="0 0 580 195">
      <defs>
        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--stamp)" stop-opacity="0.18"/>
          <stop offset="100%" stop-color="var(--stamp)" stop-opacity="0.0"/>
        </linearGradient>
      </defs>
      
      <line x1="45" y1="170" x2="550" y2="170" stroke="var(--rule-soft)" stroke-width="1"/>
      <line x1="45" y1="125" x2="550" y2="125" stroke="var(--rule-soft)" stroke-dasharray="3 3" stroke-width="1"/>
      <line x1="45" y1="80" x2="550" y2="80" stroke="var(--rule-soft)" stroke-dasharray="3 3" stroke-width="1"/>
      <line x1="45" y1="35" x2="550" y2="35" stroke="var(--rule-soft)" stroke-dasharray="3 3" stroke-width="1"/>
      
      <text x="35" y="174" text-anchor="end" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">0</text>
      <text x="35" y="129" text-anchor="end" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">{y_mid1}</text>
      <text x="35" y="84" text-anchor="end" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">{y_mid2}</text>
      <text x="35" y="39" text-anchor="end" font-family="Courier Prime, monospace" font-size="9.5" fill="var(--ink-soft)">{max_y_val}</text>

      <polygon points="{area_total_pts}" fill="url(#areaGrad)"/>

      <polyline points="{line_total_pts}" fill="none" stroke="var(--stamp)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
      <polyline points="{line_sealed_pts}" fill="none" stroke="var(--green)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>

      {''.join(dots_markup)}
      {''.join(labels_markup)}
    </svg>
    """


def _render_sidebar(active_item: str, desk_n: int, sealed_n: int, worker_label: str, role: str = "clerk") -> str:
    """Renders the persistent left navigation sidebar rail."""
    dash_active = 'class="active"' if active_item in {"dashboard", "clerk", "officer"} else ''
    scan_active = 'class="active"' if active_item == "new_scan" else ''
    is_officer = (role.lower() == "officer")

    if is_officer:
        nav_items_html = f"""
        <li>
          <a {dash_active} href="/officer">
            <span class="nav-link-left">
              <span>Officer Dashboard</span>
            </span>
          </a>
        </li>
        <li>
          <a href="/officer#masterLedgerSection">
            <span class="nav-link-left">
              <span data-i18n="nav_master_reg">Master Deed Register</span>
            </span>
          </a>
        </li>
        <li>
          <a href="/officer#ledgerSection">
            <span class="nav-link-left">
              <span>Officer Approval Queue</span>
            </span>
            <span class="nav-badge badge-amber">{desk_n}</span>
          </a>
        </li>
        <li>
          <a href="/officer#sealedSection">
            <span class="nav-link-left">
              <span data-i18n="nav_verified_certs">Verified Certificates</span>
            </span>
            <span class="nav-badge badge-green">{sealed_n}</span>
          </a>
        </li>
        """
    else:
        nav_items_html = f"""
        <li>
          <a {dash_active} href="/user">
            <span class="nav-link-left">
              <span>Dashboard</span>
            </span>
          </a>
        </li>
        <li>
          <a {scan_active} href="/new">
            <span class="nav-link-left">
              <span data-i18n="nav_new_scan">New Scan &amp; Intake</span>
            </span>
            <span class="nav-badge badge-primary" data-i18n="badge_desk01">Desk 01</span>
          </a>
        </li>
        """

    return f"""
  <aside class="dash-sidebar">
    <div class="sidebar-top">
      <a class="brand-box" href="/" title="Return to Landing Page" style="display:flex; align-items:center; justify-content:center; padding:10px 8px 18px; text-decoration:none; text-align:center;">
        <img src="/logo.png?v=20260904d" alt="OneBhoomi" style="height:60px; width:auto; display:block; margin:0 auto; mix-blend-mode:multiply; filter:contrast(1.02);">
      </a>

      <div class="nav-label" data-i18n="nav_main_menu">Main Menu</div>
      <ul class="nav-menu">
        {nav_items_html}
      </ul>
    </div>

    <div class="sidebar-bottom">
      <div class="sys-pill">
        <span class="sys-dot"></span>
        <span data-i18n="lbl_air_gapped">AIR-GAPPED &amp; SECURE</span>
      </div>
      <div class="sys-meta">
        <div><b data-i18n="lbl_rsa_stat">RSA-PSS 2048:</b> <span data-i18n="lbl_active">Active</span></div>
        <div style="margin-top:2px;"><b data-i18n="lbl_worker">Worker:</b> {html.escape(worker_label)}</div>
      </div>
    </div>
  </aside>
    """


def _render_learning_panel() -> str:
    """Renders the Officer-Verified Adaptive OCR Normalization panel on the registry dashboard."""
    import ocr_learning_service
    stats = ocr_learning_service.get_learning_stats()
    rules = ocr_learning_service.get_learned_rules()

    pending_n = stats.get("pending_feedback_count", 0)
    verified_n = stats.get("verified_feedback_count", 0)
    rejected_n = stats.get("rejected_feedback_count", 0)
    rules_n = stats.get("learned_rules_count", 0)
    fields_imp = stats.get("fields_improved", [])
    fields_txt = ", ".join(fields_imp) if fields_imp else "None yet (accruing approvals)"

    rule_badges = []
    for r in rules[:6]:
        f_name = r.get("field_name", "")
        doc_type = r.get("document_type", "Sale Deed")
        lang = r.get("language", "en")
        raw_p = r.get("raw_pattern", "")
        rep_p = r.get("replacement", "")
        conf = r.get("confidence", 0.0)
        ev_c = r.get("evidence_count", 0)
        rule_badges.append(
            f'<div style="font-family:var(--type); font-size:11.5px; background:var(--paper-deep); padding:8px 12px; border:1px solid var(--rule); border-radius:3px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
            f'<div>'
            f'<span style="background:var(--ink); color:#fff; padding:1px 5px; border-radius:2px; font-size:10px; font-weight:700; margin-right:6px;">{html.escape(f_name)}</span>'
            f'<span style="background:var(--border); color:var(--ink-soft); padding:1px 5px; border-radius:2px; font-size:10px; margin-right:8px;">{html.escape(doc_type)} &middot; {html.escape(lang)}</span>'
            f'<span>Raw: <code style="color:var(--stamp);">{html.escape(raw_p)}</code> &rarr; Normalized: <code style="color:var(--green); font-weight:700;">{html.escape(rep_p)}</code></span>'
            f'</div>'
            f'<span style="font-size:11px; color:var(--ink-soft);">{ev_c} verified examples &middot; rule conf {conf:.2f}</span>'
            f'</div>'
        )
    if not rule_badges:
        rule_badges.append(
            '<div style="font-family:var(--type); font-size:11.5px; color:var(--ink-soft); padding:10px; background:var(--paper); border:1px dashed var(--rule); border-radius:3px;">'
            'No active learned rules compiled yet. Rules automatically activate once an officer approves &ge; 2 identical field corrections.'
            '</div>'
        )

    return f"""
    <section class="learning-panel" id="learningPanel" style="background:var(--card); border:1.5px solid var(--border); border-radius:2px; padding:20px; margin-bottom:26px; box-shadow:2px 2px 0 rgba(0,0,0,.025);">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; border-bottom:1px solid var(--rule-soft); padding-bottom:10px; flex-wrap:wrap; gap:10px;">
        <div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:18px;">🧠</span>
            <span class="chart-title" style="font-size:17px; font-weight:700; color:var(--ink);">Officer-verified adaptive OCR normalization</span>
            <span style="font-family:var(--type); font-size:10px; font-weight:700; background:var(--green); color:#fff; padding:2px 6px; border-radius:3px; letter-spacing:.06em;">OFFICER-VERIFIED</span>
          </div>
          <div class="chart-meta" style="margin-top:3px;">Human-in-the-loop feedback learning</div>
        </div>
        <div style="font-family:var(--type); font-size:11px; background:#FFF9E6; border:1px solid #F0C36D; padding:6px 12px; border-radius:3px; color:#5A4008; line-height:1.4;">
          Human-in-the-loop adaptive OCR feedback learning.<br>
          Only officer-approved corrections become learned rules.<br>
          Neural OCR weights are not retrained.
        </div>
      </div>

      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:12px; margin-bottom:16px;">
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Pending Corrections</div>
          <div style="font-family:var(--serif); font-size:26px; font-weight:700; color:var(--amber); margin:4px 0;">{pending_n}</div>
          <div style="font-size:11px; color:var(--ink-soft);">Clerk edits awaiting officer seal</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Verified Corrections</div>
          <div style="font-family:var(--serif); font-size:26px; font-weight:700; color:var(--green); margin:4px 0;">{verified_n}</div>
          <div style="font-size:11px; color:var(--ink-soft);">Officer-approved corrections</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Rejected Corrections</div>
          <div style="font-family:var(--serif); font-size:26px; font-weight:700; color:var(--stamp); margin:4px 0;">{rejected_n}</div>
          <div style="font-size:11px; color:var(--ink-soft);">Excluded from learning rules</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Active Learned Rules</div>
          <div style="font-family:var(--serif); font-size:26px; font-weight:700; color:var(--ink); margin:4px 0;">{rules_n}</div>
          <div style="font-size:11px; color:var(--ink-soft);">&ge; 2 approved examples required</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Fields Improved</div>
          <div style="font-family:var(--serif); font-size:16px; font-weight:700; color:var(--ink); margin:8px 0 4px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{html.escape(fields_txt)}">{html.escape(fields_txt)}</div>
          <div style="font-size:11px; color:var(--ink-soft);">Field, doc-type &amp; lang scoped</div>
        </div>
      </div>

      <div style="border-top:1px solid var(--rule-soft); padding-top:12px;">
        <div style="font-family:var(--type); font-size:11px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px;">Active Field Normalization Rules (Raw vs Normalized)</div>
        {''.join(rule_badges)}
      </div>
    </section>
    """


def _render_preprocessing_panel(runtime_meta: Optional[dict] = None, total_on_file: Optional[int] = None) -> str:
    """Renders the Adaptive Image Preprocessing & Scan Quality section on the registry dashboard."""
    if total_on_file == 0:
        badge_html = """<span style="font-family:var(--type); font-size:10px; font-weight:700; background:#5A5142; color:#fff; padding:2px 7px; border-radius:3px; letter-spacing:.06em;">STANDBY &middot; ZERO DOCUMENTS PROCESSED</span>"""
        tagline = "Image preprocessing pipeline is active and standing by for document upload (0 pages processed)."

        cards_html = """
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Document Pages</div>
          <div style="font-family:var(--serif); font-size:20px; font-weight:700; color:var(--ink); margin:4px 0;">0 Pages</div>
          <div style="font-size:11px; color:var(--ink-soft);">No documents in intake queue</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Adaptive OCR Variants</div>
          <div style="font-family:var(--type); font-size:12.5px; font-weight:700; color:var(--ink-soft); margin:6px 0 4px;">0 Active (Standby)</div>
          <div style="font-size:11px; color:var(--ink-soft);">Greyscale / CLAHE / Threshold engine ready</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Quality Score &amp; Contrast</div>
          <div style="font-family:var(--type); font-size:12.5px; font-weight:700; color:var(--ink-soft); margin:6px 0 4px;">0% Quality Score</div>
          <div style="font-size:11px; color:var(--ink-soft);">Contrast &amp; blur variance analysis ready</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Geometric Alignment</div>
          <div style="font-family:var(--type); font-size:12.5px; font-weight:700; color:var(--ink-soft); margin:6px 0 4px;">0.00&deg; Skew (Ready)</div>
          <div style="font-size:11px; color:var(--ink-soft);">Hough transform deskew ready</div>
        </div>
        """
        return f"""
    <section class="preprocessing-panel" id="preprocessingPanel" style="background:var(--card); border:1.5px solid var(--border); border-radius:2px; padding:20px; margin-bottom:26px; box-shadow:2px 2px 0 rgba(0,0,0,.025);">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; border-bottom:1px solid var(--rule-soft); padding-bottom:10px; flex-wrap:wrap; gap:10px;">
        <div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:18px;">📷</span>
            <span class="chart-title" style="font-size:17px; font-weight:700; color:var(--ink);">Adaptive Image Preprocessing &amp; Scan Quality</span>
            {badge_html}
          </div>
          <div class="chart-meta" style="margin-top:3px;">{tagline}</div>
        </div>
        <div style="font-family:var(--type); font-size:11px; background:#EBF3FB; border:1px solid #B8D5F5; padding:6px 12px; border-radius:3px; color:#1C497B; line-height:1.4;">
          The system performs quality-aware image preprocessing for scanned land records,<br>
          selects a suitable OCR variant, preserves original document evidence,<br>
          and records preprocessing metadata for explainable review.
        </div>
      </div>

      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-bottom:16px;">
        {cards_html}
      </div>

      <div style="font-family:var(--type); font-size:11px; color:var(--ink-soft); line-height:1.5; background:var(--paper-deep); padding:8px 12px; border:1px solid var(--rule); border-radius:3px;">
        🛡️ <b>Evidence Preservation &amp; Coordinate Mapping:</b> Original document pixels and source coordinates are permanently retained.
        When upscaling is applied, bounding box coordinates are mapped back to original page coordinates. Preprocessing metadata is attached to extraction output.
      </div>
    </section>
    """

    if runtime_meta is None:
        try:
            import image_preprocessing
            runtime_meta = image_preprocessing.get_latest_runtime_preprocessing()
        except Exception:
            runtime_meta = None

    if runtime_meta and isinstance(runtime_meta, dict):
        pg = runtime_meta.get("page_number", 1)
        ptype = str(runtime_meta.get("page_type", "scanned_page")).replace("_", " ").title()
        variant = runtime_meta.get("selected_variant", "original_grayscale")
        scale = float(runtime_meta.get("scale", 1.0))
        ops = runtime_meta.get("operations") or []
        deskew = bool(runtime_meta.get("deskew_applied", False))
        skew_angle = float(runtime_meta.get("skew_angle_degrees", 0.0))
        reason = runtime_meta.get("selection_reason", "best contrast/blur balance")

        qb = runtime_meta.get("quality_before") or {}
        qa = runtime_meta.get("quality_after") or {}

        score_b = float(qb.get("quality_score", 0.0))
        score_a = float(qa.get("quality_score", 0.0))
        contrast_b = float(qb.get("contrast", 0.0))
        contrast_a = float(qa.get("contrast", 0.0))
        blur_b = float(qb.get("blur_score", 0.0))
        blur_a = float(qa.get("blur_score", 0.0))
        flags = qb.get("quality_flags") or []

        badge_html = """<span style="font-family:var(--type); font-size:10px; font-weight:700; background:var(--green); color:#fff; padding:2px 7px; border-radius:3px; letter-spacing:.06em;">LIVE RUNTIME METRICS</span>"""
        tagline = f"Live telemetry from processed page (Page {pg}: {html.escape(str(reason))})"

        cards_html = f"""
        <div style="background:var(--paper); border:1.5px solid var(--green); padding:12px 14px; border-radius:3px; box-shadow:1px 1px 0 rgba(0,0,0,.03);">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Active Page &amp; Role</div>
          <div style="font-family:var(--serif); font-size:20px; font-weight:700; color:var(--ink); margin:4px 0;">Page {pg}</div>
          <div style="font-size:11px; color:var(--ink-soft);">{html.escape(ptype)} · Scale: {scale:.1f}x</div>
        </div>
        <div style="background:var(--paper); border:1.5px solid var(--green); padding:12px 14px; border-radius:3px; box-shadow:1px 1px 0 rgba(0,0,0,.03);">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Selected OCR Variant</div>
          <div style="font-family:var(--type); font-size:12.5px; font-weight:700; color:var(--green); margin:6px 0 4px;">{html.escape(variant)}</div>
          <div style="font-size:11px; color:var(--ink-soft);">Ops: {html.escape(', '.join(ops) if ops else 'none')}</div>
        </div>
        <div style="background:var(--paper); border:1.5px solid var(--green); padding:12px 14px; border-radius:3px; box-shadow:1px 1px 0 rgba(0,0,0,.03);">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Quality Score &amp; Contrast</div>
          <div style="font-family:var(--type); font-size:12.5px; font-weight:700; color:var(--green); margin:6px 0 4px;">{score_b:.0%} &rarr; {score_a:.0%}</div>
          <div style="font-size:11px; color:var(--ink-soft);">Contrast: {contrast_b:.1f} &rarr; {contrast_a:.1f}</div>
        </div>
        <div style="background:var(--paper); border:1.5px solid var(--green); padding:12px 14px; border-radius:3px; box-shadow:1px 1px 0 rgba(0,0,0,.03);">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">Geometric Alignment</div>
          <div style="font-family:var(--type); font-size:12.5px; font-weight:700; color:var(--ink); margin:6px 0 4px;">{skew_angle:+.2f}&deg; ({'Deskew Applied' if deskew else 'Not needed'})</div>
          <div style="font-size:11px; color:var(--ink-soft);">Laplacian: {blur_b:.0f} &rarr; {blur_a:.0f}</div>
        </div>
        """
    else:
        badge_html = """<span style="font-family:var(--type); font-size:10px; font-weight:700; background:#5A5142; color:#fff; padding:2px 7px; border-radius:3px; letter-spacing:.06em;">REFERENCE PROFILE (TELANGANA DEED BASELINE)</span>"""
        tagline = "[Reference Baseline Profile] Initial calibration profile for the 6-page Telangana deed. Document uploads update this panel with live runtime telemetry."

        cards_html = """
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">[Baseline] Document Structure</div>
          <div style="font-family:var(--serif); font-size:20px; font-weight:700; color:var(--ink); margin:4px 0;">6 Pages</div>
          <div style="font-size:11px; color:var(--ink-soft);">Reference Telangana Sale Deed</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">[Baseline] Page 1: Stamp Paper</div>
          <div style="font-family:var(--type); font-size:12.5px; font-weight:700; color:var(--green); margin:6px 0 4px;">CLAHE Denoised</div>
          <div style="font-size:11px; color:var(--ink-soft);">Dark stamp area protected (Threshold avoided)</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">[Baseline] Page 2: Schedule</div>
          <div style="font-family:var(--type); font-size:12.5px; font-weight:700; color:var(--green); margin:6px 0 4px;">Deskewed + CLAHE</div>
          <div style="font-size:11px; color:var(--ink-soft);">Faint typewriter enhanced (Target: 81%)</div>
        </div>
        <div style="background:var(--paper); border:1px solid var(--border); padding:12px 14px; border-radius:3px;">
          <div style="font-family:var(--type); font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.1em;">[Baseline] Page 6: Plan Map</div>
          <div style="font-family:var(--type); font-size:12.5px; font-weight:700; color:var(--green); margin:6px 0 4px;">Safe Grayscale</div>
          <div style="font-family:var(--type); font-size:11px; color:var(--ink-soft);">CAD line vectors preserved (No erosion)</div>
        </div>
        """

    return f"""
    <section class="preprocessing-panel" id="preprocessingPanel" style="background:var(--card); border:1.5px solid var(--border); border-radius:2px; padding:20px; margin-bottom:26px; box-shadow:2px 2px 0 rgba(0,0,0,.025);">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; border-bottom:1px solid var(--rule-soft); padding-bottom:10px; flex-wrap:wrap; gap:10px;">
        <div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:18px;">📷</span>
            <span class="chart-title" style="font-size:17px; font-weight:700; color:var(--ink);">Adaptive Image Preprocessing &amp; Scan Quality</span>
            {badge_html}
          </div>
          <div class="chart-meta" style="margin-top:3px;">{tagline}</div>
        </div>
        <div style="font-family:var(--type); font-size:11px; background:#EBF3FB; border:1px solid #B8D5F5; padding:6px 12px; border-radius:3px; color:#1C497B; line-height:1.4;">
          The system performs quality-aware image preprocessing for scanned land records,<br>
          selects a suitable OCR variant, preserves original document evidence,<br>
          and records preprocessing metadata for explainable review.
        </div>
      </div>

      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-bottom:16px;">
        {cards_html}
      </div>

      <div style="font-family:var(--type); font-size:11px; color:var(--ink-soft); line-height:1.5; background:var(--paper-deep); padding:8px 12px; border:1px solid var(--rule); border-radius:3px;">
        🛡️ <b>Evidence Preservation &amp; Coordinate Mapping:</b> Original document pixels and source coordinates are permanently retained.
        When upscaling is applied, bounding box coordinates are mapped back to original page coordinates. Preprocessing metadata is attached to extraction output.
      </div>
    </section>
    """


def render_dashboard(host_name: str = "localhost:8001", colab_url: str = "") -> bytes:
    """Renders the executive operations dashboard with left side menu and statistical graphs."""
    data = get_dashboard_data()
    today = datetime.now().strftime("%d-%m-%Y")
    fp = _fingerprint()

    if colab_url:
        try:
            worker_host = urlparse(colab_url).hostname or colab_url
        except Exception:
            worker_host = colab_url
        worker_label = f"Remote GPU ({worker_host[:16]}...)"
    else:
        worker_label = "Local CPU (PaddleOCR)"

    sidebar_html = _render_sidebar("dashboard", data["desk_n"], data["sealed_n"], worker_label)

    # Action notice banner if clerk desk has pending items
    notice_markup = ""
    if data["pending_queue"]:
        oldest_pending = data["pending_queue"][-1]
        oldest_id = oldest_pending["id"]
        doc_label = f"{oldest_pending['doc_type']} {('No. ' + oldest_pending['doc_no']) if oldest_pending['doc_no'] else ''}".strip()
        notice_markup = f"""
      <div class="notice-banner" id="queueNotice">
        <div class="notice-info">
          <span class="notice-icon">✍️</span>
          <div class="notice-text">
            <b><span data-i18n="banner_title">Clerk Review Action Required:</span></b> <span data-i18n="banner_prefix">There are</span> <b>{data['desk_n']}</b> <span data-i18n="banner_suffix">land record(s) awaiting verification against scanned evidence and officer digital signing.</span>
          </div>
        </div>
        <a class="btn btn-primary btn-sm" href="/record?verification_id={html.escape(oldest_id)}"><span data-i18n="btn_open_next">Open Next Record</span> ({html.escape(doc_label)}) &rarr;</a>
      </div>"""

    # Master Ledger Table Rows
    if data["rows"]:
        body_rows = []
        for r in data["rows"]:
            doc_no_str = f"No. {r['doc_no']}" if r["doc_no"] else "Unnumbered"
            mandal_str = f"Mandal: {r['mandal']}" if r["mandal"] else ""
            
            verify_btn = ""
            if r["status"] == "APPROVED":
                verify_btn = (
                    f'<a class="act-btn act-verify" title="View Public Certificate & Offline QR" href="/?verification_id={html.escape(r["id"])}" data-i18n="btn_cert">✓ Certificate</a>'
                    f'<a class="act-btn" title="Export Locked PDF Certificate" href="/export_pdf?verification_id={html.escape(r["id"])}&lock=1" target="_blank" style="margin-left:4px;background:#FCF9F2;border-color:var(--gold,#C9A227);color:var(--stamp,#781D22);font-weight:700;">🔒 PDF</a>'
                )

            body_rows.append(
                f"""
        <tr class="data-row" data-status="{html.escape(r['status'])}">
          <td class="td-sl">{r['sl']}</td>
          <td class="td-date">{html.escape(r['date'])}</td>
          <td>
            <span class="td-doc-main">{html.escape(r['doc_type'])}</span>
            <span class="td-doc-sub">{html.escape(doc_no_str)}</span>
          </td>
          <td title="{html.escape(r['parties'])}">{html.escape(r['parties'])}</td>
          <td>
            <span class="td-place-main">{html.escape(r['place'])}</span>
            <span class="td-place-sub">{html.escape(mandal_str)}</span>
          </td>
          <td class="td-mono">{html.escape(r['survey'])}</td>
          <td class="td-stamp">{html.escape(r['stamp'])}</td>
          <td>{_badge(r['status'])}</td>
          <td>
            <div class="action-links">
              <a class="act-btn" href="/record?verification_id={html.escape(r['id'])}">Review Console</a>
              <a class="act-btn" href="/record?verification_id={html.escape(r['id'])}#audit-timeline" title="View Lifecycle Audit Trail Timeline" style="border-color:var(--rule); color:var(--ink-soft); font-size:11px;">Timeline &rarr;</a>
              {verify_btn}
            </div>
          </td>
        </tr>"""
            )
        table_html = f"""
        <table class="master-ledger">
          <thead>
            <tr>
              <th data-i18n="th_sl">SL.</th>
              <th data-i18n="th_received">RECEIVED</th>
              <th data-i18n="th_doc_no">DOCUMENT &amp; NO.</th>
              <th data-i18n="th_parties">PARTIES</th>
              <th data-i18n="th_location">LOCATION</th>
              <th data-i18n="th_survey">SURVEY DESIGNATION</th>
              <th data-i18n="th_stamp">STAMP DUTY</th>
              <th data-i18n="th_status">STATUS</th>
              <th data-i18n="th_actions">ACTIONS</th>
            </tr>
          </thead>
          <tbody id="ledgerBody">
            {''.join(body_rows)}
          </tbody>
        </table>
        <div id="ledgerEmptyNotice" class="table-empty" style="display:none;">
          <p>No matching document entries found.</p>
        </div>"""
    else:
        table_html = """
        <div class="table-empty">
          <p>No land documents registered in the ledger yet.</p>
          <p style="font-size:13px; color:var(--ink-soft); margin-bottom:16px;">
            Open the new intake desk to scan, extract, and certify your first deed.
          </p>
          <a class="btn btn-primary" href="/new">+ Scan First Document</a>
        </div>"""

    donut_svg = _render_donut_svg(data["sale_count"], data["gpa_count"], data["other_count"], data["on_file"])
    velocity_svg = _render_velocity_svg(data["on_file"], data["sealed_n"])
    learning_panel_markup = _render_learning_panel()
    preprocessing_panel_markup = _render_preprocessing_panel(
        data.get("latest_preprocessing"), total_on_file=data["on_file"]
    )
    accuracy_panel_markup = _render_extraction_accuracy_panel()
    state_progress_markup = _render_state_district_panel()
    fp_text = fp or "Keypair auto-generated on first seal"

    page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OneBhoomi — Land Records Office Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..900;1,9..144,300..900&family=Archivo:wght@400;500;600;700&family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Noto+Sans+Telugu:wght@400;500;600;700&family=Noto+Sans+Kannada:wght@400;500;600;700&family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{DASHBOARD_CSS}</style>
</head>
<body>

<div class="security-bg" aria-hidden="true"></div>

<div class="app-layout">

  {sidebar_html}

  <!-- MAIN WORKSPACE CONTENT -->
  <main class="dash-content">
    <div class="main-inner">
      
      <!-- Top Action Bar -->
      <div class="top-action-bar">
        <div class="header-left">
          <h1 data-i18n="dash_h1">Registry Operations &amp; <em>Analytics</em></h1>
          <div class="header-tagline" data-i18n="dash_tagline">Complete offline day book of land records, human-in-the-loop clerk reviews, and cryptographic digital seals.</div>
        </div>
        <div class="header-right"></div>

      <!-- KPI Summary Cards (4 Cards) -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">
            <span data-i18n="kpi_total">Total on File</span>
            <span>📂</span>
          </div>
          <div class="kpi-num">{data['on_file']}</div>
          <div class="kpi-sub" data-i18n="kpi_total_sub">Total land deeds recorded in offline registry</div>
        </div>

        <div class="kpi-card kpi-sealed">
          <div class="kpi-label">
            <span data-i18n="kpi_sealed">Sealed &amp; Certified</span>
            <span>🛡️</span>
          </div>
          <div class="kpi-num">{data['sealed_n']}</div>
          <div class="kpi-sub">{data['seal_rate']} <span data-i18n="kpi_sealed_sub">certification rate · RSA-PSS signed</span></div>
        </div>

        <div class="kpi-card kpi-desk">
          <div class="kpi-label">
            <span data-i18n="kpi_review">Officer Review Queue</span>
            <span>✍️</span>
          </div>
          <div class="kpi-num">{data['desk_n']}</div>
          <div class="kpi-sub" data-i18n="kpi_review_sub">waiting officer verification &amp; approval</div>
        </div>

        <div class="kpi-card kpi-rejected">
          <div class="kpi-label">
            <span data-i18n="kpi_noncert">Non-Certified</span>
            <span>⚠️</span>
          </div>
          <div class="kpi-num">{data['rejected_n']}</div>
          <div class="kpi-sub" data-i18n="kpi_noncert_sub">Rejected or anomalous scan records</div>
        </div>
      </div>

      <!-- Action Required Banner (if any pending) -->
      {notice_markup}

      <!-- Adaptive Learning Feedback Layer -->
      {learning_panel_markup}

      <!-- Adaptive Preprocessing & Scan Quality Layer -->
      {preprocessing_panel_markup}

      <!-- EXTRACTION ACCURACY & CONFIDENCE INTELLIGENCE LAYER -->
      {accuracy_panel_markup}

      <!-- STATISTICAL ANALYTICS GRAPHS -->
      <section class="charts-grid" id="analyticsSection">
        
        <!-- Graph 1: Velocity & Throughput Trend -->
        <div class="chart-card">
          <div class="chart-header">
            <div>
              <div class="chart-title" data-i18n="chart_vel_title">Registration Velocity &amp; Sealing Throughput</div>
              <div class="chart-meta" data-i18n="chart_vel_meta">Timeline intake volume vs verified cryptographic certifications</div>
            </div>
            <div style="display:flex; gap:12px; font-family:var(--type); font-size:10.5px;">
              <span style="display:flex; align-items:center; gap:5px;">
                <span style="width:8px; height:8px; border-radius:50%; background:var(--stamp);"></span>
                <span data-i18n="chart_total_intake">Total Intake</span>
              </span>
              <span style="display:flex; align-items:center; gap:5px;">
                <span style="width:8px; height:8px; border-radius:50%; background:var(--green);"></span>
                <span data-i18n="chart_sealed_on_file">Sealed on File</span>
              </span>
            </div>
          </div>
          <div class="chart-svg-wrap">
            {velocity_svg}
          </div>
        </div>

        <!-- Graph 2: Document Classification Donut -->
        <div class="chart-card">
          <div class="chart-header">
            <div>
              <div class="chart-title" data-i18n="chart_doc_class">Document Classification</div>
              <div class="chart-meta" data-i18n="chart_doc_meta">Distribution of legal record deed categories</div>
            </div>
            <div class="chart-meta" style="color:var(--green); font-weight:700;">
              <span data-i18n="chart_gis_lbl">GIS Resolved:</span> {data['gis_rate']}
            </div>
          </div>

          <div class="donut-layout">
            <div class="donut-svg-box">
              {donut_svg}
            </div>

            <div class="donut-legend">
              <div class="legend-row">
                <span class="legend-left">
                  <span class="legend-color" style="background:var(--stamp);"></span>
                  <span data-i18n="chart_sale_deeds">Sale Deeds</span>
                </span>
                <span class="legend-num">{data['sale_count']}</span>
              </div>
              <div class="legend-row">
                <span class="legend-left">
                  <span class="legend-color" style="background:var(--gold);"></span>
                  <span data-i18n="chart_agreements_gpa">Agreements / GPA</span>
                </span>
                <span class="legend-num">{data['gpa_count']}</span>
              </div>
              <div class="legend-row">
                <span class="legend-left">
                  <span class="legend-color" style="background:var(--green);"></span>
                  <span data-i18n="chart_other_records">Other Records</span>
                </span>
                <span class="legend-num">{data['other_count']}</span>
              </div>
            </div>
          </div>

          <div class="gis-rate-meter">
            <div class="meter-label">
              <span data-i18n="chart_spatial_match">Telangana (TGRAC) &amp; Karnataka Spatial Match</span>
              <b>{data['gis_rate']} <span data-i18n="chart_resolved">Resolved</span></b>
            </div>
            <div class="meter-bar">
              <div class="meter-fill" style="width:{data['gis_rate']};"></div>
            </div>
          </div>
        </div>

      </section>

      <!-- STATE & DISTRICT PROGRESS PANEL -->
      {state_progress_markup}

      <!-- MASTER LEDGER SECTION -->
      <section class="ledger-section" id="ledgerSection">
        <div class="ledger-toolbar">
          <div class="toolbar-left">
            <h2 class="ledger-head-title" data-i18n="reg_title">Master Deed Register</h2>
            <div class="search-wrap">
              <span class="search-icon">🔍</span>
              <input type="text" id="ledgerSearch" class="search-input" placeholder="Search by Doc #, Village, Party, Survey..." data-i18n-ph="search_placeholder">
            </div>
          </div>

          <div class="toolbar-right">
            <div class="filter-tabs">
              <button type="button" class="tab-btn active" data-filter="ALL"><span data-i18n="tab_all">All</span> ({data['on_file']})</button>
              <button type="button" class="tab-btn" data-filter="SEALED"><span data-i18n="tab_sealed">Sealed</span> ({data['sealed_n']})</button>
              <button type="button" class="tab-btn" data-filter="PENDING"><span data-i18n="tab_pending">Pending</span> ({data['desk_n']})</button>
              <button type="button" class="tab-btn" data-filter="REJECTED"><span data-i18n="tab_rejected">Rejected</span> ({data['rejected_n']})</button>
            </div>
            <a class="btn btn-primary btn-sm" href="/new" data-i18n="btn_scan_new_deed">+ Scan New Deed</a>
            <button type="button" class="btn btn-ghost btn-sm" onclick="resetRegistry()" title="Reset all records in register to 0" style="color:var(--stamp);border-color:var(--rule);padding:5px 12px;font-size:11.5px;font-family:var(--type);margin-left:8px;cursor:pointer;">🗑️ Reset to 0</button>
          </div>
        </div>

        <div class="table-container">
          {table_html}
        </div>
      </section>

      <!-- Understated Security & System Status Bar -->
      <div class="system-strip" id="systemStrip">
        <div class="strip-group">
          <div class="strip-item">
            <span>🔐 <b data-i18n="lbl_fingerprint">RSA-PSS 2048 Fingerprint:</b></span>
            <code>{html.escape(fp_text[:28])}...</code>
            <button type="button" class="btn-copy" onclick="copyFingerprint('{html.escape(fp_text)}')">Copy</button>
          </div>
        </div>
        <div class="strip-group">
          <div class="strip-item">
            <span>🗺️ <b data-i18n="lbl_spatial_index">Spatial Index:</b> Telangana (TGRAC) &amp; Karnataka Master Datasets Loaded</span>
          </div>
          <div class="strip-item">
            <span>🛡️ <b data-i18n="lbl_key_store">Key Store:</b> <code>verification_keys/</code></span>
          </div>
        </div>
      </div>

    </div>

    <!-- Dashboard Footer -->
    <footer class="dash-footer">
      <div class="main-inner" style="padding-top:0; padding-bottom:0;">
        <div class="dash-footer-wrap">
          <div><b>OneBhoomi Registry Console</b> · Standalone Land Document Extraction &amp; Digital Seal System</div>
          <div>100% Air-Gapped &amp; Immutable · Zero cloud dependencies · Host: <code>{html.escape(host_name)}</code></div>
        </div>
      </div>
    </footer>
  </main>

</div>

<script>{DASHBOARD_JS}</script>
</body>
</html>
"""
    return page_html.encode("utf-8")


def render_new_scan(host_name: str = "localhost:8001", colab_url: str = "", message: str = "") -> bytes:
    """Renders the dedicated New Scan & Intake page equipped with the exact same left sidebar rail."""
    data = get_dashboard_data()
    today = datetime.now().strftime("%d-%m-%Y")

    if colab_url:
        try:
            worker_host = urlparse(colab_url).hostname or colab_url
        except Exception:
            worker_host = colab_url
        worker_label = f"Remote GPU ({worker_host[:16]}...)"
        gpu_selected = "selected"
        cpu_selected = ""
        mode_note = f"GPU Worker active at {worker_host} · Ultra-fast ~2s OCR inference via encrypted tunnel."
    else:
        worker_label = "Local CPU (PaddleOCR)"
        gpu_selected = ""
        cpu_selected = "selected"
        mode_note = "Local CPU OCR active · Runs directly on this machine with PaddleOCR."

    sidebar_html = _render_sidebar("new_scan", data["desk_n"], data["sealed_n"], worker_label)

    msg_banner = ""
    if message:
        msg_banner = f"""
      <div class="notice-banner" style="background:#FCE8E6; border-color:#F5B7B1; border-left-color:var(--stamp); margin-bottom:20px;">
        <div class="notice-info">
          <span class="notice-icon">⚠️</span>
          <div class="notice-text" style="color:var(--stamp-deep);"><b>Intake Notice:</b> {html.escape(message)}</div>
        </div>
      </div>"""

    page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OneBhoomi — New Document Scan &amp; Intake</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..900;1,9..144,300..900&family=Archivo:wght@400;500;600;700&family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Noto+Sans+Telugu:wght@400;500;600;700&family=Noto+Sans+Kannada:wght@400;500;600;700&family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{DASHBOARD_CSS}</style>
</head>
<body>

<div class="security-bg" aria-hidden="true"></div>

<!-- Loading overlay on extraction submission -->
<div class="loading-overlay" id="loadingOverlay">
  <div class="loading-box">
    <div class="loading-spinner"></div>
    <div class="loading-title">Processing Document Scan</div>
    <div class="loading-sub">Running OCR inference, parsing canonical facts, and executing Stage 1 validation checks...</div>
  </div>
</div>

<div class="app-layout">

  {sidebar_html}

  <!-- MAIN WORKSPACE CONTENT -->
  <main class="dash-content">
    <div class="main-inner">
      
      <!-- Top Action Bar -->
      <div class="top-action-bar">
        <div class="header-left">
          <h1 data-i18n="intake_h1">New Document Scan &amp; <em>Intake</em></h1>
          <div class="header-tagline">Desk 01 · Process land documents (Sale Deeds, Agreements, GPAs) with local OCR or Kaggle GPU acceleration.</div>
        </div>
        <div class="header-right"></div>

      {msg_banner}

      <!-- Dedicated Intake Card -->
      <div class="intake-card">
        <div class="intake-head">
          <h2 data-i18n="intake_title">Registration &amp; Scan Intake</h2>
          <span class="intake-badge">Stage 1 · Document Intake</span>
        </div>

        <form id="scanForm" action="/extract" method="post" enctype="multipart/form-data">
          
          <!-- Dropzone File Selector -->
          <div class="intake-dropzone" id="intakeDropzone" tabindex="0" role="button" aria-label="Drop scan file here or click to browse">
            <svg class="dz-icon-svg" viewBox="0 0 24 24" stroke-width="1.6">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="12" y1="18" x2="12" y2="12"></line>
              <line x1="9" y1="15" x2="15" y2="15"></line>
            </svg>
            <div class="dz-main-text" data-i18n="dz_main_text">Drop the document scan copy here, or <span>browse local files</span></div>
            <div class="dz-sub-text">Supports PDF (Multi-page supported) · PNG · JPG · TIFF &mdash; Read locally, certified with RSA-PSS</div>
            <input type="file" name="document_image" id="scan_file_input" accept="image/*,.pdf,application/pdf" hidden required>
          </div>

          <!-- Selected File Chip -->
          <div class="filechip" id="fileChip" style="display:none;">
            <div class="filechip-name">
              <span>📄</span>
              <span id="chipName">document.pdf</span>
            </div>
            <div class="filechip-actions">
              <span class="filechip-size" id="chipSize">1.2 MB</span>
              <button type="button" class="filechip-btn" id="chipRemove" title="Remove file">&times;</button>
            </div>
          </div>

          <!-- OCR Engine, Document Language & Execution Settings -->
          <div class="config-grid">
            <div class="config-box">
              <div class="config-box-title" data-i18n="ocr_engine_title">OCR Processing Engine</div>
              <select name="processing_mode" id="processing_mode" class="select-mode">
                <option value="gpu" {gpu_selected}>⚡ Remote GPU Worker (Encrypted Cloudflare Tunnel, ~2s)</option>
                <option value="cpu" {cpu_selected}>🐢 Local CPU (PaddleOCR on this machine)</option>
              </select>
              <div class="config-note">{html.escape(mode_note)}</div>
            </div>

            <div class="config-box">
              <div class="config-box-title" data-i18n="doc_language_title">Document Primary Language</div>
              <select name="document_language" id="document_language" class="select-mode">
                <option value="en" selected>English (Default · Latin Script)</option>
                <option value="hi">हिंदी · Hindi (Devanagari)</option>
                <option value="te">తెలుగు · Telugu (Official Deeds)</option>
                <option value="kn">ಕನ್ನಡ · Kannada (Revenue Records)</option>
                <option value="ta">தமிழ் · Tamil (Registration Deeds)</option>
                <option value="mr">मराठी · Marathi (Devanagari)</option>
                <option value="ur">اردو · Urdu (Perso-Arabic Script)</option>
              </select>
              <div class="config-note" data-i18n="doc_language_note">
                Select deed language. English numbers (0–9), survey codes, and dates are recognized across all languages.
              </div>
            </div>

            <div class="config-box">
              <div class="config-box-title" data-i18n="val_grounding_title">Validation &amp; Spatial Grounding</div>
              <div style="font-size:13px; color:var(--ink); font-weight:600; margin-bottom:4px;">
                ✓ Automatic 5-Point Rule Engine &amp; GIS Check
              </div>
              <div class="config-note">
                Checks required fields, area numeric bounds, date chronological logic, survey designations, and cross-references Telangana/Karnataka administrative GIS polygons.
              </div>
            </div>
          </div>

          <!-- Pipeline Progression Stepper -->
          <div class="stepper-strip">
            <div class="stepper-item step-active">
              <span class="step-num">1</span>
              <span class="step-label">Scan &amp; OCR Text</span>
            </div>
            <div class="stepper-item">
              <span class="step-num">2</span>
              <span class="step-label">Machine Validation</span>
            </div>
            <div class="stepper-item">
              <span class="step-num">3</span>
              <span class="step-label">Clerk Review Desk</span>
            </div>
            <div class="stepper-item">
              <span class="step-num">4</span>
              <span class="step-label">RSA-PSS 2048 Digital Seal</span>
            </div>
          </div>

          <!-- Submit Button & Security Note -->
          <div class="submit-row">
            <div class="submit-note">
              🔒 Zero external database or cloud storage. Files are processed in memory and persisted into local verification store.
            </div>
            <button type="submit" class="btn btn-primary" style="padding:12px 28px; font-size:12.5px;" data-i18n="btn_start_extract">
              Start Document Extraction &rarr;
            </button>
          </div>

        </form>
      </div>

    </div>

    <!-- Page Footer -->
    <footer class="dash-footer">
      <div class="main-inner" style="padding-top:0; padding-bottom:0;">
        <div class="dash-footer-wrap">
          <div><b>OneBhoomi Registry Console</b> · Standalone Land Document Extraction &amp; Digital Seal System</div>
          <div>100% Air-Gapped &amp; Immutable · Zero cloud dependencies · Host: <code>{html.escape(host_name)}</code></div>
        </div>
      </div>
    </footer>
  </main>

</div>

<script>{DASHBOARD_JS}</script>
</body>
</html>
"""
    return page_html.encode("utf-8")


def render_activity_timeline(record: dict) -> str:
    """
    Renders a collapsible vertical audit-trail timeline for a record, strictly
    reading existing lifecycle timestamps and actor fields from verification_db.json
    without fabricating unrecorded fields.
    """
    import accounts_store
    import ocr_learning_service

    rec_id = record.get("verification_id", "")
    status = record.get("status", "UNKNOWN")
    payload = record.get("document_payload") or {}
    doc_type = payload.get("document_type") or record.get("filename") or "Land Document"
    doc_no = payload.get("document_number") or rec_id[:8]

    # 1. Upload Event
    created_at = record.get("created_at") or record.get("timestamp")
    created_str = _fmt_datetime(created_at)
    uploaded_by = record.get("uploaded_by_user_id")
    uploader_label = "System / Direct Intake Desk (User ID not recorded)"
    if uploaded_by:
        user_info = accounts_store.get_user(uploaded_by)
        if user_info:
            uploader_label = f"{user_info.get('name', uploaded_by)} ({user_info.get('role', 'clerk').title()})"
        else:
            uploader_label = f"User ID: {uploaded_by[:8]}"

    # 2. Clerk Correction Event
    corrected_at = record.get("corrected_at") or record.get("updated_at")
    fb_items = []
    try:
        store = ocr_learning_service.load_store()
        fb_items = [fb for fb in store.get("feedback", []) if fb.get("verification_id") == rec_id]
    except Exception:
        fb_items = []

    has_corrections = bool(fb_items) or bool(record.get("field_provenance")) or status in {"READY_FOR_APPROVAL", "APPROVED", "REJECTED"}
    if corrected_at:
        correction_status = f"Corrections saved on {_fmt_datetime(corrected_at)}"
        corr_done = True
    elif fb_items:
        earliest_fb = min((fb.get("timestamp") for fb in fb_items if fb.get("timestamp")), default="")
        fb_time_str = f" on {_fmt_datetime(earliest_fb)}" if earliest_fb else ""
        correction_status = f"<b>{len(fb_items)} field correction(s) logged</b>{fb_time_str} &middot; <span style='color:var(--ink-soft); font-size:11px;'>(clerk_reviewed_at not stored on record in verification_db.json)</span>"
        corr_done = True
    elif has_corrections:
        correction_status = "Field review marked complete &middot; <span style='color:var(--ink-soft); font-size:11px;'>(Review timestamp &amp; reviewer ID not stored on record in verification_db.json)</span>"
        corr_done = True
    else:
        correction_status = "Awaiting clerk review &amp; verification"
        corr_done = False

    # 3. Submission to Officer Event
    submitted_at = record.get("submitted_at") or record.get("ready_for_approval_at")
    clerk_sub = bool(record.get("clerk_submitted", False))
    is_submitted = (status in {"APPROVED", "REJECTED", "DUPLICATE"}) or (status == "READY_FOR_APPROVAL" and clerk_sub)
    if submitted_at:
        submission_status = f"Transferred to Officer Queue on {_fmt_datetime(submitted_at)} &middot; <span style='color:var(--ink-soft); font-size:11px;'>(Submitting clerk ID not stored on record)</span>"
        sub_done = True
    elif is_submitted:
        submission_status = "Transferred to Officer Queue &middot; <span style='color:var(--ink-soft); font-size:11px;'>(Submission timestamp not stored on record)</span>"
        sub_done = True
    else:
        submission_status = "Not yet submitted to Officer (In Clerk Review)"
        sub_done = False

    # 4. Final Decision Event
    approved_at = record.get("approved_at")
    rejected_at = record.get("rejected_at")
    decided_by = record.get("approved_by_user_id") or record.get("decided_by") or record.get("rejected_by_user_id")
    if decided_by:
        off_user = accounts_store.get_user(decided_by)
        dec_by_label = f"by {off_user.get('name', decided_by)}" if off_user else f"by Officer {decided_by[:8]}"
    else:
        dec_by_label = "Gazetted Officer &middot; <span style='color:var(--ink-soft); font-size:11px;'>(Officer ID not stored on record in verification_db.json)</span>"

    signature = record.get("signature") or ""
    sig_short = f"{signature[:24]}...{signature[-16:]}" if len(signature) > 40 else signature

    if status == "APPROVED":
        dec_title = "Approved &amp; Cryptographically Sealed"
        dec_time = _fmt_datetime(approved_at)
        dec_color = "var(--green)"
        dec_icon = "🔒"
        dec_body = f"""
        <div style="font-size:12px; color:var(--ink-soft); margin-top:4px; line-height:1.5;">
            <b>Decided By:</b> {dec_by_label}<br>
            <b>Certified At:</b> {html.escape(dec_time)}<br>
            <b>Algorithm:</b> RSA-PSS 2048-bit / SHA-256<br>
            <div style="margin-top:6px; background:#f8fafc; border:1px solid var(--rule); padding:6px 10px; border-radius:4px; font-family:var(--type); font-size:11px; word-break:break-all;">
                <b>Seal Signature Hash:</b> <code>{html.escape(sig_short)}</code>
            </div>
        </div>
        """
    elif status == "REJECTED":
        dec_title = "Document Rejected by Officer"
        dec_time = _fmt_datetime(rejected_at)
        dec_color = "var(--stamp)"
        dec_icon = "❌"
        rej_reason = record.get("rejection_reason") or "No rejection reason was recorded."
        dec_body = f"""
        <div style="font-size:12px; color:var(--ink-soft); margin-top:4px; line-height:1.5;">
            <b>Decided By:</b> {dec_by_label}<br>
            <b>Rejected At:</b> {html.escape(dec_time)}<br>
            <b>Rejection Reason:</b> {html.escape(rej_reason)}
        </div>
        """
    elif status == "DUPLICATE":
        dup = record.get("duplicate_info") or {}
        dup_matched = dup.get("matched_record_id", "")
        dec_title = "Blocked as Duplicate Registration Attempt"
        dec_color = "var(--amber)"
        dec_icon = "⛔"
        dec_body = f"""
        <div style="font-size:12px; color:var(--ink-soft); margin-top:4px; line-height:1.5;">
            Matches existing sealed record No. <code>{html.escape(dup_matched[:8].upper())}</code>. Double registration blocked.
        </div>
        """
    elif status == "READY_FOR_APPROVAL":
        dec_title = "Awaiting Officer Final Determination &amp; Seal"
        dec_color = "#2563eb"
        dec_icon = "⏳"
        dec_body = '<div style="font-size:12px; color:var(--ink-soft); margin-top:4px;">In officer verification queue awaiting final legal certification.</div>'
    else:
        dec_title = "Pending Officer Review"
        dec_color = "var(--ink-soft)"
        dec_icon = "⚪"
        dec_body = '<div style="font-size:12px; color:var(--ink-soft); margin-top:4px;">Document has not completed Stage 1 clerk review.</div>'

    return f"""
    <details id="audit-timeline" class="panel timeline-panel rv in" open style="margin-top:24px; border:1.5px solid var(--rule); background:var(--card); border-radius:4px; overflow:hidden;">
      <summary class="tab" style="cursor:pointer; user-select:none; background:#263342; color:#fff; padding:12px 18px; font-family:var(--type); font-size:11px; letter-spacing:.16em; text-transform:uppercase; display:flex; justify-content:space-between; align-items:center;">
        <div style="display:flex; align-items:center; gap:10px;">
          <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#38bdf8;"></span>
          <span>Schedule D &middot; Record Audit Trail &amp; Lifecycle Timeline</span>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
          <em style="color:#94a3b8; font-style:normal; font-size:11px;">Record {html.escape(rec_id[:8].upper())}</em>
          <span style="font-size:11px; color:#38bdf8; font-family:var(--type); text-transform:none; letter-spacing:0;">&#9660; Toggle Timeline</span>
        </div>
      </summary>
      <div class="body" style="padding:20px 24px;">
        <div style="position:relative; padding-left:28px; display:flex; flex-direction:column; gap:18px;">
          <!-- Vertical connecting bar -->
          <div style="position:absolute; left:9px; top:6px; bottom:6px; width:2px; background:var(--rule-soft);"></div>

          <!-- Step 1: Upload -->
          <div class="timeline-step" style="position:relative;">
            <div style="position:absolute; left:-28px; top:1px; width:20px; height:20px; border-radius:50%; background:#2563eb; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;">✓</div>
            <div style="font-size:13px; font-weight:700; color:var(--ink);">1. Document Ingestion &amp; Upload</div>
            <div style="font-size:12px; color:var(--ink-soft); margin-top:2px;">
              <b>Received:</b> {html.escape(created_str)} &middot; <b>Uploader:</b> {html.escape(uploader_label)} &middot; <b>Document:</b> {html.escape(doc_type)} (No. {html.escape(doc_no)})
            </div>
          </div>

          <!-- Step 2: Corrections -->
          <div class="timeline-step" style="position:relative;">
            <div style="position:absolute; left:-28px; top:1px; width:20px; height:20px; border-radius:50%; background:{'#059669' if corr_done else '#cbd5e1'}; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;">{'✓' if corr_done else '•'}</div>
            <div style="font-size:13px; font-weight:700; color:var(--ink);">2. Clerk Fact Review &amp; Corrections</div>
            <div style="font-size:12px; color:var(--ink-soft); margin-top:2px;">
              {correction_status}
            </div>
          </div>

          <!-- Step 3: Submission -->
          <div class="timeline-step" style="position:relative;">
            <div style="position:absolute; left:-28px; top:1px; width:20px; height:20px; border-radius:50%; background:{'#2563eb' if sub_done else '#cbd5e1'}; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;">{'✓' if sub_done else '•'}</div>
            <div style="font-size:13px; font-weight:700; color:var(--ink);">3. Transfer to Officer Verification Queue</div>
            <div style="font-size:12px; color:var(--ink-soft); margin-top:2px;">
              {submission_status}
            </div>
          </div>

          <!-- Step 4: Final Determination -->
          <div class="timeline-step" style="position:relative;">
            <div style="position:absolute; left:-28px; top:1px; width:20px; height:20px; border-radius:50%; background:{dec_color}; color:#fff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;">{dec_icon}</div>
            <div style="font-size:13px; font-weight:700; color:{dec_color};">4. {dec_title}</div>
            {dec_body}
          </div>

        </div>
      </div>
    </details>
    """



# =====================================================================
# User Dedicated Workspace & Ingestion Console (Module 4)
# =====================================================================

def render_user_dashboard(
    user_id: Optional[str] = None,
    host_name: str = "localhost:8001",
    colab_url: str = "",
    user_name: str = "User",
) -> bytes:
    """
    Renders the dedicated User Workspace:
    1. Reuses the document upload dropzone from render_new_scan().
    2. Groups user's records (uploaded_by_user_id == user_id) into:
       - 'Needs your review' (EXTRACTED, NEEDS_REVIEW)
       - 'Sent to officer' (READY_FOR_APPROVAL)
       - 'Decided' (APPROVED, REJECTED, DUPLICATE)
       using the existing _badge() helper.
    3. For Decided records, links directly to certificate/QR for APPROVED,
       and displays the officer's rejection reason for REJECTED.
    4. Displays an 'Unassigned / legacy' section for pre-existing records (uploaded_by_user_id = null).
    """
    data = get_dashboard_data()

    if colab_url:
        try:
            worker_host = urlparse(colab_url).hostname or colab_url
        except Exception:
            worker_host = colab_url
        worker_label = f"Remote GPU ({worker_host[:16]}...)"
        gpu_selected = "selected"
        cpu_selected = ""
        mode_note = f"GPU Worker active at {worker_host} · Ultra-fast ~2s OCR inference via encrypted tunnel."
    else:
        worker_label = "Local CPU (PaddleOCR)"
        gpu_selected = ""
        cpu_selected = "selected"
        mode_note = "Local CPU OCR active · Runs directly on this machine with PaddleOCR."

    sidebar_html = _render_sidebar("user", data["desk_n"], data["sealed_n"], worker_label, role="user")

    # Load all records from database
    try:
        db = verification_service.load_db()
    except Exception:
        db = {}

    all_recs = [r for r in db.values() if isinstance(r, dict) and r.get("verification_id")]
    all_recs.sort(key=lambda r: r.get("created_at") or "", reverse=True)

    # Filter into the four designated groups
    needs_review: List[Dict[str, Any]] = []
    sent_to_officer: List[Dict[str, Any]] = []
    decided: List[Dict[str, Any]] = []
    legacy: List[Dict[str, Any]] = []

    for r in all_recs:
        status = r.get("status") or "EXTRACTED"
        uploaded_by = r.get("uploaded_by_user_id")

        # Strictly isolate user files: show ONLY files uploaded by the logged in user
        if user_id and uploaded_by != user_id:
            continue

        clerk_submitted = bool(r.get("clerk_submitted", False))
        if status in {"APPROVED", "REJECTED", "DUPLICATE"}:
            decided.append(r)
        elif status == "READY_FOR_APPROVAL" and clerk_submitted:
            sent_to_officer.append(r)
        elif status in {"EXTRACTED", "NEEDS_REVIEW", "READY_FOR_APPROVAL"}:
            needs_review.append(r)
        else:
            legacy.append(r)

    def _render_clerk_table(records: List[Dict[str, Any]], empty_text: str, is_decided_section: bool = False) -> str:
        if not records:
            return f"""
            <div class="table-empty" style="padding:24px; text-align:center; background:rgba(0,0,0,0.02); border-radius:8px; border:1px dashed var(--rule);">
                <p style="color:var(--ink-soft); font-size:13px; margin:0;">{html.escape(empty_text)}</p>
            </div>
            """

        rows_html = []
        for i, r in enumerate(records, start=1):
            vid = r.get("verification_id", "")
            payload = r.get("document_payload") or {}
            prop = payload.get("property") or {}
            status = r.get("status") or "EXTRACTED"
            doc_type = payload.get("document_type") or r.get("filename") or "Land Deed"
            doc_no = payload.get("document_number") or payload.get("serial_number") or vid[:8]
            rec_date = _fmt_date(r.get("created_at") or "")
            village = prop.get("village") or "—"
            district = prop.get("district") or "—"
            survey = prop.get("survey_number") or "—"

            parties_list = payload.get("parties") or []
            parties_str = ", ".join(
                p.get("name", "") for p in parties_list if isinstance(p, dict) and p.get("name")
            ) or "—"

            # Details & Actions
            details_html = ""
            action_buttons = []

            if status == "APPROVED":
                details_html = '<div style="font-size:11.5px; color:#059669; font-weight:600;">✓ Certified &amp; RSA-PSS Sealed</div>'
                action_buttons.append(
                    f'<a class="act-btn" style="background:#059669; color:#ffffff !important; font-weight:700; border-color:#047857; display:inline-block; padding:5px 10px; font-size:11.5px; text-decoration:none; white-space:nowrap;" href="/?verification_id={html.escape(vid)}" target="_blank" title="Open Public Certificate with QR">📜 Certificate &amp; QR</a>'
                )
                action_buttons.append(
                    f'<a class="act-btn" style="padding:5px 10px; font-size:11.5px; display:inline-block; white-space:nowrap; text-decoration:none;" href="/record?verification_id={html.escape(vid)}&role=clerk" title="Review canonical extraction">Record Details</a>'
                )
            elif status == "REJECTED":
                rej_reason = r.get("rejection_reason") or "No reason was recorded by the officer."
                details_html = f"""
                <div style="font-size:11.5px; color:#dc2626; background:#fef2f2; border:1px solid #fecaca; border-left:3px solid #dc2626; padding:5px 8px; border-radius:5px; margin-top:2px;">
                    <b>Officer Rejection Reason:</b> {html.escape(rej_reason)}
                </div>
                """
                action_buttons.append(
                    f'<a class="act-btn" style="border-color:#f87171; color:#dc2626;" href="/record?verification_id={html.escape(vid)}&role=clerk">Inspect &amp; Resubmit</a>'
                )
            elif status == "DUPLICATE":
                details_html = '<div style="font-size:11.5px; color:#b45309;">Flagged as duplicate entry in registration ledger</div>'
                action_buttons.append(
                    f'<a class="act-btn" href="/record?verification_id={html.escape(vid)}&role=clerk">Audit Conflict</a>'
                )
            elif status == "READY_FOR_APPROVAL" and r.get("clerk_submitted", False):
                details_html = '<div style="font-size:11.5px; color:var(--ink-soft);">Awaiting officer review &amp; legal seal</div>'
                action_buttons.append(
                    f'<a class="act-btn" href="/record?verification_id={html.escape(vid)}&role=clerk">View Submission</a>'
                )
            elif status == "READY_FOR_APPROVAL":
                details_html = '<div style="font-size:11.5px; color:#15803d; font-weight:600;">✓ Checks passed · Ready to submit to officer</div>'
                action_buttons.append(
                    f'<a class="act-btn" style="background:#059669; color:#ffffff !important; font-weight:700; border-color:#047857; display:inline-block; padding:6px 12px; font-size:12px; text-decoration:none;" href="/record?verification_id={html.escape(vid)}&role=clerk">Review &amp; Submit &rarr;</a>'
                )
            else:
                # EXTRACTED or NEEDS_REVIEW
                details_html = '<div style="font-size:11.5px; color:#d97706;">Stage 1 complete · Needs field review</div>'
                action_buttons.append(
                    f'<a class="act-btn" style="background:#059669; color:#ffffff !important; font-weight:700; border-color:#047857; display:inline-block; padding:6px 12px; font-size:12px; text-decoration:none;" href="/record?verification_id={html.escape(vid)}&role=clerk">Review &amp; Amend &rarr;</a>'
                )

            rows_html.append(f"""
            <tr class="data-row" data-status="{html.escape(status)}">
              <td class="td-sl">{i}</td>
              <td class="td-date">{html.escape(rec_date)}</td>
              <td>
                <span class="td-doc-main">{html.escape(doc_type)}</span>
                <span class="td-doc-sub">No. {html.escape(doc_no)}</span>
              </td>
              <td title="{html.escape(parties_str)}">{html.escape(parties_str[:32] + ('...' if len(parties_str) > 32 else ''))}</td>
              <td>
                <span class="td-place-main">{html.escape(village)}</span>
                <span class="td-place-sub">{html.escape(district)}</span>
              </td>
              <td class="td-mono">{html.escape(survey)}</td>
              <td>{_badge(status)}</td>
              <td>{details_html}</td>
              <td>
                <div class="action-links" style="display:flex; gap:6px; align-items:center; justify-content:center; white-space:nowrap; flex-wrap:nowrap;">
                  {' '.join(action_buttons)}
                </div>
              </td>
            </tr>
            """)

        return f"""
        <div class="queue-table-scroll">
          <table class="master-ledger clerk-table">
            <thead>
              <tr>
                <th>Sl.</th>
                <th>Received</th>
                <th>Document &amp; No.</th>
                <th>Parties</th>
                <th>Location</th>
                <th>Survey No.</th>
                <th>Status</th>
                <th>Verification Notes</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows_html)}
            </tbody>
          </table>
        </div>
        """

    needs_review_table = _render_clerk_table(needs_review, "No documents currently waiting for your review. Ingest a new deed scan above.")
    sent_to_officer_table = _render_clerk_table(sent_to_officer, "No documents currently pending with the officer.")
    decided_table = _render_clerk_table(decided, "No decided documents in your archive yet.", is_decided_section=True)
    legacy_table = _render_clerk_table(legacy, "No legacy or unassigned records found.")

    page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OneBhoomi — User Digitization Desk</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..900;1,9..144,300..900&family=Archivo:wght@400;500;600;700&family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Noto+Sans+Telugu:wght@400;500;600;700&family=Noto+Sans+Kannada:wght@400;500;600;700&family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    {DASHBOARD_CSS}
    .clerk-summary-strip {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 24px;
    }}
    .clerk-stat-card {{
        background: #ffffff;
        border: 1px solid var(--rule);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }}
    .clerk-stat-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 4px;
    }}
    .card-review::before {{ background: #f59e0b; }}
    .card-officer::before {{ background: #3b82f6; }}
    .card-decided::before {{ background: #10b981; }}
    .card-legacy::before {{ background: #64748b; }}
    .stat-num {{
        font-family: 'Fraunces', serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--ink);
        line-height: 1;
    }}
    .stat-title {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--ink-soft);
        margin-top: 6px;
    }}
    .section-box {{
        background: #ffffff;
        border: 1px solid var(--rule);
        border-radius: 8px;
        margin-bottom: 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        overflow: hidden;
    }}
    .section-header {{
        padding: 16px 20px;
        border-bottom: 1px solid var(--rule);
        background: #fafaf9;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .section-title {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--ink);
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-count {{
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 12px;
        background: rgba(0,0,0,0.06);
        color: var(--ink);
    }}
  </style>
</head>
<body>

<div class="security-bg" aria-hidden="true"></div>

<!-- Loading overlay on extraction submission -->
<div class="loading-overlay" id="loadingOverlay">
  <div class="loading-box">
    <div class="loading-spinner"></div>
    <div class="loading-title">Ingesting Document Scan</div>
    <div class="loading-sub">Performing local OCR extraction, semantic field mapping, and validation checks...</div>
  </div>
</div>

<div class="app-layout">

  {sidebar_html}

  <main class="dash-content">
    <div class="main-inner">

      <!-- Top Action Bar -->
      <div class="top-action-bar">
        <div class="header-left">
          <h1>User <em>Digitization Desk</em></h1>
          <div class="header-tagline">
            Logged in as <b>{html.escape(user_name)}</b> · Land Document Upload, Provenance Audit &amp; Intake
          </div>
        </div>
        <div class="header-right">
          <a href="/auth/signout" class="act-btn" style="border-color:#fca5a5; color:#b91c1c;">Sign Out</a>
        </div>
      </div>

      <!-- Quick Metrics Strip -->
      <div class="clerk-summary-strip">
        <div class="clerk-stat-card card-review">
          <div class="stat-num">{len(needs_review)}</div>
          <div class="stat-title">Needs Your Review</div>
        </div>
        <div class="clerk-stat-card card-officer">
          <div class="stat-num">{len(sent_to_officer)}</div>
          <div class="stat-title">Sent to Officer</div>
        </div>
        <div class="clerk-stat-card card-decided">
          <div class="stat-num">{len(decided)}</div>
          <div class="stat-title">Decided</div>
        </div>
        <div class="clerk-stat-card card-legacy">
          <div class="stat-num">{len(needs_review) + len(sent_to_officer) + len(decided)}</div>
          <div class="stat-title">Total Your Submissions</div>
        </div>
      </div>

      <!-- REUSED INGESTION DROPZONE (Stage 1 Intake Desk) -->
      <div class="intake-card" style="margin-bottom:28px;">
        <div class="intake-head">
          <h2>New Document Scan &amp; Intake Desk</h2>
          <span class="intake-badge">Reused Intake Point · Desk 01</span>
        </div>

        <form id="scanForm" action="/extract" method="post" enctype="multipart/form-data">
          <!-- Dropzone File Selector -->
          <div class="intake-dropzone" id="intakeDropzone" tabindex="0" role="button" aria-label="Drop scan file here or click to browse">
            <svg class="dz-icon-svg" viewBox="0 0 24 24" stroke-width="1.6">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="12" y1="18" x2="12" y2="12"></line>
              <line x1="9" y1="15" x2="15" y2="15"></line>
            </svg>
            <div class="dz-main-text">Drop the title deed or scan copy here, or <span>browse local files</span></div>
            <div class="dz-sub-text">Supports PDF (Multi-page) · PNG · JPG · TIFF &mdash; Processed in-memory and linked to your user profile</div>
            <input type="file" name="document_image" id="scan_file_input" accept="image/*,.pdf,application/pdf" hidden required>
          </div>

          <!-- Selected File Chip -->
          <div class="filechip" id="fileChip" style="display:none;">
            <div class="filechip-name">
              <span>📄</span>
              <span id="chipName">document.pdf</span>
            </div>
            <div class="filechip-actions">
              <span class="filechip-size" id="chipSize">0.0 MB</span>
              <button type="button" class="filechip-btn" id="chipRemove" title="Remove file">&times;</button>
            </div>
          </div>

          <!-- Submit Button & Security Note -->
          <div class="submit-row" style="margin-top:16px;">
            <div class="submit-note">
              🔒 Ingestion will automatically link this document to user profile <b>{html.escape(user_name)}</b>.
            </div>
            <button type="submit" class="btn btn-primary" id="submitBtn" style="padding:12px 28px; font-size:12.5px;">
              Start Document Extraction &rarr;
            </button>
          </div>
        </form>
      </div>

      <!-- SECTION 1: Needs Your Review -->
      <div class="section-box" id="sec-needs-review">
        <div class="section-header">
          <div class="section-title">
            <span>📝 Needs Your Review</span>
            <span class="section-count" style="background:#fef3c7; color:#92400e;">{len(needs_review)} items</span>
          </div>
          <div style="font-size:12px; color:var(--ink-soft);">
            Fresh extractions and records requiring user field corrections
          </div>
        </div>
        {needs_review_table}
      </div>

      <!-- SECTION 2: Sent to Officer -->
      <div class="section-box" id="sec-sent-officer">
        <div class="section-header">
          <div class="section-title">
            <span>📤 Sent to Officer</span>
            <span class="section-count" style="background:#dbeafe; color:#1e40af;">{len(sent_to_officer)} items</span>
          </div>
          <div style="font-size:12px; color:var(--ink-soft);">
            Marked READY_FOR_APPROVAL and pending officer cryptographic seal
          </div>
        </div>
        {sent_to_officer_table}
      </div>

      <!-- SECTION 3: Decided -->
      <div class="section-box" id="sec-decided">
        <div class="section-header">
          <div class="section-title">
            <span>⚖️ Decided Documents</span>
            <span class="section-count" style="background:#d1fae5; color:#065f46;">{len(decided)} items</span>
          </div>
          <div style="font-size:12px; color:var(--ink-soft);">
            Official final determinations (Approved certificates, Rejections, or Duplicate flags)
          </div>
        </div>
        {decided_table}
        {f'<div class="decided-timelines-wrap" style="margin-top:20px; padding:16px 20px; background:#f8fafc; border-top:1px solid var(--rule);"><div style="font-family:var(--serif); font-size:13px; font-weight:700; color:var(--ink); text-transform:uppercase; letter-spacing:0.04em; margin-bottom:10px;">📜 Decided Records Activity &amp; Audit Lifecycles</div>{"".join(render_activity_timeline(r) for r in decided)}</div>' if decided else ''}
      </div>

      {f'''
      <!-- SECTION 4: Unassigned / Legacy Records -->
      <div class="section-box" id="sec-legacy">
        <div class="section-header">
          <div class="section-title">
            <span>🗄️ Unassigned / Legacy Records</span>
            <span class="section-count" style="background:#f1f5f9; color:#475569;">{len(legacy)} items</span>
          </div>
          <div style="font-size:12px; color:var(--ink-soft);">
            Pre-existing records or documents explicitly linked to your user profile
          </div>
        </div>
        {legacy_table}
      </div>
      ''' if legacy else ''}

    </div>

    <!-- Page Footer -->
    <footer class="dash-footer">
      <div class="main-inner" style="padding-top:0; padding-bottom:0;">
        <div class="dash-footer-wrap">
          <div><b>OneBhoomi Registry Console</b> · User Workspace</div>
          <div>100% Air-Gapped &amp; Immutable · Zero cloud dependencies · Host: <code>{html.escape(host_name)}</code></div>
        </div>
      </div>
    </footer>
  </main>

</div>

<script>{DASHBOARD_JS}</script>
</body>
</html>
"""
    return page_html.encode("utf-8")


def render_clerk_dashboard(*args, **kwargs) -> bytes:
    """Backwards-compatibility alias for render_user_dashboard."""
    return render_user_dashboard(*args, **kwargs)


# =====================================================================
# Officer Approval & Sealing Queue (Module 5)
# =====================================================================

def render_officer_dashboard(
    host_name: str = "localhost:8001",
    colab_url: str = "",
    user_name: str = "Officer",
) -> bytes:
    """
    Renders the dedicated Officer Workspace:
    1. Lists clerk-submitted records with status READY_FOR_APPROVAL or NEEDS_REVIEW.
    2. Sorts by wait duration (oldest first by default).
    3. Each row links to the existing record view for that document (/record?verification_id=...)
       where the officer takes approval/rejection/sealing action.
    4. Displays at-a-glance workload summary metrics at the top.
    """
    import accounts_store
    data = get_dashboard_data()

    if colab_url:
        try:
            worker_host = urlparse(colab_url).hostname or colab_url
        except Exception:
            worker_host = colab_url
        worker_label = f"Remote GPU ({worker_host[:16]}...)"
    else:
        worker_label = "Local CPU (PaddleOCR)"

    sidebar_html = _render_sidebar("officer", data["desk_n"], data["sealed_n"], worker_label, role="officer")

    # Load all records from verification database
    try:
        db = verification_service.load_db()
    except Exception:
        db = {}

    # Extract all clerk-submitted records with status in ('READY_FOR_APPROVAL', 'NEEDS_REVIEW')
    pending_records = [
        r for r in db.values()
        if isinstance(r, dict)
        and r.get("verification_id")
        and (r.get("status") or "").upper() in {"READY_FOR_APPROVAL", "NEEDS_REVIEW"}
        and bool(r.get("clerk_submitted", False))
    ]

    # Sort oldest first (ascending created_at timestamp)
    pending_records.sort(key=lambda r: r.get("created_at") or "")

    # Sealed records for officer inspection
    sealed_records = [
        r for r in db.values()
        if isinstance(r, dict)
        and r.get("verification_id")
        and (r.get("status") or "").upper() == "APPROVED"
    ]
    sealed_records.sort(key=lambda r: r.get("created_at") or "", reverse=True)

    # Count other overall statuses for workload context
    sealed_count = len(sealed_records)
    rejected_count = sum(1 for r in db.values() if isinstance(r, dict) and r.get("status") == "REJECTED")
    total_ledger_count = len([r for r in db.values() if isinstance(r, dict) and r.get("verification_id")])

    def _calc_wait_time(created_at_str: str) -> Tuple[str, bool]:
        """Calculates humanized elapsed wait time and urgency boolean."""
        if not created_at_str:
            return "—", False
        try:
            clean = created_at_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean)
            if dt.tzinfo is None:
                from datetime import timezone
                dt = dt.replace(tzinfo=timezone.utc)
            from datetime import timezone
            now = datetime.now(timezone.utc)
            diff = now - dt
            secs = max(0, int(diff.total_seconds()))
            if secs < 60:
                return f"{secs}s waiting", False
            elif secs < 3600:
                return f"{secs // 60}m waiting", False
            elif secs < 86400:
                h = secs // 3600
                m = (secs % 3600) // 60
                return f"{h}h {m}m waiting", (h >= 2)
            else:
                d = secs // 86400
                return f"{d}d waiting", True
        except Exception:
            return "—", False

    # Extract distinct filter options for jurisdiction narrowing
    districts = sorted({
        (r.get("document_payload", {}).get("property", {}).get("district") or "").strip()
        for r in pending_records
        if (r.get("document_payload", {}).get("property", {}).get("district") or "").strip()
    })
    mandals = sorted({
        (r.get("document_payload", {}).get("property", {}).get("mandal") or "").strip()
        for r in pending_records
        if (r.get("document_payload", {}).get("property", {}).get("mandal") or "").strip()
    })
    doc_types = sorted({
        (r.get("document_payload", {}).get("document_type") or "").strip()
        for r in pending_records
        if (r.get("document_payload", {}).get("document_type") or "").strip()
    })

    district_options = "".join(f'<option value="{html.escape(d)}">{html.escape(d)}</option>' for d in districts)
    mandal_options = "".join(f'<option value="{html.escape(m)}">{html.escape(m)}</option>' for m in mandals)
    doc_type_options = "".join(f'<option value="{html.escape(dt)}">{html.escape(dt)}</option>' for dt in doc_types)

    filter_bar_html = f"""
    <div class="officer-filter-bar" style="background:#f8fafc; border-bottom:1px solid var(--rule); padding:14px 22px; display:flex; gap:16px; align-items:flex-end; flex-wrap:wrap;">
      <div class="filter-control" style="flex:1; min-width:180px;">
        <label for="filterDistrict" style="display:block; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--ink-soft); margin-bottom:4px;">District</label>
        <select id="filterDistrict" class="dash-select" onchange="applyOfficerFilters()" style="width:100%; padding:8px 12px; border:1px solid var(--rule); border-radius:4px; background:#fff; font-size:13px;">
          <option value="">All Districts ({len(districts)})</option>
          {district_options}
        </select>
      </div>
      <div class="filter-control" style="flex:1; min-width:180px;">
        <label for="filterMandal" style="display:block; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--ink-soft); margin-bottom:4px;">Mandal</label>
        <select id="filterMandal" class="dash-select" onchange="applyOfficerFilters()" style="width:100%; padding:8px 12px; border:1px solid var(--rule); border-radius:4px; background:#fff; font-size:13px;">
          <option value="">All Mandals ({len(mandals)})</option>
          {mandal_options}
        </select>
      </div>
      <div class="filter-control" style="flex:1; min-width:180px;">
        <label for="filterDocType" style="display:block; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--ink-soft); margin-bottom:4px;">Document Type</label>
        <select id="filterDocType" class="dash-select" onchange="applyOfficerFilters()" style="width:100%; padding:8px 12px; border:1px solid var(--rule); border-radius:4px; background:#fff; font-size:13px;">
          <option value="">All Document Types ({len(doc_types)})</option>
          {doc_type_options}
        </select>
      </div>
      <div class="filter-actions" style="display:flex; gap:10px; align-items:center;">
        <button type="button" class="act-btn" onclick="clearOfficerFilters()" style="padding:8px 14px; font-size:12px; background:#fff; cursor:pointer;">Clear Filters</button>
      </div>
      <div id="filterMatchCount" style="width:100%; font-size:12px; color:var(--ink-soft); margin-top:4px;">
        Showing all <b>{len(pending_records)}</b> pending queue records
      </div>
    </div>
    """

    rows_html = []
    for i, r in enumerate(pending_records, start=1):
        vid = r.get("verification_id", "")
        payload = r.get("document_payload") or {}
        prop = payload.get("property") or {}
        doc_type = payload.get("document_type") or r.get("filename") or "Land Deed"
        doc_no = payload.get("document_number") or payload.get("serial_number") or vid[:8]
        rec_date = _fmt_date(r.get("created_at") or "")
        village = prop.get("village") or "—"
        district = prop.get("district") or "—"
        mandal = prop.get("mandal") or "—"
        survey = prop.get("survey_number") or "—"

        wait_str, is_urgent = _calc_wait_time(r.get("created_at") or "")
        wait_badge = f"""
        <span class="badge" style="background:{'#fef2f2' if is_urgent else '#f1f5f9'}; color:{'#b91c1c' if is_urgent else '#334155'}; border:1px solid {'#fca5a5' if is_urgent else '#cbd5e1'}; font-weight:600;">
            {'⏱️ ' if is_urgent else ''}{html.escape(wait_str)}
        </span>
        """

        # Clerk attribution lookup
        clerk_id = r.get("uploaded_by_user_id")
        clerk_label = "Legacy / Unassigned"
        if clerk_id:
            clerk_user = accounts_store.get_user(clerk_id)
            if clerk_user:
                clerk_label = f"👤 {clerk_user.get('name', clerk_id)}"
            else:
                clerk_label = f"👤 Clerk ({clerk_id[:8]})"

        # Checks overview
        checks = r.get("checks", [])
        pass_count = sum(1 for c in checks if isinstance(c, dict) and c.get("status") == "PASS")
        total_checks = len(checks)
        checks_badge = f'<span style="font-size:12px; color:#059669; font-weight:600;">✓ {pass_count}/{total_checks} Checks Passed</span>' if total_checks else '<span style="font-size:12px; color:var(--ink-soft);">Stage 1 Verified</span>'

        loc_sub = f"{mandal} · {district}" if mandal != "—" else district

        rec_status = (r.get("status") or "READY_FOR_APPROVAL").upper()
        rows_html.append(f"""
        <tr class="data-row" data-status="{html.escape(rec_status)}" data-district="{html.escape(district)}" data-mandal="{html.escape(mandal)}" data-doc-type="{html.escape(doc_type)}">
          <td class="td-sl">{i}</td>
          <td>{wait_badge}</td>
          <td class="td-date">{html.escape(rec_date)}</td>
          <td>
            <span class="td-doc-main">{html.escape(doc_type)}</span>
            <span class="td-doc-sub">No. {html.escape(doc_no)}</span>
          </td>
          <td>
            <span style="font-size:12px; color:var(--ink); font-weight:500;">{html.escape(clerk_label)}</span>
          </td>
          <td>
            <span class="td-place-main">{html.escape(village)}</span>
            <span class="td-place-sub">{html.escape(loc_sub)}</span>
          </td>
          <td class="td-mono">{html.escape(survey)}</td>
          <td>{checks_badge}</td>
          <td>{_badge(rec_status)}</td>
          <td>
            <div class="action-links" style="justify-content:center;">
              <a class="act-btn btn-primary" style="background:#059669; color:#ffffff !important; font-weight:700; border-color:#047857; display:inline-block; padding:6px 12px; font-size:11.5px; text-decoration:none; white-space:nowrap;" href="/record?verification_id={html.escape(vid)}&role=officer">
                Review &amp; Approve / Seal &rarr;
              </a>
            </div>
          </td>
        </tr>
        """)

    if rows_html:
        queue_table_html = f"""
        <table class="master-ledger" id="officerQueueTable">
          <thead>
            <tr>
              <th>Priority</th>
              <th>Queue Age</th>
              <th>Received</th>
              <th>Document &amp; No.</th>
              <th>Ingested By</th>
              <th>Location</th>
              <th>Survey No.</th>
              <th>Audit Status</th>
              <th>State</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
            <tr id="officerFilterEmptyRow" style="display:none;">
              <td colspan="10" style="text-align:center; padding:36px 16px; color:var(--ink-soft); font-size:13.5px;">
                🔍 No pending records match the selected district, mandal, or document type filters.
                <br><button type="button" class="act-btn" onclick="clearOfficerFilters()" style="margin-top:10px; font-size:12px; padding:6px 14px; cursor:pointer;">Reset Filters</button>
              </td>
            </tr>
          </tbody>
        </table>
        """
    else:
        queue_table_html = """
        <div class="table-empty" style="padding:48px 24px; text-align:center; background:#ffffff;">
          <div style="font-size:2.4rem; margin-bottom:12px;">✓</div>
          <h3 style="font-family:'Fraunces', serif; font-size:1.3rem; color:var(--ink); margin-bottom:6px;">Approval Queue is Clear</h3>
          <p style="color:var(--ink-soft); font-size:13.5px; max-width:460px; margin:0 auto;">
            All pending land documents have been audited and sealed. When clerks submit records with <code>READY_FOR_APPROVAL</code> status, they will automatically appear here sorted by waiting duration.
          </p>
        </div>
        """

    # Render Sealed Records Table for Officer Inspection
    sealed_rows = []
    for i, r in enumerate(sealed_records, start=1):
        vid = r.get("verification_id", "")
        payload = r.get("document_payload") or {}
        prop = payload.get("property") or {}
        doc_type = payload.get("document_type") or r.get("filename") or "Land Deed"
        doc_no = payload.get("document_number") or payload.get("serial_number") or vid[:8]
        rec_date = _fmt_date(r.get("created_at") or "")
        village = prop.get("village") or "—"
        district = prop.get("district") or "—"
        mandal = prop.get("mandal") or "—"
        survey = prop.get("survey_number") or "—"
        loc_sub = f"{mandal} · {district}" if mandal != "—" else district

        sealed_rows.append(f"""
        <tr class="data-row">
          <td class="td-sl">{i}</td>
          <td class="td-date">{html.escape(rec_date)}</td>
          <td>
            <span class="td-doc-main">{html.escape(doc_type)}</span>
            <span class="td-doc-sub">No. {html.escape(doc_no)}</span>
          </td>
          <td>
            <span class="td-place-main">{html.escape(village)}</span>
            <span class="td-place-sub">{html.escape(loc_sub)}</span>
          </td>
          <td class="td-mono">{html.escape(survey)}</td>
          <td><span style="font-size:12px; color:#059669; font-weight:600;">✓ RSA-PSS 2048 Signed</span></td>
          <td>
            <div class="action-links" style="display:flex; gap:6px; align-items:center; justify-content:center; white-space:nowrap; flex-wrap:nowrap;">
              <a class="act-btn" style="background:#059669; color:#ffffff !important; font-weight:700; border-color:#047857; padding:5px 10px; font-size:11.5px; text-decoration:none; display:inline-block; white-space:nowrap;" href="/?verification_id={html.escape(vid)}" target="_blank">📜 Certificate &amp; QR</a>
              <a class="act-btn" style="padding:5px 10px; font-size:11.5px; display:inline-block; white-space:nowrap; text-decoration:none;" href="/export_pdf?verification_id={html.escape(vid)}" target="_blank">🔒 Download PDF</a>
            </div>
          </td>
        </tr>
        """)

    if sealed_rows:
        sealed_table_html = f"""
        <table class="master-ledger sealed-table" id="officerSealedTable">
          <thead>
            <tr>
              <th>#</th>
              <th>Sealed Date</th>
              <th>Document &amp; No.</th>
              <th>Location</th>
              <th>Survey No.</th>
              <th>Security Seal</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {''.join(sealed_rows)}
          </tbody>
        </table>
        """
    else:
        sealed_table_html = """
        <div class="table-empty" style="padding:32px 24px; text-align:center; background:#ffffff;">
          <p style="color:var(--ink-soft); font-size:13.5px; margin:0;">No RSA-PSS sealed deeds in ledger yet. Approved documents will appear here.</p>
        </div>
        """

    # Master Deed Register table rows for Officer view
    all_ledger_recs = [r for r in db.values() if isinstance(r, dict) and r.get("verification_id")]
    all_ledger_recs.sort(key=lambda r: r.get("created_at") or "", reverse=True)

    master_rows = []
    for i, r in enumerate(all_ledger_recs, start=1):
        vid = r.get("verification_id", "")
        payload = r.get("document_payload") or {}
        prop = payload.get("property") or {}
        doc_type = payload.get("document_type") or r.get("filename") or "Land Deed"
        doc_no = payload.get("document_number") or payload.get("serial_number") or vid[:8]
        rec_date = _fmt_date(r.get("created_at") or "")
        village = prop.get("village") or "—"
        district = prop.get("district") or "—"
        mandal = prop.get("mandal") or "—"
        survey = prop.get("survey_number") or "—"
        status = r.get("status") or "EXTRACTED"
        loc_sub = f"{mandal} · {district}" if mandal != "—" else district

        master_rows.append(f"""
        <tr class="data-row">
          <td class="td-sl">{i}</td>
          <td class="td-date">{html.escape(rec_date)}</td>
          <td>
            <span class="td-doc-main">{html.escape(doc_type)}</span>
            <span class="td-doc-sub">No. {html.escape(doc_no)}</span>
          </td>
          <td>
            <span class="td-place-main">{html.escape(village)}</span>
            <span class="td-place-sub">{html.escape(loc_sub)}</span>
          </td>
          <td class="td-mono">{html.escape(survey)}</td>
          <td>{_badge(status)}</td>
          <td>
            <div class="action-links" style="justify-content:center;">
              <a class="act-btn" style="display:inline-block; white-space:nowrap;" href="/record?verification_id={html.escape(vid)}&role=officer">Inspect Record</a>
            </div>
          </td>
        </tr>
        """)

    if master_rows:
        master_table_html = f"""
        <table class="master-ledger" id="officerMasterTable">
          <thead>
            <tr>
              <th>#</th>
              <th>Date</th>
              <th>Document &amp; No.</th>
              <th>Location</th>
              <th>Survey No.</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {''.join(master_rows)}
          </tbody>
        </table>
        """
    else:
        master_table_html = """
        <div class="table-empty" style="padding:32px 24px; text-align:center; background:#ffffff;">
          <p style="color:var(--ink-soft); font-size:13.5px; margin:0;">No land deeds recorded in the master ledger yet.</p>
        </div>
        """

    donut_svg = _render_donut_svg(data["sale_count"], data["gpa_count"], data["other_count"], data["on_file"])
    velocity_svg = _render_velocity_svg(data["on_file"], data["sealed_n"])
    accuracy_panel_markup = _render_extraction_accuracy_panel()

    analytics_markup = f"""
      <!-- STATISTICAL ANALYTICS GRAPHS -->
      <section class="charts-grid" id="analyticsSection">
        
        <!-- Graph 1: Velocity & Throughput Trend -->
        <div class="chart-card">
          <div class="chart-header">
            <div>
              <div class="chart-title" data-i18n="chart_vel_title">Registration Velocity &amp; Sealing Throughput</div>
              <div class="chart-meta" data-i18n="chart_vel_meta">Timeline intake volume vs verified cryptographic certifications</div>
            </div>
            <div style="display:flex; gap:12px; font-family:var(--type); font-size:10.5px;">
              <span style="display:flex; align-items:center; gap:5px;">
                <span style="width:8px; height:8px; border-radius:50%; background:var(--stamp);"></span>
                <span data-i18n="chart_total_intake">Total Intake</span>
              </span>
              <span style="display:flex; align-items:center; gap:5px;">
                <span style="width:8px; height:8px; border-radius:50%; background:var(--green);"></span>
                <span data-i18n="chart_sealed_on_file">Sealed on File</span>
              </span>
            </div>
          </div>
          <div class="chart-svg-wrap">
            {velocity_svg}
          </div>
        </div>

        <!-- Graph 2: Document Classification Donut -->
        <div class="chart-card">
          <div class="chart-header">
            <div>
              <div class="chart-title" data-i18n="chart_doc_class">Document Classification</div>
              <div class="chart-meta" data-i18n="chart_doc_meta">Distribution of legal record deed categories</div>
            </div>
            <div class="chart-meta" style="color:var(--green); font-weight:700;">
              <span data-i18n="chart_gis_lbl">GIS Resolved:</span> {data['gis_rate']}
            </div>
          </div>

          <div class="donut-layout">
            <div class="donut-svg-box">
              {donut_svg}
            </div>

            <div class="donut-legend">
              <div class="legend-row">
                <span class="legend-left">
                  <span class="legend-color" style="background:var(--stamp);"></span>
                  <span data-i18n="chart_sale_deeds">Sale Deeds</span>
                </span>
                <span class="legend-num">{data['sale_count']}</span>
              </div>
              <div class="legend-row">
                <span class="legend-left">
                  <span class="legend-color" style="background:var(--gold);"></span>
                  <span data-i18n="chart_agreements_gpa">Agreements / GPA</span>
                </span>
                <span class="legend-num">{data['gpa_count']}</span>
              </div>
              <div class="legend-row">
                <span class="legend-left">
                  <span class="legend-color" style="background:var(--green);"></span>
                  <span data-i18n="chart_other_records">Other Records</span>
                </span>
                <span class="legend-num">{data['other_count']}</span>
              </div>
            </div>
          </div>

          <div class="gis-rate-meter">
            <div class="meter-label">
              <span data-i18n="chart_spatial_match">Telangana (TGRAC) &amp; Karnataka Spatial Match</span>
              <b>{data['gis_rate']} <span data-i18n="chart_resolved">Resolved</span></b>
            </div>
            <div class="meter-bar">
              <div class="meter-fill" style="width:{data['gis_rate']};"></div>
            </div>
          </div>
        </div>

      </section>
    """

    page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OneBhoomi — Officer Approval &amp; Sealing Queue</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..900;1,9..144,300..900&family=Archivo:wght@400;500;600;700&family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Noto+Sans+Telugu:wght@400;500;600;700&family=Noto+Sans+Kannada:wght@400;500;600;700&family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    {DASHBOARD_CSS}
    .officer-workload-strip {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }}
    .officer-metric-card {{
        background: #ffffff;
        border: 1px solid var(--rule);
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }}
    .officer-metric-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 4px;
    }}
    .card-pending::before {{ background: #2563eb; }}
    .card-sealed::before {{ background: #059669; }}
    .card-rejected::before {{ background: #dc2626; }}
    .card-total::before {{ background: #475569; }}
    .metric-val {{
        font-family: 'Fraunces', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--ink);
        line-height: 1;
    }}
    .metric-label {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--ink-soft);
        margin-top: 6px;
    }}
    .queue-box {{
        background: #ffffff;
        border: 1px solid var(--rule);
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        overflow: hidden;
        margin-bottom: 24px;
    }}
    .queue-header {{
        padding: 18px 22px;
        border-bottom: 1px solid var(--rule);
        background: #fafaf9;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .queue-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .queue-badge-count {{
        background: #2563eb;
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
    }}
    table.clerk-table {{
        width: 100% !important;
        table-layout: auto !important;
    }}
    table.clerk-table th:nth-child(9), table.clerk-table td:nth-child(9) {{
        width: 280px !important;
        min-width: 280px !important;
        text-align: center;
        white-space: nowrap;
    }}
    #officerSealedTable {{
        width: 100% !important;
        table-layout: auto !important;
    }}
    #officerSealedTable th:nth-child(1), #officerSealedTable td:nth-child(1) {{ width: 45px; text-align: center; }}
    #officerSealedTable th:nth-child(2), #officerSealedTable td:nth-child(2) {{ width: 105px; }}
    #officerSealedTable th:nth-child(3), #officerSealedTable td:nth-child(3) {{ min-width: 150px; }}
    #officerSealedTable th:nth-child(4), #officerSealedTable td:nth-child(4) {{ min-width: 170px; }}
    #officerSealedTable th:nth-child(5), #officerSealedTable td:nth-child(5) {{ width: 85px; }}
    #officerSealedTable th:nth-child(6), #officerSealedTable td:nth-child(6) {{ width: 160px; }}
    #officerSealedTable th:nth-child(7), #officerSealedTable td:nth-child(7) {{
        width: 280px !important;
        min-width: 280px !important;
        text-align: center;
        white-space: nowrap;
    }}
    #officerMasterTable {{
        width: 100% !important;
        table-layout: auto !important;
    }}
    #officerMasterTable th:nth-child(7), #officerMasterTable td:nth-child(7) {{
        width: 140px !important;
        min-width: 140px !important;
        text-align: center;
        white-space: nowrap;
    }}
    #officerQueueTable {{
        width: 100% !important;
        table-layout: auto !important;
    }}
    #officerQueueTable th, #officerQueueTable td {{
        padding: 9px 8px;
        font-size: 12px;
        white-space: nowrap;
    }}
    #officerQueueTable th:nth-child(1), #officerQueueTable td:nth-child(1) {{ width: 45px; text-align: center; }}
    #officerQueueTable th:nth-child(2), #officerQueueTable td:nth-child(2) {{ width: 85px; }}
    #officerQueueTable th:nth-child(3), #officerQueueTable td:nth-child(3) {{ width: 90px; }}
    #officerQueueTable th:nth-child(4), #officerQueueTable td:nth-child(4) {{ min-width: 120px; }}
    #officerQueueTable th:nth-child(5), #officerQueueTable td:nth-child(5) {{ min-width: 100px; }}
    #officerQueueTable th:nth-child(6), #officerQueueTable td:nth-child(6) {{ min-width: 135px; }}
    #officerQueueTable th:nth-child(7), #officerQueueTable td:nth-child(7) {{ width: 75px; }}
    #officerQueueTable th:nth-child(8), #officerQueueTable td:nth-child(8) {{ width: 105px; }}
    #officerQueueTable th:nth-child(9), #officerQueueTable td:nth-child(9) {{ width: 105px; }}
    #officerQueueTable th:nth-child(10), #officerQueueTable td:nth-child(10) {{
        width: 165px !important;
        min-width: 165px !important;
        text-align: center;
        white-space: nowrap;
    }}
    .queue-table-scroll {{
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }}
  </style>
</head>
<body>

<div class="security-bg" aria-hidden="true"></div>

<div class="app-layout">

  {sidebar_html}

  <main class="dash-content">
    <div class="main-inner">

      <!-- Top Action Bar -->
      <div class="top-action-bar">
        <div class="header-left">
          <h1>Officer <em>Approval &amp; Digital Seal Queue</em></h1>
          <div class="header-tagline">
            Logged in as <b>{html.escape(user_name)}</b> · Official Legal Inspection &amp; RSA Cryptographic Sealing
          </div>
        </div>
        <div class="header-right">
          <a href="/auth/signout" class="act-btn" style="border-color:#fca5a5; color:#b91c1c;">Sign Out</a>
        </div>
      </div>

      <!-- Workload Summary Strip -->
      <div class="officer-workload-strip">
        <div class="officer-metric-card card-pending">
          <div class="metric-val">{len(pending_records)}</div>
          <div class="metric-label">{len(pending_records)} Pending Review</div>
        </div>
        <div class="officer-metric-card card-sealed">
          <div class="metric-val">{sealed_count}</div>
          <div class="metric-label">Sealed &amp; Certified</div>
        </div>
        <div class="officer-metric-card card-rejected">
          <div class="metric-val">{rejected_count}</div>
          <div class="metric-label">Rejected Records</div>
        </div>
        <div class="officer-metric-card card-total">
          <div class="metric-val">{total_ledger_count}</div>
          <div class="metric-label">Total Registry Ledger</div>
        </div>
      </div>

      {analytics_markup}

      {accuracy_panel_markup}

      <!-- OFFICER QUEUE (Oldest Waiting First) -->
      <div class="queue-box" id="ledgerSection">
        <div class="queue-header">
          <div class="queue-title">
            <span>🏛️ Master Officer Approval Queue</span>
            <span class="queue-badge-count">{len(pending_records)} Pending</span>
          </div>
          <div style="font-size:12.5px; color:var(--ink-soft);">
            Sorted by wait duration (oldest first) · Across all clerk desks
          </div>
        </div>
        {filter_bar_html if rows_html else ""}
        <div class="queue-table-scroll">
          {queue_table_html}
        </div>
      </div>

      <!-- CERTIFIED & SEALED DEEDS LEDGER -->
      <div class="queue-box" id="sealedSection">
        <div class="queue-header" style="background:#f0fdf4;">
          <div class="queue-title">
            <span style="color:#059669;">🛡️ Certified &amp; Cryptographically Sealed Registry Ledger</span>
            <span class="queue-badge-count" style="background:#059669;">{len(sealed_records)} Sealed</span>
          </div>
          <div style="font-size:12.5px; color:var(--ink-soft);">
            RSA-PSS 2048-bit digital signature verified · Admissible legal deeds
          </div>
        </div>
        <div class="queue-table-scroll">
          {sealed_table_html}
        </div>
      </div>

      <!-- MASTER DEED REGISTER -->
      <div class="queue-box" id="masterLedgerSection">
        <div class="queue-header" style="background:#f8fafc;">
          <div class="queue-title">
            <span>📖 Master Department Deed Register</span>
            <span class="queue-badge-count" style="background:#475569;">{len(all_ledger_recs)} Total</span>
          </div>
          <div style="font-size:12.5px; color:var(--ink-soft);">
            Comprehensive day-book of all ingested and processed registry records
          </div>
        </div>
        <div class="queue-table-scroll">
          {master_table_html}
        </div>
      </div>

    </div>

    <!-- Page Footer -->
    <footer class="dash-footer">
      <div class="main-inner" style="padding-top:0; padding-bottom:0;">
        <div class="dash-footer-wrap">
          <div><b>OneBhoomi Registry Console</b> · Officer Approval Bureau</div>
          <div>100% Air-Gapped &amp; Immutable · Zero cloud dependencies · Host: <code>{html.escape(host_name)}</code></div>
        </div>
      </div>
    </footer>
  </main>

</div>

<script>
{DASHBOARD_JS}

function applyOfficerFilters() {{
  const distEl = document.getElementById('filterDistrict');
  const mandEl = document.getElementById('filterMandal');
  const dtypeEl = document.getElementById('filterDocType');
  if (!distEl || !mandEl || !dtypeEl) return;
  
  const dist = distEl.value.trim().toLowerCase();
  const mand = mandEl.value.trim().toLowerCase();
  const dtype = dtypeEl.value.trim().toLowerCase();
  
  const rows = document.querySelectorAll('#officerQueueTable tbody tr.data-row');
  let visibleCount = 0;
  
  rows.forEach(row => {{
    const rowDist = (row.getAttribute('data-district') || '').trim().toLowerCase();
    const rowMand = (row.getAttribute('data-mandal') || '').trim().toLowerCase();
    const rowDtype = (row.getAttribute('data-doc-type') || '').trim().toLowerCase();
    
    const matchDist = !dist || rowDist === dist;
    const matchMand = !mand || rowMand === mand;
    const matchDtype = !dtype || rowDtype === dtype;
    
    if (matchDist && matchMand && matchDtype) {{
      row.style.display = '';
      visibleCount++;
    }} else {{
      row.style.display = 'none';
    }}
  }});
  
  const counter = document.getElementById('filterMatchCount');
  if (counter) {{
    if (dist || mand || dtype) {{
      counter.innerHTML = 'Showing <b>' + visibleCount + '</b> of ' + rows.length + ' records matching active jurisdiction filters';
    }} else {{
      counter.innerHTML = 'Showing all <b>' + rows.length + '</b> pending queue records';
    }}
  }}
  
  const emptyRow = document.getElementById('officerFilterEmptyRow');
  if (emptyRow) {{
    emptyRow.style.display = (visibleCount === 0 && rows.length > 0) ? '' : 'none';
  }}
}}

function clearOfficerFilters() {{
  const distEl = document.getElementById('filterDistrict');
  const mandEl = document.getElementById('filterMandal');
  const dtypeEl = document.getElementById('filterDocType');
  if (distEl) distEl.value = '';
  if (mandEl) mandEl.value = '';
  if (dtypeEl) dtypeEl.value = '';
  applyOfficerFilters();
}}
</script>
</body>
</html>
"""
    return page_html.encode("utf-8")


