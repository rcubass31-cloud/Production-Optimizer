# -*- coding: utf-8 -*-
"""
============================================================================
 PRODUCTION OPTIMIZER — Soil Nailing / Pernos de Anclaje
 Factory Physics (Hopp & Spearman)
 Interfaz web: Streamlit  (migración 1:1 desde production_v13.py / Tkinter)
----------------------------------------------------------------------------
 · Barra lateral : panel de control (parámetros por estación + flujo global)
 · Área principal: 4 pestañas
       Vista 1 : Flow Analysis (TH y CT vs WIP)
       Vista 2 : Capacity Utilization
       Vista 3 : Métricas OS + tablas por centro de proceso
       Vista 4 : Cycle Time Analysis (descomposición RPT/QT/BT/MT/SDT)

 Ejecución local :  streamlit run app.py
 Requisitos      :  streamlit, numpy, pandas, simpy, matplotlib
============================================================================
"""

import io

import numpy as np
import pandas as pd
import simpy
import streamlit as st

import matplotlib
matplotlib.use("Agg")                        # backend no interactivo (servidor)
from matplotlib.figure import Figure


# ============================================================================
#  A)  MOTOR ANALÍTICO Y DE SIMULACIÓN
# ============================================================================
NOMBRES_CORTOS = ["Perforación", "Colocación acero", "Inyección lechada"]


def calcular_metricas_nucleo(procesos, demand):
    """RPT = ΣTe ; BNR = min(m/Te) ; W0 = BNR·RPT ; MINWIP = d(W0-1)/(BNR-d)."""
    Te    = np.array([p["Te"] for p in procesos], dtype=float)
    m     = np.array([p["servers"] for p in procesos], dtype=float)
    tasas = m / Te

    RPT = Te.sum()
    BNR = tasas.min()
    idx_bn = int(np.argmin(tasas))
    W0  = BNR * RPT

    if demand >= BNR:
        raise ValueError(
            f"Sistema INESTABLE: demanda d = {demand:.3f} u/h ≥ BNR = {BNR:.3f} u/h. "
            "El cuello de botella no puede sostener la demanda. "
            "Reduzca d, agregue servidores o baje Te en el cuello de botella."
        )
    minwip = demand * (W0 - 1.0) / (BNR - demand)
    return {"Te": Te, "tasas": tasas, "RPT": RPT, "BNR": BNR,
            "W0": W0, "MINWIP": minwip, "idx_bottleneck": idx_bn}


def calcular_curvas_flujo(wip_array, RPT, BNR):
    """Vectores Best/PWC de Throughput y Cycle Time vs WIP.
    Condiciones de borde físicas:
      · Throughput: en WIP = 0 no hay unidades en el sistema -> TH = 0
        (las curvas de TH nacen en el origen 0,0).
      · Cycle Time: el límite físico inferior de permanencia en la línea es el
        Raw Process Time (RPT). Por eso el CT NO cae a 0, sino que se acota
        inferiormente en RPT (una unidad siempre tarda al menos RPT en cruzar)."""
    w  = wip_array.astype(float)
    W0 = BNR * RPT

    # --- Throughput ---
    best_th = np.where(w <= W0, w / RPT, BNR)
    with np.errstate(divide="ignore", invalid="ignore"):
        pwc_th = np.where(w > 0, (w / (W0 + w - 1.0)) * BNR, 0.0)
    # En WIP = 0 el Throughput nace explícitamente en el origen (0,0)
    best_th = np.where(w == 0, 0.0, best_th)
    pwc_th  = np.where(w == 0, 0.0, pwc_th)

    # --- Cycle Time (acotado inferiormente en RPT, nunca cae por debajo) ---
    best_ct = np.where(w <= W0, RPT, w / BNR)                 # meseta en RPT para w <= W0
    pwc_ct  = np.maximum(RPT, RPT + (w - 1.0) / BNR)          # cota mínima = RPT

    return {"wip": w, "best_th": best_th, "best_ct": best_ct,
            "pwc_th": pwc_th, "pwc_ct": pwc_ct}


def calcular_utilizacion(procesos, demand):
    """u_i = d / (servers_i / Te_i) · 100 %."""
    Te  = np.array([p["Te"] for p in procesos], dtype=float)
    m   = np.array([p["servers"] for p in procesos], dtype=float)
    cap = m / Te
    return demand / cap * 100.0, cap


def calcular_qt_analitico_conwip(procesos, rate_flow, scv_arrival):
    """Queue Time medio por estación evaluado a una tasa de flujo dada
    (`rate_flow`), vía la aproximación VUT (Kingman / Sakasegawa) de
    Factory Physics:
        CTq_i ≈ [(ca²_i + ce²_i) / 2] · [u_i^(√(2(m_i+1))−1) / (m_i·(1−u_i))] · Te_i
    La variabilidad se propaga estación a estación con la ecuación de enlace
    (linking equation) de Hopp & Spearman:
        ca²_{i+1} = u_i²·ce²_i + (1 − u_i²)·ca²_i
    En régimen CONWIP la tasa procesada NO es la demanda externa `d`, sino el
    rendimiento restringido por el WIP (TH_CONWIP); ese valor debe pasarse en
    `rate_flow`. Devuelve (qt_por_estacion, utilizacion_%_por_estacion)."""
    n   = len(procesos)
    Te  = np.array([p["Te"] for p in procesos], dtype=float)
    m   = np.array([p["servers"] for p in procesos], dtype=float)
    ce2 = np.array([p["scv"] for p in procesos], dtype=float)
    cap = m / Te
    u = np.clip(rate_flow / cap, 0.0, 0.995)   # se acota <1 para evitar CTq -> ∞

    qt = np.zeros(n)
    ca2 = scv_arrival
    for i in range(n):
        if m[i] <= 1:
            factor_espera = u[i] / (1.0 - u[i])
        else:
            expo = np.sqrt(2.0 * (m[i] + 1.0)) - 1.0
            factor_espera = (u[i] ** expo) / (m[i] * (1.0 - u[i]))
        qt[i] = ((ca2 + ce2[i]) / 2.0) * factor_espera * Te[i]
        ca2 = u[i] ** 2 * ce2[i] + (1.0 - u[i] ** 2) * ca2   # ecuación de enlace

    return qt, u * 100.0


def calcular_bt_mt_sdt(procesos):
    """Batch Time (BT), Move Time (MT) y Shift Diff Time (SDT) por estación,
        BT_i = ((K_i - 1) * Te_i) / 2.0        (tiempo de espera por lote)
    Los totales del sistema son la suma de cada componente por estación."""
    Te = np.array([p["Te"] for p in procesos], dtype=float)
    K  = np.array([p["K"]  for p in procesos], dtype=float)
    MT = np.array([p["MT"]  for p in procesos], dtype=float)
    SDT = np.array([p["SDT"] for p in procesos], dtype=float)

    BT_i = ((K - 1.0) * Te) / 2.0
    return {"BT_i": BT_i, "BT_total": float(BT_i.sum()),
            "MT_i": MT, "MT_total": float(MT.sum()),
            "SDT_i": SDT, "SDT_total": float(SDT.sum())}


def metricas_conwip(wip, RPT, BNR):
    """Predicción CONWIP vía PWC (cumple Ley de Little)."""
    W0 = BNR * RPT
    th = (wip / (W0 + wip - 1.0)) * BNR
    ct = RPT + (wip - 1.0) / BNR
    return {"WIP": wip, "TH": th, "CT": ct, "RPT": RPT,
            "QT": ct - RPT, "BT": 0.0, "MT": 0.0, "SDT": 0.0}


def _gamma_params(media, scv):
    """Gamma con media y SCV objetivo: shape = 1/SCV, scale = media·SCV."""
    return 1.0 / scv, media * scv


def simular_push(procesos, demand, scv_arrival, n_jobs, warmup, seed=42):
    """DES estocástica en régimen PUSH (SimPy). Mide QT por estación y CT."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    recursos = [simpy.Resource(env, capacity=p["servers"]) for p in procesos]
    nst = len(procesos)

    reg_id, reg_ct, reg_fin, reg_qt, reg_st = [], [], [], [], []
    gp_serv = [_gamma_params(p["Te"], p["scv"]) for p in procesos]

    # Medición de la ocupación REAL de cada recurso (para la utilización PUSH
    # medida por la DES, en lugar de la teórica d/cap):
    #   u_i = tiempo_ocupado_recurso_i / (m_i · ventana_observada)
    busy   = np.zeros(nst)                                  # tiempo-servidor ocupado por estación
    obs    = {"t_start": None}                              # inicio de la ventana post-warmup
    m_serv = np.array([p["servers"] for p in procesos], dtype=float)

    def flujo_perno(jid):
        t_entrada = env.now
        if jid == warmup and obs["t_start"] is None:
            obs["t_start"] = t_entrada                      # arranca la ventana de observación
        qt_row = np.zeros(nst)
        st_row = np.zeros(nst)
        for i in range(nst):
            t0 = env.now
            with recursos[i].request() as req:
                yield req
                qt_row[i] = env.now - t0
                serv = rng.gamma(*gp_serv[i])
                st_row[i] = serv
                if jid >= warmup:                           # ocupación medida solo en estado estable
                    busy[i] += serv
                yield env.timeout(serv)
        reg_id.append(jid); reg_qt.append(qt_row); reg_st.append(st_row)
        reg_ct.append(env.now - t_entrada); reg_fin.append(env.now)

    def fuente():
        sh_a, sc_a = _gamma_params(1.0 / demand, scv_arrival)
        for jid in range(n_jobs):
            yield env.timeout(rng.gamma(sh_a, sc_a))
            env.process(flujo_perno(jid))

    env.process(fuente())
    env.run()

    ids = np.array(reg_id); ct = np.array(reg_ct); fin = np.array(reg_fin)
    qt = np.vstack(reg_qt); st_arr = np.vstack(reg_st)
    mask = ids >= warmup
    qt_est = qt[mask].mean(axis=0)
    st_est = st_arr[mask].mean(axis=0)
    ct_sim = ct[mask].mean()

    # --- FIX: Throughput en régimen PUSH -----------------------------------
    # En PUSH el sistema es "impulsado por la demanda": en estado estable
    # (sistema estable, d < BNR) el Throughput observado converge exactamente
    # a la tasa de llegada d, por Ley de Conservación de flujo. No se usa el
    # throughput medido en la ventana finita de simulación (que solo es un
    # estimador ruidoso de d), sino la demanda misma.
    th_push = demand
    # PUSHWIP = d · CT_push_simulado (Ley de Little aplicada al CT simulado)
    pushwip = demand * ct_sim

    # --- Utilización PUSH MEDIDA por el recurso (ocupación real del servidor) ---
    # En un sistema PUSH estable (d < BNR) converge a d/cap por conservación de
    # flujo, pero aquí se reporta el valor empírico observado en la DES.
    t_start = obs["t_start"] if obs["t_start"] is not None else fin.min()
    t_obs   = max(fin.max() - t_start, 1e-9)                # ventana de observación
    util_push_sim = busy / (m_serv * t_obs) * 100.0        # % por estación

    return {"CT_push": ct_sim, "TH_push": th_push, "PUSHWIP": pushwip,
            "QT_estacion": qt_est, "ST_estacion": st_est,
            "QT_total": qt_est.sum(), "RPT_sim": st_est.sum(),
            "util_push_sim": util_push_sim,
            "n_completados": int(mask.sum())}


# ============================================================================
#  B)  CONFIGURACIÓN VISUAL Y VALORES POR DEFECTO
# ============================================================================
st.set_page_config(
    page_title="Production Optimizer — Soil Nailing",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta. Los colores de las CURVAS se conservan idénticos al script de
# escritorio validado, para que las figuras del artículo sean comparables.
# Los tonos de la interfaz (chrome) siguen una línea azul marino / gris
# ejecutivo, con rojo reservado al cuello de botella.
COL = {
    "navy":    "#1f3a5f",   # azul marino  — títulos y encabezados
    "slate":   "#5a6b7d",   # gris ejecutivo — texto secundario
    "line":    "#d8dee5",   # bordes tenues
    "card":    "#f7f9fb",   # fondo de tarjeta
    "accent":  "#c0392b",   # rojo — cuello de botella / valores destacados
    "best_th": "#2ecc71", "pwc_th": "#e74c3c", "demand": "#27ae60",
    "best_ct": "#8e44ad", "pwc_ct": "#e67e22",
    "minwip":  "#7f8c8d", "pushwip": "#2980b9", "wip_asg": "#c0392b",
    "bar": "#e67e22", "bar_bn": "#c0392b",
}

# Valores por defecto de los 3 procesos: (nombre largo, Te, servidores, SCV)
PROC_DEFAULTS = [
    ("Perforación y soplado de barreno", 0.34, 1, "1.5 (Alta)"),
    ("Colocación manual de acero",       0.08, 1, "0.5 (Baja)"),
    ("Inyección de lechada de cemento",  0.12, 1, "1.0 (Media/PWC)"),
]
SCV_OPCIONES = ["0.5 (Baja)", "1.0 (Media/PWC)", "1.5 (Alta)"]

# Batch / Move / Shift Diff por estación: (K, MT, SDT)
BATCH_DEFAULTS = [(1, 0.00, 0.00), (1, 0.00, 0.00), (1, 0.00, 0.00)]

# Defaults globales
GLOBAL_DEFAULTS = {"d": 2.10, "scv_a": 1.0, "wip": 8, "n_iter": 10000, "th_act": 2.30}

# Default del nivel de CONWIP a evaluar (Vista 3 · Bloque 2)
CONWIP_EVAL_DEFAULT = 3

st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{
          font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      }}
      .block-container {{ padding-top: 2.0rem; padding-bottom: 2.5rem; }}
      h1, h2, h3, h4 {{ color: {COL['navy']}; letter-spacing: .2px; }}
      .po-title {{
          font-size: 1.55rem; font-weight: 700; color: {COL['navy']};
          margin-bottom: .15rem;
      }}
      .po-sub {{
          font-size: .92rem; color: {COL['slate']}; margin-bottom: 1.0rem;
      }}
      .po-block {{
          font-size: 1.05rem; font-weight: 700; color: {COL['navy']};
          border-left: 3px solid {COL['navy']}; padding-left: .55rem;
          margin: 1.1rem 0 .55rem 0;
      }}
      .po-note {{ font-size: .82rem; color: {COL['slate']}; font-style: italic; }}
      /* Tarjetas de métrica: borde tenue, fondo claro, valor en rojo ejecutivo */
      div[data-testid="stMetric"] {{
          background: {COL['card']};
          border: 1px solid {COL['line']};
          border-radius: 6px;
          padding: 12px 14px 10px 14px;
      }}
      div[data-testid="stMetricLabel"] p {{
          font-size: .74rem !important; font-weight: 700 !important;
          color: {COL['navy']} !important; text-transform: uppercase;
          letter-spacing: .3px; line-height: 1.15;
      }}
      div[data-testid="stMetricValue"] {{
          font-size: 1.30rem !important; font-weight: 700 !important;
          color: {COL['accent']} !important;
      }}
      section[data-testid="stSidebar"] {{ border-right: 1px solid {COL['line']}; }}
      section[data-testid="stSidebar"] h2 {{ font-size: 1.0rem; }}
      .stTabs [data-baseweb="tab"] {{
          font-size: .90rem; font-weight: 600; color: {COL['slate']};
      }}
      .stTabs [aria-selected="true"] {{ color: {COL['navy']}; }}
      div[data-testid="stDataFrame"] {{ border: 1px solid {COL['line']}; border-radius: 6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
#  C)  PANEL DE CONTROL (BARRA LATERAL)
# ============================================================================
def leer_config_sidebar():
    """Recrea el panel de control izquierdo del ejecutable Tkinter.
    Devuelve (cfg, calcular_pulsado). Los number_input ya validan rangos, de
    modo que se replican las mismas restricciones del script original."""
    sb = st.sidebar
    sb.markdown(f"<div class='po-title' style='font-size:1.05rem'>PANEL DE CONTROL</div>",
                unsafe_allow_html=True)
    sb.caption("Soil Nailing / Pernos de anclaje — Factory Physics")

    procesos = []
    for i, (nombre_largo, te_d, m_d, scv_d) in enumerate(PROC_DEFAULTS):
        k_d, mt_d, sdt_d = BATCH_DEFAULTS[i]
        with sb.expander(f"{i+1}. {NOMBRES_CORTOS[i]}", expanded=True):
            st.caption(nombre_largo)
            te = st.number_input("Tiempo medio Te (h)", min_value=0.001, value=float(te_d),
                                 step=0.01, format="%.3f", key=f"te_{i}")
            m = st.number_input("N.º de servidores", min_value=1, value=int(m_d),
                                step=1, key=f"m_{i}")
            scv_txt = st.selectbox("Variabilidad SCV_e", SCV_OPCIONES,
                                   index=SCV_OPCIONES.index(scv_d), key=f"scv_{i}")
            k = st.number_input("Tamaño de lote K (u)", min_value=1, value=int(k_d),
                                step=1, key=f"k_{i}")
            mt = st.number_input("Move Time MT (h)", min_value=0.0, value=float(mt_d),
                                 step=0.01, format="%.3f", key=f"mt_{i}")
            sdt = st.number_input("Shift Diff Time SDT (h)", min_value=0.0, value=float(sdt_d),
                                  step=0.01, format="%.3f", key=f"sdt_{i}")
        procesos.append({"nombre": nombre_largo, "Te": float(te), "servers": int(m),
                         "scv": float(scv_txt.split()[0]), "K": int(k),
                         "MT": float(mt), "SDT": float(sdt)})

    with sb.expander("Flujo global", expanded=True):
        d = st.number_input("Tasa de demanda d (u/h)", min_value=0.001,
                            value=GLOBAL_DEFAULTS["d"], step=0.05, format="%.3f", key="g_d")
        scv_a = st.number_input("SCV_a de llegadas", min_value=0.001,
                                value=GLOBAL_DEFAULTS["scv_a"], step=0.1, format="%.2f",
                                key="g_scva")
        wip = st.number_input("WIP asignado (u)", min_value=1,
                              value=GLOBAL_DEFAULTS["wip"], step=1, key="g_wip")
        n_iter = st.number_input("Iteraciones DES (pernos)", min_value=100,
                                 value=GLOBAL_DEFAULTS["n_iter"], step=500, key="g_iter")
        th_act = st.number_input("Throughput actual de campo TH_act (u/h)", min_value=0.001,
                                 value=GLOBAL_DEFAULTS["th_act"], step=0.05, format="%.3f",
                                 key="g_thact")

    sb.markdown("")
    calcular = sb.button("Calcular y Simular", type="primary", width="stretch")
    sb.caption("La simulación DES se ejecuta en el servidor; con 10 000 pernos "
               "toma unos pocos segundos.")

    warmup = min(500, max(20, int(n_iter) // 5))
    cfg = {"procesos": procesos, "d": float(d), "scv_a": float(scv_a),
           "wip": int(wip), "n_iter": int(n_iter), "warmup": warmup,
           "th_act": float(th_act)}
    return cfg, calcular


# ============================================================================
#  D)  ORQUESTACIÓN DEL CÁLCULO (equivalente al _worker del ejecutable)
# ============================================================================
def ejecutar_analisis(cfg):
    """Reproduce exactamente la secuencia de cálculo del hilo de trabajo Tkinter."""
    nucleo = calcular_metricas_nucleo(cfg["procesos"], cfg["d"])
    wip_arr = np.arange(0, 31)
    curvas = calcular_curvas_flujo(wip_arr, nucleo["RPT"], nucleo["BNR"])
    util, cap = calcular_utilizacion(cfg["procesos"], cfg["d"])
    conwip = metricas_conwip(cfg["wip"], nucleo["RPT"], nucleo["BNR"])
    push = simular_push(cfg["procesos"], cfg["d"], cfg["scv_a"],
                        cfg["n_iter"], cfg["warmup"])

    # --- Batch Time (BT), Move Time (MT) y Shift Diff Time (SDT) ---
    # CT_total = RPT + QT + BT + MT + SDT
    bms = calcular_bt_mt_sdt(cfg["procesos"])
    push["BT_total"]  = bms["BT_total"]
    push["MT_total"]  = bms["MT_total"]
    push["SDT_total"] = bms["SDT_total"]
    push["CT_push_total"] = (push["CT_push"] + bms["BT_total"]
                             + bms["MT_total"] + bms["SDT_total"])
    # PUSHWIP recalculado con el CT total (Ley de Little: WIP = d · CT)
    push["PUSHWIP_total"] = cfg["d"] * push["CT_push_total"]

    # --- Ley de Little aplicada al TH actual de campo (dato empírico) ---
    ct_actual_campo = cfg["wip"] / cfg["th_act"]
    campo_actual = {"TH_act": cfg["th_act"], "CT_act": ct_actual_campo,
                    "WIP": cfg["wip"]}

    return {"nucleo": nucleo, "curvas": curvas, "util": util, "cap": cap,
            "conwip": conwip, "push": push, "cfg": cfg, "campo_actual": campo_actual}


# ============================================================================
#  E)  GRÁFICOS (Matplotlib — mismas figuras que la versión de escritorio)
# ============================================================================
def fig_flow(res):
    """Vista 1 — Flow Analysis: TH y CT vs WIP (Best / PWC), demanda, líneas
    verticales de MINWIP / PUSHWIP / WIP asignado y los dos puntos operativos
    reales de campo (TH actual y CT actual)."""
    c = res["curvas"]; nuc = res["nucleo"]; cfg = res["cfg"]; push = res["push"]
    campo = res["campo_actual"]
    w = c["wip"]

    fig = Figure(figsize=(11.0, 5.4), dpi=110, facecolor="white")
    ax = fig.add_subplot(111)

    ax.plot(w, c["best_th"], color=COL["best_th"], lw=2.2, label="Best TH")
    ax.plot(w, c["pwc_th"],  color=COL["pwc_th"],  lw=2.2, label="Predicted TH (PWC)")
    ax.axhline(cfg["d"], color=COL["demand"], lw=1.8, ls="--",
               label=f"Demanda = {cfg['d']:.2f} u/h")
    ax.set_xlabel("Work-in-Process (pernos)", fontsize=9)
    ax.set_ylabel("Throughput (pernos/h)", fontsize=9, color=COL["accent"])
    ax.tick_params(axis="y", labelcolor=COL["accent"], labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0, max(nuc["BNR"] * 1.25, cfg["d"] * 1.3, campo["TH_act"] * 1.2))

    ax2 = ax.twinx()
    ax2.plot(w, c["best_ct"], color=COL["best_ct"], lw=2.2, label="Best CT")
    ax2.plot(w, c["pwc_ct"],  color=COL["pwc_ct"],  lw=2.2, label="Predicted CT (PWC)")
    ax2.set_ylabel("Cycle Time (h)", fontsize=9, color=COL["best_ct"])
    ax2.tick_params(axis="y", labelcolor=COL["best_ct"], labelsize=8)
    ax2.set_ylim(0, max(c["pwc_ct"].max(), campo["CT_act"]) * 1.1)

    # Origen (0,0) visible en ambos ejes gemelos
    ax.set_xlim(-0.5, w.max() + 1.0)
    ax2.set_xlim(-0.5, w.max() + 1.0)

    ax.axvline(nuc["MINWIP"], color=COL["minwip"], lw=1.6, ls=":",
               label=f"MINWIP = {nuc['MINWIP']:.2f}")
    ax.axvline(push["PUSHWIP"], color=COL["pushwip"], lw=1.6, ls="-.",
               label=f"PUSHWIP sim = {push['PUSHWIP']:.2f}")
    ax.axvline(cfg["wip"], color=COL["wip_asg"], lw=1.2, ls="--", alpha=0.7,
               label=f"WIP asignado = {cfg['wip']}")

    # Puntos operativos ACTUALES DE CAMPO (dato empírico)
    ax.plot(cfg["wip"], campo["TH_act"], marker="o", markersize=12,
            markerfacecolor="#2980b9", markeredgecolor="black", markeredgewidth=0.8,
            linestyle="None", zorder=6, label="TH actual")
    ax2.plot(cfg["wip"], campo["CT_act"], marker="o", markersize=11,
             markerfacecolor="red", markeredgecolor="black", markeredgewidth=1.0,
             linestyle="None", zorder=6, label="CT actual")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right", framealpha=0.9)
    ax.set_title("Flow Analysis — Throughput y Cycle Time vs WIP",
                 fontsize=11, fontweight="bold", color=COL["navy"])
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.90, bottom=0.12)
    return fig


def fig_util(res):
    """Vista 2 — Capacity Utilization por centro de proceso (rojo = cuello)."""
    util = res["util"]; idx = res["nucleo"]["idx_bottleneck"]
    fig = Figure(figsize=(11.0, 5.0), dpi=110, facecolor="white")
    ax = fig.add_subplot(111)
    colores = [COL["bar_bn"] if i == idx else COL["bar"] for i in range(len(util))]
    barras = ax.bar(NOMBRES_CORTOS, util, color=colores, edgecolor="black", lw=0.6)
    for b, v in zip(barras, util):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(100, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Utilización de Capacidad (%)", fontsize=9)
    ax.set_ylim(0, max(100, util.max() * 1.2))
    ax.set_title("Capacity Utilization por Centro de Proceso  (rojo = cuello de botella)",
                 fontsize=11, fontweight="bold", color=COL["navy"])
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def fig_ct_percent(push):
    """Vista 4 — Barra horizontal apilada al 100 % (Percent of Cycle Time),
    descompone el CT simulado en RPT + QT + BT + MT + SDT."""
    rpt = push["RPT_sim"]; qt = push["QT_total"]
    bt = push["BT_total"]; mt = push["MT_total"]; sdt = push["SDT_total"]
    ct_total = max(push["CT_push_total"], 1e-9)

    componentes = [
        ("Raw Process Time - RPT",  rpt, "#2ecc71"),   # verde
        ("Queue Time - QT",         qt,  "#2980b9"),   # azul
        ("Batch Time - BT",         bt,  "#e67e22"),   # naranja
        ("Move Time - MT",          mt,  "#f1c40f"),   # amarillo
        ("Shift Diff. Time - SDT",  sdt, "#c0392b"),   # rojo
    ]

    fig = Figure(figsize=(11.0, 2.9), dpi=110, facecolor="white")
    ax = fig.add_subplot(111)
    y = 0
    izquierda = 0.0
    for nombre, valor, color in componentes:
        pct = valor / ct_total * 100.0
        ax.barh(y, pct, left=izquierda, color=color, edgecolor="white",
                height=0.5, label=nombre)
        if pct >= 4.0:                                  # etiqueta si el segmento es visible
            ax.text(izquierda + pct / 2.0, y, f"{pct:.1f}%",
                    ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
        izquierda += pct

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([y])
    ax.set_yticklabels(["PUSH CT"], fontsize=9)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.set_xlabel(f"Porcentaje del Cycle Time (%)   ·   CT total simulado = "
                  f"{ct_total:.3f} h", fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=5, fontsize=7.5, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.32)
    return fig


def descargar_png(fig, nombre, etiqueta="Descargar figura (PNG 300 dpi)"):
    """Exporta la figura a PNG de alta resolución para el artículo."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    st.download_button(etiqueta, data=buf.getvalue(), file_name=nombre,
                       mime="image/png")


# ============================================================================
#  F)  TABLAS POR CENTRO DE PROCESO
# ============================================================================
COLS_EST = ["Centro de Proceso", "Servidores", "Te (h)", "Capacidad (u/h)",
            "Utilización (%)", "QT medio (h)", "SCV_e"]


def tabla_est_conwip(conwip, cfg, cap, scv_arrival):
    """Tabla de centros bajo régimen CONWIP.
    Tanto la utilización como el QT medio se evalúan al Throughput REAL del
    régimen CONWIP (TH_CONWIP), NO a la demanda externa d, ya que el WIP finito
    fija el caudal que realmente circula por la línea:
        util_conwip_i = (TH_CONWIP / cap_i) · 100
        qt_conwip_i   = VUT (Kingman-Sakasegawa) evaluado a TH_CONWIP"""
    th_conwip = conwip["TH"]
    qt_conwip, _ = calcular_qt_analitico_conwip(cfg["procesos"], th_conwip, scv_arrival)
    filas = []
    for i, p in enumerate(cfg["procesos"]):
        filas.append({
            COLS_EST[0]: NOMBRES_CORTOS[i],
            COLS_EST[1]: p["servers"],
            COLS_EST[2]: f"{p['Te']:.2f}",
            COLS_EST[3]: f"{cap[i]:.2f}",
            COLS_EST[4]: f"{th_conwip / cap[i] * 100.0:.2f}",
            COLS_EST[5]: f"{qt_conwip[i]:.3f}",
            COLS_EST[6]: f"{p['scv']:.1f}",
        })
    return pd.DataFrame(filas)


def tabla_est_push(push, cfg, cap):
    """Tabla de centros bajo régimen PUSH: la utilización es la MEDIDA por la
    DES (ocupación real del servidor) y el QT es el observado en la simulación."""
    util_push_sim = push["util_push_sim"]
    filas = []
    for i, p in enumerate(cfg["procesos"]):
        filas.append({
            COLS_EST[0]: NOMBRES_CORTOS[i],
            COLS_EST[1]: p["servers"],
            COLS_EST[2]: f"{p['Te']:.2f}",
            COLS_EST[3]: f"{cap[i]:.2f}",
            COLS_EST[4]: f"{util_push_sim[i]:.2f}",
            COLS_EST[5]: f"{push['QT_estacion'][i]:.3f}",
            COLS_EST[6]: f"{p['scv']:.1f}",
        })
    return pd.DataFrame(filas)


# ============================================================================
#  G)  ENCABEZADO + EJECUCIÓN
# ============================================================================
st.markdown(
    "<div class='po-title'>Production Optimizer — Soil Nailing / Pernos de Anclaje</div>"
    "<div class='po-sub'>Análisis de sistemas de producción bajo Factory Physics "
    "(Hopp &amp; Spearman) con simulación de eventos discretos (SimPy) · "
    "descomposición del Cycle Time</div>",
    unsafe_allow_html=True,
)

cfg, calcular = leer_config_sidebar()

if "res" not in st.session_state:
    st.session_state.res = None
if "conwip_aplicado" not in st.session_state:
    st.session_state.conwip_aplicado = CONWIP_EVAL_DEFAULT

if calcular:
    try:
        with st.spinner(f"Simulando {cfg['n_iter']} pernos en régimen PUSH…"):
            st.session_state.res = ejecutar_analisis(cfg)
    except ValueError as e:
        st.session_state.res = None
        st.error(str(e))
    except Exception as e:                                  # errores del motor
        st.session_state.res = None
        st.error(f"Error de cálculo: {e}")

res = st.session_state.res

if res is not None:
    nuc = res["nucleo"]; push = res["push"]
    st.success(
        f"Cálculo completado.  BNR = {nuc['BNR']:.2f} u/h  ·  "
        f"MINWIP = {nuc['MINWIP']:.2f} u  ·  "
        f"PUSHWIP = {push['PUSHWIP_total']:.2f} u  ·  "
        f"CT PUSH = {push['CT_push_total']:.3f} h  ·  "
        f"{push['n_completados']} pernos en estado estable."
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "Vista 1 · Flow Analysis",
    "Vista 2 · Capacity Utilization",
    "Vista 3 · Métricas OS y Centros",
    "Vista 4 · Cycle Time Analysis",
])

AVISO = "Configure los parámetros en el panel lateral y presione «Calcular y Simular»."


# ------------------------------------------------------- VISTA 1: FLOW ANALYSIS
with tab1:
    if res is None:
        st.info(AVISO)
    else:
        fig = fig_flow(res)
        st.pyplot(fig, width="stretch")
        c1, c2 = st.columns([1, 3])
        with c1:
            descargar_png(fig, "vista1_flow_analysis.png")
        with st.expander("Valores tabulados de las curvas (Best / PWC)"):
            c = res["curvas"]
            df_curvas = pd.DataFrame({
                "WIP (u)": c["wip"].astype(int),
                "Best TH (u/h)": np.round(c["best_th"], 4),
                "PWC TH (u/h)": np.round(c["pwc_th"], 4),
                "Best CT (h)": np.round(c["best_ct"], 4),
                "PWC CT (h)": np.round(c["pwc_ct"], 4),
            })
            st.dataframe(df_curvas, width="stretch", hide_index=True, height=320)
            st.download_button("Descargar curvas (CSV)",
                               data=df_curvas.to_csv(index=False).encode("utf-8"),
                               file_name="curvas_flujo.csv", mime="text/csv")


# ------------------------------------------------- VISTA 2: CAPACITY UTILIZATION
with tab2:
    if res is None:
        st.info(AVISO)
    else:
        fig2 = fig_util(res)
        st.pyplot(fig2, width="stretch")
        c1, c2 = st.columns([1, 3])
        with c1:
            descargar_png(fig2, "vista2_capacity_utilization.png")
        idx = res["nucleo"]["idx_bottleneck"]
        st.markdown(
            f"<div class='po-note'>Cuello de botella: <b>{NOMBRES_CORTOS[idx]}</b> "
            f"(capacidad {res['cap'][idx]:.2f} u/h = BNR). La utilización se evalúa "
            f"a la demanda objetivo d = {res['cfg']['d']:.2f} u/h.</div>",
            unsafe_allow_html=True,
        )


# ----------------------------------------------- VISTA 3: MÉTRICAS OS Y CENTROS
with tab3:
    if res is None:
        st.info(AVISO)
    else:
        nuc = res["nucleo"]; push = res["push"]; util = res["util"]
        cfg_r = res["cfg"]; cap = res["cap"]; campo = res["campo_actual"]
        idx = nuc["idx_bottleneck"]

        # ---------------- BLOQUE 1: métricas según la demanda actual (10 tarjetas)
        st.markdown("<div class='po-block'>Métricas según la demanda actual</div>",
                    unsafe_allow_html=True)

        util_bn = float(util[idx])                       # u_bn = d / BNR · 100 %
        min_ct = nuc["MINWIP"] / cfg_r["d"]              # MIN CT = MINWIP / d

        fila1 = [
            ("Throughput actual (TH_act)", f"{campo['TH_act']:.2f} u/h"),
            ("CT actual · Ley de Little",  f"{campo['CT_act']:.3f} h"),
            ("Demanda objetivo (d)",       f"{cfg_r['d']:.2f} u/h"),
            ("Bottleneck Rate (BNR)",      f"{nuc['BNR']:.2f} u/h"),
        ]
        fila2 = [
            ("Raw Process Time (RPT)",     f"{nuc['RPT']:.3f} h"),
            ("MIN WIP",                    f"{nuc['MINWIP']:.2f} u"),
            ("PUSH WIP (simulado)",        f"{push['PUSHWIP_total']:.2f} u"),
            ("PUSH Cycle Time (simulado)", f"{push['CT_push_total']:.3f} h"),
        ]
        fila3 = [
            ("Bottleneck Utilization",     f"{util_bn:.1f} %"),
            ("MIN Cycle Time (MIN CT)",    f"{min_ct:.3f} h"),
        ]
        for fila in (fila1, fila2):
            cols = st.columns(4)
            for col, (titulo, valor) in zip(cols, fila):
                col.metric(titulo, valor)
        cols = st.columns(4)
        for col, (titulo, valor) in zip(cols[1:3], fila3):
            col.metric(titulo, valor)

        # ---------------- BLOQUE 2: selector y rango operativo de CONWIP
        st.markdown("<div class='po-block'>Selector de CONWIP</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='po-note'>Rango sugerido: MINWIP ({nuc['MINWIP']:.2f}) "
            f"≤ CONWIP ≤ PUSHWIP ({push['PUSHWIP_total']:.2f})</div>",
            unsafe_allow_html=True,
        )
        cs1, cs2, cs3 = st.columns([1.2, 1.4, 3])
        with cs1:
            conwip_in = st.number_input("Nivel de CONWIP a evaluar (u)", min_value=1,
                                        value=int(st.session_state.conwip_aplicado),
                                        step=1, key="conwip_input")
        with cs2:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            if st.button("Recalcular CONWIP", width="stretch"):
                st.session_state.conwip_aplicado = int(conwip_in)
        with cs3:
            st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='po-note'>Nivel aplicado: "
                f"<b>{int(st.session_state.conwip_aplicado)} u</b> — recálculo "
                f"analítico vía PWC, sin volver a simular.</div>",
                unsafe_allow_html=True,
            )

        # CONWIP evaluado al nivel aplicado (mismo comportamiento del ejecutable:
        # el valor solo se actualiza al presionar «Recalcular CONWIP»).
        conwip_eval = metricas_conwip(int(st.session_state.conwip_aplicado),
                                      nuc["RPT"], nuc["BNR"])

        # ---------------- BLOQUE 3: tarjetas basadas en el nivel de CONWIP
        st.markdown("<div class='po-block'>Basado en el nivel de CONWIP</div>",
                    unsafe_allow_html=True)
        util_bn_conwip = (conwip_eval["TH"] / nuc["BNR"]) * 100.0
        b3 = [
            ("Throughput (TH CONWIP)", f"{conwip_eval['TH']:.3f} u/h"),
            ("Cycle Time (CT CONWIP)", f"{conwip_eval['CT']:.3f} h"),
            ("WIP Level",              f"{conwip_eval['WIP']:.2f} u"),
            ("Bottleneck Utilization", f"{util_bn_conwip:.1f} %"),
        ]
        cols = st.columns(4)
        for col, (titulo, valor) in zip(cols, b3):
            col.metric(titulo, valor)

        # ---------------- BLOQUE 4: tablas por centro de proceso
        st.markdown("<div class='po-block'>Centros de Proceso bajo régimen CONWIP "
                    "(analítico · VUT)</div>", unsafe_allow_html=True)
        df_conwip = tabla_est_conwip(conwip_eval, cfg_r, cap, cfg_r["scv_a"])
        st.dataframe(df_conwip, width="stretch", hide_index=True)

        st.markdown("<div class='po-block'>Centros de Proceso bajo régimen PUSH "
                    "(simulado · DES)</div>", unsafe_allow_html=True)
        df_push = tabla_est_push(push, cfg_r, cap)
        st.dataframe(df_push, width="stretch", hide_index=True)

        st.markdown(
            "<div class='po-note'>Nota: la tabla CONWIP usa la aproximación analítica "
            "VUT (Kingman-Sakasegawa) de Factory Physics; la tabla PUSH usa los valores "
            "medidos por la simulación DES (SimPy).</div>",
            unsafe_allow_html=True,
        )

        cd1, cd2, cd3 = st.columns([1, 1, 3])
        with cd1:
            st.download_button("Descargar tabla CONWIP (CSV)",
                               data=df_conwip.to_csv(index=False).encode("utf-8"),
                               file_name="centros_conwip.csv", mime="text/csv")
        with cd2:
            st.download_button("Descargar tabla PUSH (CSV)",
                               data=df_push.to_csv(index=False).encode("utf-8"),
                               file_name="centros_push.csv", mime="text/csv")


# ------------------------------------------------ VISTA 4: CYCLE TIME ANALYSIS
with tab4:
    if res is None:
        st.info(AVISO)
    else:
        push = res["push"]
        st.markdown("<div class='po-block'>Cycle Time Analysis (PUSH simulado)</div>",
                    unsafe_allow_html=True)
        df_ct = pd.DataFrame([{
            "Escenario": "PUSH (simulado)",
            "WIP (u)":   f"{push['PUSHWIP_total']:.3f}",
            "TH (u/h)":  f"{push['TH_push']:.3f}",
            "CT (h)":    f"{push['CT_push_total']:.3f}",
            "RPT (h)":   f"{push['RPT_sim']:.3f}",
            "QT (h)":    f"{push['QT_total']:.3f}",
            "BT (h)":    f"{push['BT_total']:.3f}",
            "MT (h)":    f"{push['MT_total']:.3f}",
            "SDT (h)":   f"{push['SDT_total']:.3f}",
        }])
        st.dataframe(df_ct, width="stretch", hide_index=True)

        st.markdown("<div class='po-block'>Percent of Cycle Time — composición del "
                    "CT simulado</div>", unsafe_allow_html=True)
        fig4 = fig_ct_percent(push)
        st.pyplot(fig4, width="stretch")

        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            descargar_png(fig4, "vista4_percent_cycle_time.png")
        with c2:
            st.download_button("Descargar tabla CT (CSV)",
                               data=df_ct.to_csv(index=False).encode("utf-8"),
                               file_name="cycle_time_analysis.csv", mime="text/csv")


st.markdown("---")
st.markdown(
    "<i>Factory Physics</i>; descomposición del Cycle Time.</div>",
    unsafe_allow_html=True,
)


# ============================================================================
#  H)  ARCHIVOS DE DESPLIEGUE (Streamlit Community Cloud)
# ============================================================================
# --- .streamlit/config.toml  (fuerza el tema claro de forma permanente) -----
#
# [theme]
# primaryColor="#2980b9"
# backgroundColor="#ffffff"
# secondaryBackgroundColor="#f4f6f7"
# textColor="#2c3e50"
# font="sans serif"
#
# --- requirements.txt --------------------------------------------------------
#
# streamlit>=1.50
# numpy>=1.26
# pandas>=2.0
# simpy>=4.1
# matplotlib>=3.8
#
# --- Estructura del repositorio ---------------------------------------------
#
# mi-repo/
# ├── app.py
# ├── requirements.txt
# └── .streamlit/
#     └── config.toml
# ============================================================================
