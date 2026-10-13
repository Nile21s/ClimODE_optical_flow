"""
Geschwindigkeitsfelder im ClimODE-Forward-Pass sichtbar machen.
 
Basiert auf evaluation_global.py aus dem offiziellen Repo (Aalto-QuML/ClimODE).
Für EINEN Test-Batch werden zwei Forward-Passes gerechnet:
  1. mit dem angepassten v0 (Vorfit-Datei, wie in der normalen Evaluation)
  2. mit v0 = 0
Aus beiden wird die Geschwindigkeits-Trajektorie v(t) im Solver gelesen und
als 2x2-Abbildung gespeichert:
  v0 = 0 bei 0 h | v0 = 0 bei 12 h
  v0 = 0 bei 42 h | angepasstes v0 bei 42 h
Zusätzlich werden Median/95-%-Quantil von |v| für beide Läufe ausgegeben.
 
Wichtig zur Einheit: forward() integriert über t = 0.01 * Stunden.
Die Geschwindigkeit im Solver hat daher die Einheit
  Gitterzellen pro Solver-Zeiteinheit (= 100 h).
Mit --per_hour wird durch 100 geteilt (Gitterzellen pro Stunde).
 
v(t) wird abgegriffen, indem odeint im Modul model_function kurz ersetzt
wird. Dein eigenes Attribut self.v_traj ist dafür nicht nötig.
 
Aufruf (im ClimODE-Ordner, wie evaluation_global.py):
  python evaluation_vel_traj.py \
      --checkpoint Models/Base_models/Base_model_42/ClimODE_global_euler_0_model_299.pt \
      --vel_fit initial_velocities/original_vel/original_vel_42/test_10year_2day_mm_vel.npy \
      --label "seed 42" --entry 0 --year 0 --var z --out plots/zero_fields
"""
import argparse
import os
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
LEVELS = ["z", "t", "t2m", "u10", "v10"]
TITLES = {"z": "z500", "t": "t850", "t2m": "t2m", "u10": "u10", "v10": "v10"}
 
TEXTWIDTH_IN = 6.496  # 165 mm, \textwidth der scrreprt-Seite
 
# Deine Thesis-Konfiguration, ohne savefig.bbox = "tight"
# (das würde die Breite beim Speichern ändern und LaTeX müsste skalieren).
THESIS_STYLE = {
    "savefig.format": "pdf",
    "savefig.bbox": "standard",
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.linewidth": 0.8,
}
# Für Karten: kein Gitter, voller Rahmen
MAP_STYLE = {**THESIS_STYLE,
             "axes.grid": False,
             "axes.spines.top": True,
             "axes.spines.right": True}
 
 
# --------------------------------------------------------------------------
# Hilfsfunktionen (ohne Modellabhängigkeiten, auch einzeln importierbar)
# --------------------------------------------------------------------------
def init_date(entry, yr, first_year=2017):
    """Startzeitpunkt eines Test-Batches.
 
    Test_loader nutzt Final_test_data[2:] mit batch_size 8 (6-stündlich),
    Batch `entry` beginnt also bei Zeitindex 2 + 8*entry des Jahres.
    """
    t0 = pd.Timestamp(f"{first_year + yr}-01-01 00:00")
    return t0 + pd.Timedelta(hours=6 * (2 + 8 * entry))
 
 
def components(v, k, yr, q, out_ch, lat):
    """v_x, v_y (Nord positiv) und |v| für Zeit k, Jahr yr, Variable q.
 
    Kanäle laut pde(): 0..out_ch-1 = v_x, out_ch..2*out_ch-1 = v_y.
    v_y ist der Gradient entlang der Array-Zeilen (dim=2). Laufen die
    Breitengrade im Array nach Süden, zeigt +v_y nach Süden und wird für
    die Darstellung umgedreht.
    """
    vx = v[k, yr, q]
    vy = v[k, yr, out_ch + q] * np.sign(lat[1] - lat[0])
    return vx, vy, np.hypot(vx, vy)
 
 
def magnitude_stats(fields):
    """Median und 95-%-Quantil von |v| je Feld als DataFrame."""
    rows = []
    for name, mag in fields.items():
        rows.append({"field": name,
                     "median |v|": float(np.median(mag)),
                     "p95 |v|": float(np.percentile(mag, 95)),
                     "max |v|": float(np.max(mag))})
    return pd.DataFrame(rows).set_index("field")
 
 
def plot_v_evolution(v_zero, v_fit, lat, lon, out_ch=5, var="z", yr=0,
                     hours=(0, 12, 42), fit_hour=42, label="", date=None,
                     per_hour=False, vmax=None, textwidth_in=TEXTWIDTH_IN,
                     stride=3, show_title=False, fname=None):
    """2x2: v0 = 0 bei hours[0..2], angepasstes v0 bei fit_hour.
 
    Farbe = |v| (gemeinsame Skala), Pfeile = nur Richtung (normiert),
    Richtung in der Konvention des Repos (negative physikalische Strömung).
    Variable, Seed und Datum gehören in die Caption; mit show_title=True
    stehen sie zusätzlich als Überschrift in der Abbildung.
    """
    q = LEVELS.index(var)
    scale = 0.01 if per_hour else 1.0
    unit = (r"$|v|$ [grid cells h$^{-1}$]" if per_hour else
            r"$|v|$ [grid cells / 100 h]")
    LON, LAT = np.meshgrid(lon, lat)
 
    panels = [(v_zero, hours[0], rf"$v_0 = 0$, $t = {hours[0]}\,\mathrm{{h}}$"),
              (v_zero, hours[1], rf"$v_0 = 0$, $t = {hours[1]}\,\mathrm{{h}}$"),
              (v_zero, hours[2], rf"$v_0 = 0$, $t = {hours[2]}\,\mathrm{{h}}$"),
              (v_fit, fit_hour,  rf"fitted $v_0$, $t = {fit_hour}\,\mathrm{{h}}$")]
 
    comps = [components(v, k, yr, q, out_ch, lat) for v, k, _ in panels]
    if vmax is None:   # gemeinsame Skala: 99-%-Quantil über alle Panels
        vmax = np.percentile(np.concatenate([c[2].ravel() for c in comps]), 99)
        vmax *= scale
 
    with plt.rc_context(MAP_STYLE):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                                 figsize=(textwidth_in, textwidth_in * 0.5),
                                 layout="constrained")
        for ax, (vx, vy, mag), (_, _, title) in zip(axes.ravel(), comps, panels):
            im = ax.pcolormesh(LON, LAT, mag * scale, shading="auto",
                               cmap="viridis", vmin=0, vmax=vmax,
                               edgecolors="face", linewidth=0)  # keine Haarlinien
            n = np.where(mag > 0, mag, 1.0)
            s = (slice(None, None, stride), slice(None, None, stride))
            ax.quiver(LON[s], LAT[s], (vx / n)[s], (vy / n)[s],
                      color="white", scale=40, width=0.003,
                      headwidth=3.5, headlength=4)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_yticks([-60, 0, 60])
        for ax in axes[1]:
            ax.set_xlabel(r"Longitude")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"Latitude")
        fig.colorbar(im, ax=axes, extend="max", shrink=0.95, label=unit)
        head = f"{TITLES[var]}"
        if label:
            head += f", {label}"
        if date is not None:
            head += f", initialised {date:%Y-%m-%d %H:%M} UTC"
        if show_title:
            fig.suptitle(head)
        print(f"Für die Caption: {head}")
        if fname:
            fig.savefig(fname)          # bewusst ohne bbox_inches="tight"
    return fig
 
 
# --------------------------------------------------------------------------
# Hauptprogramm (Datenladen wie evaluation_global.py)
# --------------------------------------------------------------------------
def main():
    import torch
    from torch.utils.data import DataLoader
    import model_function as mf           # Klassen für torch.load + odeint
    import model_utils                     # noqa: F401  (für torch.load)
    from utils import (get_train_test_data_without_scales_batched,
                       add_constant_info)
 
    p = argparse.ArgumentParser("ClimODE velocity trajectory")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vel_fit", required=True,
                   help="v0-Datei aus dem Vorfit (wie in der Evaluation)")
    p.add_argument("--entry", type=int, default=0, help="Index des Test-Batches")
    p.add_argument("--year", type=int, default=0, help="0 = 2017, 1 = 2018")
    p.add_argument("--var", default="z", choices=LEVELS)
    p.add_argument("--label", default="", help="z. B. 'seed 42'")
    p.add_argument("--per_hour", action="store_true",
                   help="|v| in Gitterzellen pro Stunde statt pro 100 h")
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--textwidth_in", type=float, default=TEXTWIDTH_IN,
                   help=r"\the\textwidth / 72.27")
    p.add_argument("--show_title", action="store_true",
                   help="Variable/Seed/Datum als Überschrift in die Abbildung")
    p.add_argument("--out", default="plots/zero_fields")
    args = p.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cwd = os.getcwd()
    data_dir = '/mnt/nils'
    os.makedirs(args.out, exist_ok=True)
 
    # ---- Daten wie in evaluation_global.py -------------------------------
    train_s, val_s, test_s = slice("2006", "2016"), slice("2016", "2016"), slice("2017", "2018")
    paths = [f"{data_dir}/era5_data/geopotential_500/*.nc",
             f"{data_dir}/era5_data/temperature_850/*.nc",
             f"{data_dir}/era5_data/2m_temperature/*.nc",
             f"{data_dir}/era5_data/10m_u_component_of_wind/*.nc",
             f"{data_dir}/era5_data/10m_v_component_of_wind/*.nc"]
    const_path = [f"{data_dir}/era5_data/constants/constants_5.625deg.nc"]
 
    Final_test = None
    for idx, path in enumerate(paths):
        _, _, Test_data, time_steps, lat, lon, *_ = \
            get_train_test_data_without_scales_batched(
                path, train_s, val_s, test_s, LEVELS[idx], 0)
        Final_test = Test_data if Final_test is None else \
            torch.cat([Final_test, Test_data], dim=2)
 
    const_info, lat_map, lon_map = add_constant_info(const_path)
    H, W = Final_test.shape[3], Final_test.shape[4]
    num_years, out_ch = 2, len(paths)
    Test_loader = DataLoader(Final_test[2:], batch_size=8, shuffle=False)
    time_loader = DataLoader(time_steps[2:], batch_size=8, shuffle=False)
 
    vel_fit = torch.from_numpy(np.load(args.vel_fit))
    model = torch.load(args.checkpoint, map_location="cpu",
                       weights_only=False).to(device)
    # Hinweis: wie in der Hauptauswertung ohne model.eval()
 
    # ---- v(t) aus dem Solver abgreifen ----------------------------------
    captured = {}
    _odeint = mf.odeint
 
    def odeint_capture(*a, **kw):
        res = _odeint(*a, **kw)
        captured["res"] = res.detach()
        return res
 
    def run(time_steps, batch, v0):
        data = batch[0].to(device).view(num_years, 1, out_ch, H, W)
        model.update_param([v0.to(device), const_info.to(device),
                            lat_map.to(device), lon_map.to(device)])
        t = time_steps.float().to(device).flatten()
        mf.odeint = odeint_capture
        try:
            with torch.no_grad():
                model(t, data)
        finally:
            mf.odeint = _odeint
        # (Stunden, Jahre, 2*out_ch, H, W); Index k = k Stunden nach Start
        return captured["res"][:, :, :2 * out_ch].cpu().numpy()
 
    for entry, (ts, batch) in enumerate(zip(time_loader, Test_loader)):
        if entry != args.entry:
            continue
        v0_fit = vel_fit[entry].view(num_years, 2 * out_ch, H, W)
        v_fit = run(ts, batch, v0_fit)
        v_zero = run(ts, batch, torch.zeros_like(v0_fit))
        break
    else:
        raise ValueError(f"entry {args.entry} existiert nicht")
 
    date = init_date(args.entry, args.year)
    tag = f"{args.var}_entry{args.entry}_yr{args.year}"
    np.savez_compressed(f"{args.out}/v_traj_{tag}.npz",
                        v_fit=v_fit, v_zero=v_zero, lat=lat, lon=lon)
 
    # ---- Abbildung -------------------------------------------------------
    plot_v_evolution(v_zero, v_fit, lat, lon, out_ch=out_ch, var=args.var,
                     yr=args.year, label=args.label, date=date,
                     per_hour=args.per_hour, vmax=args.vmax,
                     textwidth_in=args.textwidth_in,
                     show_title=args.show_title,
                     fname=f"{args.out}/v_traj_{tag}.pdf")
 
    # ---- Größenordnungen vergleichen ------------------------------------
    q, yr = LEVELS.index(args.var), args.year
    fields = {
        "v0 fitted (t = 0 h)":      components(v_fit, 0, yr, q, out_ch, lat)[2],
        "fitted v0, t = 12 h":      components(v_fit, 12, yr, q, out_ch, lat)[2],
        "fitted v0, t = 42 h":      components(v_fit, 42, yr, q, out_ch, lat)[2],
        "v0 = 0, t = 12 h":         components(v_zero, 12, yr, q, out_ch, lat)[2],
        "v0 = 0, t = 42 h":         components(v_zero, 42, yr, q, out_ch, lat)[2],
    }
    stats = magnitude_stats(fields)
    print("\n|v| in Gitterzellen pro Solver-Zeiteinheit (100 h):")
    print(stats.to_string(float_format=lambda x: f"{x:.3f}"))
    stats.to_csv(f"{args.out}/v_magnitudes_{tag}.csv")
    print(f"\nGespeichert in {args.out}/")
 
 
if __name__ == "__main__":
    main()