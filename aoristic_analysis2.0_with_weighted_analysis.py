import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vehicle_theft_analysis.log'),
        logging.StreamHandler()
    ]
)

# ----------------------------------------------------------------------
# 1. Helper: smart formatting (removes .0)
# ----------------------------------------------------------------------
def smart_format(val):
    """Show .1f only when there is a fractional part, otherwise integer."""
    if isinstance(val, (int, float)):
        if val == int(val):
            return f"{int(val)}"
        else:
            return f"{val:.1f}"
    return str(val)

# ----------------------------------------------------------------------
# 2. Date conversion
# ----------------------------------------------------------------------
def excel_serial_to_datetime(value):
    if pd.isna(value) or value == '':
        return None
    try:
        dt = pd.to_datetime(value, errors='coerce')
        if pd.isna(dt) and isinstance(value, (int, float)):
            base_date = datetime(1900, 1, 1)
            days = int(value) - 2
            seconds = (value % 1) * 86400
            dt = base_date + timedelta(days=days, seconds=seconds)
        return dt if not pd.isna(dt) else None
    except Exception as e:
        logging.error(f"Error converting datetime: {e}")
        return None

# ----------------------------------------------------------------------
# 3. Standardise makes
# ----------------------------------------------------------------------
def standardize_make(make):
    if pd.isna(make) or make == '':
        return 'Unknown'
    make = make.strip().lower()
    make_map = {
        'chevy': 'Chevrolet', 'chevrolet': 'Chevrolet',
        'ford': 'Ford', 'Ford ': 'Ford',
        'toyota': 'Toyota',
        'honda': 'Honda', 'Honda ': 'Honda',
        'nissan': 'Nissan',
        'dodge': 'Dodge',
        'chrysler': 'Chrysler',
        'jeep': 'Jeep', 'Jeep ': 'Jeep',
        'gmc': 'GMC',
        'hyundai': 'Hyundai', 'Hyundai ': 'Hyundai',
        'kia': 'Kia', 'Kia ': 'Kia',
        'benz': 'Mercedes benz',
        'Mercedes': 'Mercedes benz',
        'Chrysler ': 'Chrysler',
        'Mitsubishi ': 'Mitsubishi',
        'Uhaul': 'U-Haul', 'U-Haul ': 'U-Haul',
        'Volkswagen ': 'Volkswagen'
    }
    return make_map.get(make, make.capitalize())

# ----------------------------------------------------------------------
# 4. Aoristic core
# ----------------------------------------------------------------------
def perform_aoristic_analysis(df, start_col, end_col, weight_col=None):
    weights = {(dow, h): 0 for dow in range(7) for h in range(24)}
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    record_weights = []

    if weight_col:
        weight_values = pd.to_numeric(df[weight_col], errors='coerce').fillna(0).tolist()
    else:
        weight_values = [1.0] * len(df)

    for i, row in enumerate(df.itertuples(index=False)):
        start = excel_serial_to_datetime(getattr(row, start_col))
        end = excel_serial_to_datetime(getattr(row, end_col)) if end_col else None
        record_weight = {f'weight_{day_labels[dow]}_h{h}': 0 for dow in range(7) for h in range(24)}
        w = weight_values[i]

        if start is None:
            record_weights.append(record_weight)
            continue

        if end is None or start == end:
            dow, hour = start.weekday(), start.hour
            weights[(dow, hour)] += w
            record_weight[f'weight_{day_labels[dow]}_h{hour}'] = w
        else:
            duration = (end - start).total_seconds() / 3600
            if duration <= 0:
                record_weights.append(record_weight)
                continue

            current = start.replace(minute=0, second=0, microsecond=0)
            hours_counted = 0
            while current <= end:
                if current == end.replace(minute=0, microsecond=0) and end.minute < 30:
                    break
                hours_counted += 1
                current += timedelta(hours=1)

            if hours_counted == 0:
                record_weights.append(record_weight)
                continue

            weight_per_hour = w / hours_counted
            current = start.replace(minute=0, second=0, microsecond=0)
            while current <= end:
                if current == end.replace(minute=0, microsecond=0) and end.minute < 30:
                    break
                dow, hour = current.weekday(), current.hour
                weights[(dow, hour)] += weight_per_hour
                record_weight[f'weight_{day_labels[dow]}_h{hour}'] += weight_per_hour
                current += timedelta(hours=1)

        record_weights.append(record_weight)

    aoristic_df = pd.DataFrame([
        {'day': dow, 'hour': hour, 'weight': round(weight, 2)}
        for (dow, hour), weight in weights.items()
    ])
    weights_df = pd.DataFrame(record_weights)
    weights_df.index = df.index
    return aoristic_df, weights_df

# ----------------------------------------------------------------------
# 5. Heatmap with totals – **file name as subtitle**
# ----------------------------------------------------------------------
def create_heatmap_with_totals(data, filters, output_file, input_filename=None):
    if data['weight'].sum() == 0:
        return False

    # ----- subtitle = file name without extension -----
    subtitle = (os.path.splitext(os.path.basename(input_filename))[0]
                if input_filename else "Data Only")

    # ----- filter description -----
    active_filters = []
    for key, values in filters.items():
        if 'All' not in values and values:
            selected = ', '.join(values)
            active_filters.append(f"{key}: {selected}")
    filter_text = '; '.join(active_filters) if active_filters else "No Filters"

    # ----- pivot -----
    pivot_full = data.pivot(index='day', columns='hour', values='weight').fillna(0)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot_full.index = [day_labels[d] for d in pivot_full.index]

    pivot_data = pivot_full.copy()
    hour_totals = pivot_data.sum(axis=0)
    day_totals = pivot_data.sum(axis=1)
    grand_total = hour_totals.sum()

    pivot_display = pivot_data.copy()
    pivot_display.loc['TOTAL'] = hour_totals
    pivot_display['TOTAL'] = day_totals
    pivot_display.loc['TOTAL', 'TOTAL'] = grand_total

    # ----- string annotations (no .0) -----
    annot_data = pivot_display.astype(float).applymap(smart_format)

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(16, 7.5))
    sns.heatmap(
        pivot_display.iloc[:7, :24],
        cmap='RdYlGn_r',
        linewidths=0.5,
        annot=annot_data.iloc[:7, :24].values,
        fmt="",
        ax=ax,
        cbar_kws={'label': 'Aoristic Weight'},
        annot_kws={'fontsize': 7.5, 'fontweight': 'bold', 'color': 'black'}
    )

    # ----- TOTAL row (bottom) -----
    for col_idx, hour in enumerate(pivot_display.columns[:24]):
        val = pivot_display.loc['TOTAL', hour]
        ax.text(
            col_idx + 0.5, 7.5, smart_format(val),
            ha='center', va='center', rotation=90,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5),
            fontsize=7.5, fontweight='bold'
        )

    # ----- TOTAL column (right) -----
    for row_idx, day in enumerate(pivot_display.index[:7]):
        val = pivot_display.loc[day, 'TOTAL']
        ax.text(
            24.8, row_idx + 0.5, smart_format(val),
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5),
            fontsize=7.5, fontweight='bold'
        )

    # ----- Grand total (corner) -----
    ax.text(
        24.8, 7.5, smart_format(grand_total),
        ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#d0d0d0', edgecolor='black', linewidth=1),
        fontsize=9, fontweight='bold', color='black'
    )

    # ----- axes & title -----
    ax.set_xlim(0, 25)
    ax.set_ylim(8, 0)
    ax.axhline(7, color='black', linewidth=1.5)
    ax.axvline(24, color='black', linewidth=1.5)

    ax.set_title(
        f"Aoristic Heatmap ({subtitle})\n{filter_text}",
        fontsize=14, pad=20
    )
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return True

# ----------------------------------------------------------------------
# 6. Filtering & saving
# ----------------------------------------------------------------------
def filter_data(df, filters):
    try:
        filtered = df.copy()
        for key, values in filters.items():
            if key in df.columns and values and 'All' not in values:
                selected = [v.lower() for v in values]
                filtered = filtered[df[key].fillna('').str.lower().isin(selected)]
        return filtered
    except Exception as e:
        logging.error(f"Data filtering failed: {e}")
        return df

def save_extended_dataset(original_df, weights_df, output_file):
    try:
        extended_df = pd.concat([original_df, weights_df], axis=1)
        extended_df.to_excel(output_file, index=False)
        logging.info(f"Extended dataset saved as '{output_file}'")
    except Exception as e:
        logging.error(f"Failed to save extended dataset: {e}")

def create_output_folder():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"Aoristic_Outputs_{timestamp}"
    folder_path = os.path.join(desktop, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# ----------------------------------------------------------------------
# 7. GUI
# ----------------------------------------------------------------------
class AoristicAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Aoristic Analysis of Vehicle Thefts")
        self.root.geometry("1000x800")
        self.df = None
        self.time_columns = []
        self.filter_columns = []
        self.filter_widgets = {}
        self.filter_options = {}

        # ----- 1. File -----
        tk.Label(root, text="Input CSV or Excel File:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.file_entry = tk.Entry(root, width=70)
        self.file_entry.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        tk.Button(root, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=10, pady=5)

        # ----- 2. Time columns -----
        tk.Label(root, text="Start Time Column:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.start_var = tk.StringVar()
        self.start_dropdown = ttk.Combobox(root, textvariable=self.start_var, state='disabled', width=40)
        self.start_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky='ew')

        tk.Label(root, text="End Time Column (optional):").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.end_var = tk.StringVar()
        self.end_dropdown = ttk.Combobox(root, textvariable=self.end_var, state='disabled', width=40)
        self.end_dropdown.grid(row=2, column=1, padx=10, pady=5, sticky='ew')

        # ----- 3. Weighted -----
        self.weighted_var = tk.BooleanVar()
        self.weighted_check = tk.Checkbutton(
            root, text="Use Weighted Aoristic (sum of selected column)", variable=self.weighted_var,
            command=self.toggle_weight_column
        )
        self.weighted_check.grid(row=3, column=0, columnspan=2, padx=10, pady=8, sticky='w')

        tk.Label(root, text="Weight Column:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
        self.weight_var = tk.StringVar()
        self.weight_dropdown = ttk.Combobox(root, textvariable=self.weight_var, state='disabled', width=40)
        self.weight_dropdown.grid(row=4, column=1, padx=10, pady=5, sticky='ew')

        # ----- 4. Filters (scrollable) -----
        self.filter_frame = tk.Frame(root)
        self.filter_frame.grid(row=5, column=0, columnspan=3, sticky='nsew', pady=10)
        self.root.rowconfigure(5, weight=1)
        self.root.columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(self.filter_frame)
        self.scrollbar = tk.Scrollbar(self.filter_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_inner = tk.Frame(self.canvas)

        self.scrollable_inner.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # ----- 5. Run -----
        self.run_button = tk.Button(root, text="Run Analysis", command=self.run_analysis, state='disabled')
        self.run_button.grid(row=6, column=0, columnspan=3, pady=15, sticky='ew')

    def toggle_weight_column(self):
        self.weight_dropdown['state'] = 'readonly' if self.weighted_var.get() else 'disabled'
        if not self.weighted_var.get():
            self.weight_var.set('')

    def _identify_time_columns(self, df):
        return [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'datetime', 'timestamp', 'dttm'])]

    def _identify_filter_columns(self, df, time_cols):
        return [c for c in df.columns if c not in time_cols and 0 < df[c].dropna().replace('', pd.NA).nunique() <= 30]

    def _identify_numeric_columns(self, df):
        numeric = []
        for c in df.columns:
            try:
                conv = pd.to_numeric(df[c], errors='coerce')
                if conv.notna().sum() > 0.5 * len(df):
                    numeric.append(c)
            except:
                pass
        return numeric

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV or Excel Files", "*.csv *.xlsx *.xls")])
        if not file_path:
            return
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, file_path)

        try:
            logging.info(f"Loading file: {file_path}")
            self.df = (pd.read_csv(file_path, dtype=str, na_values=['', 'NaN'])
                       if file_path.lower().endswith('.csv')
                       else pd.read_excel(file_path, dtype=str, na_values=['', 'NaN']))

            # time columns
            self.time_columns = self._identify_time_columns(self.df)
            if not self.time_columns:
                raise ValueError("No date/time columns detected.")
            self.start_dropdown['values'] = self.time_columns
            self.start_dropdown['state'] = 'readonly'
            self.end_dropdown['values'] = [''] + self.time_columns
            self.end_dropdown['state'] = 'readonly'

            # auto-select
            self.start_var.set(next((c for c in self.time_columns if 'first' in c.lower() or 'start' in c.lower()), self.time_columns[0]))
            self.end_var.set(next((c for c in self.time_columns if 'last' in c.lower() or 'end' in c.lower()), ''))

            # filters
            self.filter_columns = self._identify_filter_columns(self.df, self.time_columns)
            for w in self.scrollable_inner.winfo_children():
                w.destroy()
            self.filter_widgets.clear()
            self.filter_options.clear()

            row = 0
            for col in self.filter_columns:
                tk.Label(self.scrollable_inner, text=f"{col.replace('_', ' ')} (Ctrl+Click):", anchor='w')\
                    .grid(row=row, column=0, padx=10, pady=4, sticky='w')
                lb = tk.Listbox(self.scrollable_inner, selectmode='multiple', height=4, exportselection=0, width=50)
                lb.grid(row=row, column=1, padx=10, pady=4, sticky='ew')
                self.filter_widgets[col] = lb

                opts = ['All'] + sorted(self.df[col].dropna().replace('', pd.NA).unique().tolist())
                self.filter_options[col] = opts
                for o in opts:
                    lb.insert(tk.END, o)
                lb.select_set(0)
                row += 1
            self.scrollable_inner.grid_columnconfigure(1, weight=1)

            # weight column
            self.numeric_columns = self._identify_numeric_columns(self.df)
            if self.numeric_columns:
                self.weight_dropdown['values'] = self.numeric_columns
                if self.weighted_var.get():
                    self.weight_dropdown['state'] = 'readonly'
            else:
                self.weight_dropdown['values'] = []
                self.weight_dropdown['state'] = 'disabled'

            self.run_button['state'] = 'normal'
            messagebox.showinfo("Success", "File loaded. Filters and weight column ready.")
        except Exception as e:
            logging.error(f"Failed to load file: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.df = None
            self.run_button['state'] = 'disabled'

    def run_analysis(self):
        if self.df is None:
            messagebox.showerror("Error", "No file loaded.")
            return

        start_col = self.start_var.get()
        end_col = self.end_var.get() or None
        if not start_col:
            messagebox.showerror("Error", "Select a start time column.")
            return

        weight_col = self.weight_var.get() if self.weighted_var.get() else None
        if self.weighted_var.get() and not weight_col:
            messagebox.showerror("Error", "Select a weight column.")
            return

        try:
            df_copy = self.df.copy()
            df_copy[start_col] = df_copy[start_col].apply(excel_serial_to_datetime)
            if end_col:
                df_copy[end_col] = df_copy[end_col].apply(excel_serial_to_datetime)
                df_copy = df_copy.dropna(subset=[start_col, end_col])
            else:
                df_copy = df_copy.dropna(subset=[start_col])

            if df_copy.empty:
                messagebox.showerror("Error", "No valid time records.")
                return

            # ----- filters -----
            filters = {}
            for col, lb in self.filter_widgets.items():
                sel = [lb.get(i) for i in lb.curselection()]
                if sel and 'All' not in sel:
                    filters[col] = sel

            df_filtered = filter_data(df_copy, filters)
            if df_filtered.empty:
                messagebox.showerror("Error", "No records match filters.")
                return

            output_folder = create_output_folder()
            input_filename = self.file_entry.get()          # <-- pass to heatmap

            # ----- standard -----
            aoristic_df, weights_df = perform_aoristic_analysis(df_filtered, start_col, end_col)
            heatmap_path = os.path.join(output_folder, 'heatmap.png')
            excel_path = os.path.join(output_folder, 'extended_dataset.xlsx')
            create_heatmap_with_totals(aoristic_df, filters, heatmap_path, input_filename)
            save_extended_dataset(df_filtered, weights_df, excel_path)

            # ----- weighted -----
            if weight_col:
                df_weighted = df_filtered.copy()
                df_weighted[weight_col] = pd.to_numeric(df_weighted[weight_col], errors='coerce')
                df_weighted = df_weighted.dropna(subset=[weight_col])
                if df_weighted.empty:
                    messagebox.showwarning("Warning", f"No valid numeric values in weight column '{weight_col}' after filtering.")
                else:
                    aoristic_w_df, weights_w_df = perform_aoristic_analysis(df_weighted, start_col, end_col, weight_col=weight_col)
                    heatmap_w_path = os.path.join(output_folder, 'heatmap_weighted.png')
                    excel_w_path = os.path.join(output_folder, 'extended_weighted_dataset.xlsx')
                    create_heatmap_with_totals(aoristic_w_df, filters, heatmap_w_path, input_filename)
                    save_extended_dataset(df_weighted, weights_w_df, excel_w_path)

            # ----- success -----
            msg = "Analysis complete!\n\n"
            msg += "• Standard heatmap → heatmap.png\n"
            msg += "• Standard dataset → extended_dataset.xlsx\n"
            if weight_col and not df_weighted.empty:
                msg += "• Weighted heatmap → heatmap_weighted.png\n"
                msg += "• Weighted dataset → extended_weighted_dataset.xlsx\n"
            messagebox.showinfo("Success", msg)

        except Exception as e:
            logging.error(f"Analysis failed: {e}", exc_info=True)
            messagebox.showerror("Error", f"Analysis failed:\n{e}")

# ----------------------------------------------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = AoristicAnalysisGUI(root)
    root.mainloop()