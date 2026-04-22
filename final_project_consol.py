import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# CHANGE FILE PATH! CODE WILL NOT WORK IF THE DATABASE FP DOES NOT MATCH
DATA_PATH = r"C:\VS projects\ds2500 final project\Yelp JSON\yelp_academic_dataset_business.json"


# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────

def load_data(path):
    """Load Yelp dataset and filter to FL/CA restaurants, downsampling FL to match CA size."""
    df = pd.read_json(path, lines=True)
    df_restaurants = df[df["categories"].str.contains("Restaurants", na=False)]

    df_fl = df_restaurants[df_restaurants["state"] == "FL"]
    df_ca = df_restaurants[df_restaurants["state"] == "CA"]

    df_fl_sampled = df_fl.sample(n=len(df_ca), random_state=42)
    df_final = pd.concat([df_fl_sampled, df_ca]).reset_index(drop=True)

    return df_final


def get_attr(attr_dict, key):
    """Retrieve a value from the attributes dictionary."""
    if isinstance(attr_dict, dict):
        return attr_dict.get(key, None)
    return None


def extract_attributes(df):
    """Pull specific attributes out of the nested attributes dict into flat columns."""
    df = df.copy()
    df["noise_level"]        = df["attributes"].apply(lambda x: get_attr(x, "NoiseLevel"))
    df["has_parking"]        = df["attributes"].apply(lambda x: get_attr(x, "BusinessParking"))
    df["price_range"]        = df["attributes"].apply(lambda x: get_attr(x, "RestaurantsPriceRange2"))
    df["takes_reservations"] = df["attributes"].apply(lambda x: get_attr(x, "RestaurantsReservations"))
    return df


# ─────────────────────────────────────────────
# KENNEITH — PRICE RANGE ANALYSIS (FL + CA)
# ─────────────────────────────────────────────

def clean_price_range(df):
    """Drop rows with missing/null price_range and cast to int."""
    df = df[df["price_range"].notna()].copy()
    df = df[df["price_range"] != "None"]
    df["price_range"] = df["price_range"].astype(int)
    df = df[df["price_range"] != 4]
    return df


def compute_price_avg(df):
    """Return average star rating per price tier per state."""
    return df.groupby(["state", "price_range"])["stars"].mean().reset_index()


def plot_price_vs_stars(price_avg, save_path="stars_vs_price.png"):
    """Bar chart of average star rating by price tier for FL vs CA."""
    fl_avg = price_avg[price_avg["state"] == "FL"]
    ca_avg = price_avg[price_avg["state"] == "CA"]

    x = [1, 2, 3]
    width = 0.35
    labels = ["fast food, diners", "casual sit-down", "nicer restaurants"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], fl_avg["stars"], width, label="FL", color="#FF6B6B")
    ax.bar([i + width / 2 for i in x], ca_avg["stars"], width, label="CA", color="#4ECDC4")

    ax.set_xlabel("Price Tier")
    ax.set_ylabel("Average Star Rating")
    ax.set_title("Average Star Rating by Price Tier: FL vs CA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def run_linear_regression(df):
    """Run linear regression (price -> stars) for each state and print results."""
    for state in ["FL", "CA"]:
        subset = df[df["state"] == state].dropna(subset=["price_range", "stars"])

        X = subset["price_range"].values.reshape(-1, 1)
        y = subset["stars"].values

        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        r2 = r2_score(y, preds)

        print(f"\n{state} Linear Regression — Price vs Stars")
        print(f"  Coefficient: {model.coef_[0]:.4f}")
        print(f"  Intercept:   {model.intercept_:.4f}")
        print(f"  R² Score:    {r2:.4f}")


# ─────────────────────────────────────────────
# KENNEITH — HOURS OF OPERATION ANALYSIS (FL + CA)
# ─────────────────────────────────────────────

def parse_hours(hours_dict):
    """Calculate average daily hours open from the hours dict."""
    if not isinstance(hours_dict, dict):
        return None

    total_hours = 0
    count = 0

    for day, hours in hours_dict.items():
        try:
            open_str, close_str = hours.split("-")

            open_h, open_m = map(int, open_str.split(":"))
            close_h, close_m = map(int, close_str.split(":"))

            open_time = open_h + open_m / 60
            close_time = close_h + close_m / 60

            if close_time < open_time:
                close_time += 24

            daily_hours = close_time - open_time
            total_hours += daily_hours
            count += 1
        except:
            continue

    return total_hours / count if count > 0 else None


def parse_total_hours(hours_dict):
    """Calculate total weekly hours open by summing across all days."""
    if not isinstance(hours_dict, dict):
        return None

    total_hours = 0
    count = 0

    for day, hours in hours_dict.items():
        try:
            open_str, close_str = hours.split("-")

            open_h, open_m = map(int, open_str.split(":"))
            close_h, close_m = map(int, close_str.split(":"))

            open_time = open_h + open_m / 60
            close_time = close_h + close_m / 60

            if close_time < open_time:
                close_time += 24

            total_hours += close_time - open_time
            count += 1
        except:
            continue

    return total_hours if count > 0 else None


def clean_hours(df):
    """Drop rows with missing avg_daily_hours and bin into Short/Medium/Long."""
    df = df.copy()
    df["avg_daily_hours"]    = df["hours"].apply(parse_hours)
    df["total_weekly_hours"] = df["hours"].apply(parse_total_hours)
    df = df.dropna(subset=["avg_daily_hours"])

    bins = [0, 8, 12, 24]
    labels = ["Short (<8h)", "Medium (8-12h)", "Long (12h+)"]
    df["hours_bin"] = pd.cut(df["avg_daily_hours"], bins=bins, labels=labels)

    return df


def compute_hours_avg(df):
    """Return average star rating per hours bin per state."""
    return df.groupby(["state", "hours_bin"], observed=True)["stars"].mean().reset_index()


def plot_hours_vs_stars(hours_avg, save_path="stars_vs_hours.png"):
    """Bar chart of average star rating by hours bin for FL vs CA."""
    fl_avg = hours_avg[hours_avg["state"] == "FL"]
    ca_avg = hours_avg[hours_avg["state"] == "CA"]

    x = [1, 2, 3]
    width = 0.35
    labels = ["Short (<8h)", "Medium (8-12h)", "Long (12h+)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], fl_avg["stars"], width, label="FL", color="#FF6B6B")
    ax.bar([i + width / 2 for i in x], ca_avg["stars"], width, label="CA", color="#4ECDC4")

    ax.set_xlabel("Hours of Operation")
    ax.set_ylabel("Average Star Rating")
    ax.set_title("Average Star Rating by Hours of Operation: FL vs CA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def plot_weekly_hours_distribution(df, save_path="weekly_hours_distribution.png"):
    """Side-by-side histograms of total weekly hours open for FL and CA."""
    fl_hours = df[df["state"] == "FL"]["total_weekly_hours"].dropna()
    ca_hours = df[df["state"] == "CA"]["total_weekly_hours"].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax1.hist(fl_hours, bins=20, color="#FF6B6B", edgecolor="black")
    ax1.set_title("FL Total Weekly Hours Distribution")
    ax1.set_xlabel("Total Weekly Hours Open")
    ax1.set_ylabel("Number of Restaurants")

    ax2.hist(ca_hours, bins=20, color="#4ECDC4", edgecolor="black")
    ax2.set_title("CA Total Weekly Hours Distribution")
    ax2.set_xlabel("Total Weekly Hours Open")

    plt.suptitle("Total Weekly Hours Open: FL vs CA", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def run_hours_regression(df):
    """Run linear regression (avg_daily_hours -> stars) for each state and print results."""
    for state in ["FL", "CA"]:
        subset = df[df["state"] == state].dropna(subset=["avg_daily_hours", "stars"])

        X = subset["avg_daily_hours"].values.reshape(-1, 1)
        y = subset["stars"].values

        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        r2 = r2_score(y, preds)

        print(f"\n{state} Linear Regression — Hours vs Stars")
        print(f"  Coefficient: {model.coef_[0]:.4f}")
        print(f"  Intercept:   {model.intercept_:.4f}")
        print(f"  R² Score:    {r2:.4f}")


def calc_total_weekly_hours(hours_dict):
    """Sum total hours open across all days in a week."""
    if not isinstance(hours_dict, dict):
        return None

    total = 0
    for day, hours in hours_dict.items():
        try:
            open_str, close_str = hours.split("-")

            open_h, open_m = map(int, open_str.split(":"))
            close_h, close_m = map(int, close_str.split(":"))

            open_time = open_h + open_m / 60
            close_time = close_h + close_m / 60

            if close_time < open_time:
                close_time += 24

            total += close_time - open_time
        except:
            continue

    return total if total > 0 else None


def plot_weekly_hours_distribution(df, save_path="weekly_hours_distribution.png"):
    """Side-by-side histograms of total weekly hours open for FL vs CA."""
    df = df.copy()
    df["total_weekly_hours"] = df["hours"].apply(calc_total_weekly_hours)

    fl_hours = df[df["state"] == "FL"]["total_weekly_hours"].dropna()
    ca_hours = df[df["state"] == "CA"]["total_weekly_hours"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].hist(fl_hours, bins=20, color="#FF6B6B", edgecolor="black")
    axes[0].set_title("FL Total Weekly Hours Distribution")
    axes[0].set_xlabel("Total Weekly Hours Open")
    axes[0].set_ylabel("Number of Restaurants")

    axes[1].hist(ca_hours, bins=20, color="#4ECDC4", edgecolor="black")
    axes[1].set_title("CA Total Weekly Hours Distribution")
    axes[1].set_xlabel("Total Weekly Hours Open")

    plt.suptitle("Total Weekly Hours Open: FL vs CA", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


# ─────────────────────────────────────────────
# KENNEITH — DISTRIBUTION + CORRELATION PLOTS
# ─────────────────────────────────────────────

def plot_star_distribution(df, save_path="star_distribution.png"):
    """KDE plot showing star rating distribution for FL vs CA."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for state, color in zip(["FL", "CA"], ["#FF6B6B", "#4ECDC4"]):
        subset = df[df["state"] == state]["stars"]
        subset.plot.kde(ax=ax, label=state, color=color, linewidth=2)

    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Density")
    ax.set_title("Star Rating Distribution: FL vs CA")
    ax.set_xlim(0, 5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def plot_correlation_bars(df, save_path="correlation_bars_fl_ca.png"):
    """Horizontal bar chart: correlation of price, hours, review_count with stars (FL + CA)."""
    df_corr = df[["stars", "price_range", "avg_daily_hours", "review_count"]].dropna()
    corr = df_corr.corr()["stars"].drop("stars").sort_values()

    colors = ["#FF6B6B" if v < 0 else "#4ECDC4" for v in corr]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with Star Rating")
    ax.set_title("Feature Correlation with Star Rating (FL + CA)")
    ax.set_yticklabels(["Avg Daily Hours", "Review Count", "Price Tier"])
    ax.set_xlim(-0.3, 0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


# ─────────────────────────────────────────────
# EILEEN — PARKING + NOISE ANALYSIS (CA ONLY)
# ─────────────────────────────────────────────

def clean_noise(noise):
    """Clean the noise level value by removing extra characters and normalizing case."""
    if noise is None:
        return None
    cleaned = str(noise).replace("u'", "").replace("'", "").strip().lower()
    if cleaned in ("none", "nan", ""):
        return None
    return cleaned


def parse_parking(parking):
    """Return 'Yes' if any parking type is True, 'No' if all False, else None.
    Uses ast.literal_eval to safely parse stringified dicts."""
    if parking is None:
        return None
    if isinstance(parking, str):
        try:
            parking = ast.literal_eval(parking)
        except Exception:
            return None
    if isinstance(parking, dict):
        if any(str(v).lower() == "true" for v in parking.values()):
            return "Yes"
        return "No"
    return None


def add_parking_and_noise_columns(df_ca):
    """Add cleaned noise_level and has_parking columns to a CA dataframe."""
    df_ca = df_ca.copy()
    df_ca["noise_level"] = df_ca["noise_level"].apply(clean_noise)
    df_ca["has_parking"] = df_ca["has_parking"].apply(parse_parking)
    return df_ca


def number_features(df_ca):
    """Encode parking and noise as numeric values for regression."""
    parking_map = {"Yes": 1, "No": 0}
    noise_map = {"quiet": 0, "average": 1, "loud": 2, "very_loud": 3}
    df_ca = df_ca.copy()
    df_ca["parking_num"] = df_ca["has_parking"].map(parking_map)
    df_ca["noise_num"]   = df_ca["noise_level"].map(noise_map)
    return df_ca


def plot_stars_vs_parking(df_ca, save_path="stars_vs_parking_CA.png"):
    """Bar chart of average star rating by parking availability (CA only)."""
    df_parking = df_ca[df_ca["has_parking"].notna() & (df_ca["has_parking"] != "None") & (df_ca["has_parking"] != "")].copy()
    parking_avg = df_parking.groupby("has_parking")["stars"].mean().reset_index()

    plt.figure(figsize=(6, 4))
    plt.bar(parking_avg["has_parking"], parking_avg["stars"], color="#4ECDC4")
    plt.xlabel("Has Parking")
    plt.ylabel("Average Star Rating")
    plt.title("Average Star Rating for CA Restaurants: Parking vs No Parking")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def plot_stars_vs_noise(df_ca, save_path="stars_vs_noise_CA.png"):
    """Bar chart of average star rating by noise level (CA only)."""
    df_noise = df_ca[df_ca["noise_level"].notna() & (df_ca["noise_level"] != "None") & (df_ca["noise_level"] != "")].copy()
    noise_avg = df_noise.groupby("noise_level", dropna=True)["stars"].mean().reset_index()

    plt.figure(figsize=(7, 4))
    plt.bar(noise_avg["noise_level"], noise_avg["stars"], color="#4ECDC4")
    plt.xlabel("Noise Level")
    plt.ylabel("Average Star Rating")
    plt.title("Average Star Rating for CA Restaurants by Noise Level")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def parking_linear_regression(df_ca):
    """Run linear regression (parking -> stars) for CA and print results."""
    subset = df_ca.dropna(subset=["stars", "parking_num"])

    X = subset["parking_num"].values.reshape(-1, 1)
    y = subset["stars"].values

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    print("\nCA Linear Regression — Parking vs Stars")
    print(f"  Coefficient: {model.coef_[0]:.4f}")
    print(f"  Intercept:   {model.intercept_:.4f}")
    print(f"  R² Score:    {r2:.4f}")

    return model


def noise_linear_regression(df_ca):
    """Run linear regression (noise level -> stars) for CA and print results."""
    subset = df_ca.dropna(subset=["noise_num", "stars"])

    X = subset["noise_num"].values.reshape(-1, 1)
    y = subset["stars"].values

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    print("\nCA Linear Regression — Noise Level vs Stars")
    print(f"  Coefficient: {model.coef_[0]:.4f}")
    print(f"  Intercept:   {model.intercept_:.4f}")
    print(f"  R² Score:    {r2:.4f}")

    return model


def plot_correlation_bars_ca(df_ca, save_path="correlation_bars_CA.png"):
    """Horizontal bar chart: correlation of parking, noise, review_count with stars (CA only)."""
    df_corr = df_ca[["stars", "review_count", "parking_num", "noise_num"]].dropna()
    corr = df_corr.corr()["stars"].drop("stars").sort_values()

    colors = ["#FF6B6B" if v < 0 else "#4ECDC4" for v in corr]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with Star Rating")
    ax.set_title("Feature Correlation with Star Rating (CA Only)")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_xlim(-0.3, 0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


# ─────────────────────────────────────────────
# KHAI — PARKING + NOISE ANALYSIS (FL ONLY)
# ─────────────────────────────────────────────

def plot_stars_vs_parking_fl(df_fl, save_path="stars_vs_parking_FL.png"):
    """Bar chart of average star rating by parking availability (FL only)."""
    df_parking = df_fl[df_fl["has_parking"].notna() & (df_fl["has_parking"] != "None") & (df_fl["has_parking"] != "")].copy()
    parking_avg = df_parking.groupby("has_parking")["stars"].mean().reset_index()

    plt.figure(figsize=(6, 4))
    plt.bar(parking_avg["has_parking"], parking_avg["stars"], color="#FF6B6B")
    plt.xlabel("Has Parking")
    plt.ylabel("Average Star Rating")
    plt.title("Average Star Rating for FL Restaurants: Parking vs No Parking")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def plot_stars_vs_noise_fl(df_fl, save_path="stars_vs_noise_FL.png"):
    """Bar chart of average star rating by noise level (FL only)."""
    df_noise = df_fl[df_fl["noise_level"].notna() & (df_fl["noise_level"] != "None") & (df_fl["noise_level"] != "")].copy()
    noise_avg = df_noise.groupby("noise_level", dropna=True)["stars"].mean().reset_index()

    plt.figure(figsize=(7, 4))
    plt.bar(noise_avg["noise_level"], noise_avg["stars"], color="#FF6B6B")
    plt.xlabel("Noise Level")
    plt.ylabel("Average Star Rating")
    plt.title("Average Star Rating for FL Restaurants by Noise Level")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def parking_linear_regression_fl(df_fl):
    """Run linear regression (parking -> stars) for FL and print results."""
    subset = df_fl.dropna(subset=["stars", "parking_num"])

    X = subset["parking_num"].values.reshape(-1, 1)
    y = subset["stars"].values

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    print("\nFL Linear Regression — Parking vs Stars")
    print(f"  Coefficient: {model.coef_[0]:.4f}")
    print(f"  Intercept:   {model.intercept_:.4f}")
    print(f"  R² Score:    {r2:.4f}")

    return model


def noise_linear_regression_fl(df_fl):
    """Run linear regression (noise level -> stars) for FL and print results."""
    subset = df_fl.dropna(subset=["noise_num", "stars"])

    X = subset["noise_num"].values.reshape(-1, 1)
    y = subset["stars"].values

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    print("\nFL Linear Regression — Noise Level vs Stars")
    print(f"  Coefficient: {model.coef_[0]:.4f}")
    print(f"  Intercept:   {model.intercept_:.4f}")
    print(f"  R² Score:    {r2:.4f}")

    return model


def plot_correlation_bars_fl(df_fl, save_path="correlation_bars_FL.png"):
    """Horizontal bar chart: correlation of parking, noise, review_count with stars (FL only)."""
    df_corr = df_fl[["stars", "review_count", "parking_num", "noise_num"]].dropna()
    corr = df_corr.corr()["stars"].drop("stars").sort_values()

    colors = ["#FF6B6B" if v < 0 else "#4ECDC4" for v in corr]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with Star Rating")
    ax.set_title("Feature Correlation with Star Rating (FL Only)")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_xlim(-0.3, 0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load & prep ──────────────────────────
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(df["state"].value_counts())
    print(df.shape)

    print("\nExtracting attributes...")
    df = extract_attributes(df)

    # ── Star distribution ─────────────────────
    print("\nPlotting star distribution...")
    plot_star_distribution(df)

    # ── Price range analysis (FL + CA) ────────
    print("\nCleaning price range...")
    df_price = clean_price_range(df)

    print("\nAverage stars by price tier:")
    price_avg = compute_price_avg(df_price)
    print(price_avg)

    print("\nPlotting price vs stars...")
    plot_price_vs_stars(price_avg)

    print("\nRunning price linear regression...")
    run_linear_regression(df_price)
    # Positive correlation between price_range and star rating, but relationship is weak.

    # ── Hours analysis (FL + CA) ──────────────
    print("\nCleaning hours and binning...")
    df_hours = clean_hours(df)

    print("\nAverage stars by hours bin:")
    hours_avg = compute_hours_avg(df_hours)
    print(hours_avg)

    print("\nPlotting hours vs stars...")
    plot_hours_vs_stars(hours_avg)

    print("\nPlotting weekly hours distribution...")
    plot_weekly_hours_distribution(df_hours)

    print("\nRunning hours linear regression...")
    run_hours_regression(df_hours)
    # Longer hours correlates with lower ratings in both states.
    # Hours is slightly stronger predictor than price (R² ~0.05-0.09) but still weak overall.

    # ── FL + CA correlation bars ──────────────
    print("\nPlotting FL+CA correlation bars...")
    plot_correlation_bars(df_hours)

    # ── Eileen: parking + noise analysis (CA only) ──
    print("\nPreparing CA-only data for parking and noise analysis...")
    df_ca = df[df["state"] == "CA"].copy()
    df_ca = add_parking_and_noise_columns(df_ca)
    df_ca = number_features(df_ca)

    print("\nPlotting stars vs parking (CA)...")
    plot_stars_vs_parking(df_ca)

    print("\nPlotting stars vs noise (CA)...")
    plot_stars_vs_noise(df_ca)

    print("\nRunning parking linear regression (CA)...")
    parking_linear_regression(df_ca)

    print("\nRunning noise linear regression (CA)...")
    noise_linear_regression(df_ca)

    print("\nPlotting CA correlation bars...")
    plot_correlation_bars_ca(df_ca)

    # ── Khai: parking + noise analysis (FL only) ──
    print("\nPreparing FL-only data for parking and noise analysis...")
    df_fl = df[df["state"] == "FL"].copy()
    df_fl = add_parking_and_noise_columns(df_fl)
    df_fl = number_features(df_fl)

    print("\nPlotting stars vs parking (FL)...")
    plot_stars_vs_parking_fl(df_fl)

    print("\nPlotting stars vs noise (FL)...")
    plot_stars_vs_noise_fl(df_fl)

    print("\nRunning parking linear regression (FL)...")
    parking_linear_regression_fl(df_fl)

    print("\nRunning noise linear regression (FL)...")
    noise_linear_regression_fl(df_fl)

    print("\nPlotting FL correlation bars...")
    plot_correlation_bars_fl(df_fl)


if __name__ == "__main__":
    main()
