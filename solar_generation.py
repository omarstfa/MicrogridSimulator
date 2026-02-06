import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def simulate_solar_generation(start_date, days):
    """Simulate solar generation for Karlsruhe location"""
    latitude = 49.0  # Karlsruhe latitude
    days_in_year = 365
    hours_in_day = 24
    
    # PV system parameters
    pv_capacity_kw = 10  # 10 kW PV system
    pv_efficiency = 0.18  # 18% efficiency
    inverter_efficiency = 0.95  # 95% inverter efficiency
    
    total_hours = days * 24
    time_index = pd.date_range(start=start_date, periods=total_hours, freq='H')
    
    solar_data = []
    
    for i, timestamp in enumerate(time_index):
        day_of_year = timestamp.timetuple().tm_yday
        hour_of_day = timestamp.hour
        
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365) * 0.5 + 0.5
        
        hour_angle = (hour_of_day - 12) * 15
        if 6 <= hour_of_day <= 18:
            diurnal_factor = np.cos(np.radians(hour_angle)) * 0.8 + 0.2
        else:
            diurnal_factor = 0
            
        weather_factor = np.random.uniform(0.6, 1.0)
        
        solar_radiation = 1.0 * seasonal_factor * diurnal_factor * weather_factor
        
        pv_output = pv_capacity_kw * solar_radiation * pv_efficiency * inverter_efficiency
        
        pv_output += np.random.normal(0, 0.1)
        pv_output = max(0, pv_output)
        
        solar_data.append({
            'timestamp': timestamp,
            'solar_radiation_kwh_m2': solar_radiation,
            'pv_generation_kw': pv_output
        })
    
    solar_df = pd.DataFrame(solar_data)
    solar_df.set_index('timestamp', inplace=True)
    
    return solar_df

if __name__ == "__main__":
    start_date = pd.Timestamp("2026-01-01 00:00:00")
    days = 180
    
    solar_df = simulate_solar_generation(start_date, days)
    
    solar_df.to_csv('solar_generation.csv')
    
    plt.figure(figsize=(14, 6))
    plt.plot(solar_df.index, solar_df['pv_generation_kw'], label='PV Generation (kW)', color='orange', linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.title('Solar PV Generation in Karlsruhe (180 days)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"Solar data saved to solar_generation.csv")
    print(f"Total energy generated: {solar_df['pv_generation_kw'].sum():.2f} kWh")
    print(f"Average daily generation: {solar_df['pv_generation_kw'].sum()/days:.2f} kWh/day")