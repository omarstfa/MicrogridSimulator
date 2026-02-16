import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def simulate_solar_generation(start_date, days):
    """Simulate solar generation for Karlsruhe location using physical PV model"""
    latitude = 49.0  # Karlsruhe latitude
    days_in_year = 365
    hours_in_day = 24
    
    # PV system parameters
    pv_capacity_kw = 20  # 10 kW PV system
    
    # New PV module parameters for physical model
    pv_area = pv_capacity_kw*6.7 # m² (typical area for 10 kW system, ~6.7 m² per kW)
    eta_ref = 0.18  # Module efficiency at STC (18%)
    gamma = -0.004  # Temperature coefficient (-0.4%/°C)
    T_ref = 25  # Reference temperature (°C)
    
    # Environmental parameters
    T_ambient_base = 10  # Base ambient temperature (°C)
    G_stc = 1000  # Standard Test Condition irradiance (W/m²)
    
    total_hours = days * 24
    time_index = pd.date_range(start=start_date, periods=total_hours, freq='h')
    
    solar_data = []
    
    for i, timestamp in enumerate(time_index):
        day_of_year = timestamp.timetuple().tm_yday
        hour_of_day = timestamp.hour
        month = timestamp.month
        
        # 1. Calculate solar position and irradiance components
        # Seasonal variation (daylight hours)
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365) * 0.5 + 0.5
        
        # Hour angle for diurnal variation
        hour_angle = (hour_of_day - 12) * 15
        
        # Calculate solar radiation (in-plane irradiance G in W/m²)
        if 6 <= hour_of_day <= 18:
            # Cosine of solar zenith angle approximation
            cos_theta = max(0, np.cos(np.radians(hour_angle)) * 0.8 + 0.2)
            # Direct normal irradiance component
            DNI = 800 * seasonal_factor * cos_theta
            # Diffuse irradiance component (simplified)
            DHI = 150 * seasonal_factor * (1 - cos_theta * 0.7)
            # Total in-plane irradiance (simplified for fixed tilt ≈ latitude)
            G = (DNI * cos_theta + DHI) * np.random.uniform(0.7, 1.0)
        else:
            G = 0
            cos_theta = 0
        
        # 2. Calculate cell temperature (simplified NOCT model)
        # Ambient temperature with seasonal variation
        T_ambient = T_ambient_base + 15 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
        # Add diurnal variation
        T_ambient += 10 * (1 - abs(hour_of_day - 14) / 8) if 6 <= hour_of_day <= 22 else -5
        
        # Cell temperature (NOCT = Nominal Operating Cell Temperature, typically 45°C)
        # Simplified: T_cell = T_ambient + (G/800) * (NOCT - 20)
        if G > 0:
            T_cell = T_ambient + (G / 800) * (45 - 20)
        else:
            T_cell = T_ambient
        
        # 3. Calculate PV power output using physical model
        if G > 0:
            # Temperature correction factor
            temp_correction = 1 + gamma * (T_cell - T_ref)
            
            # PV power output in Watts using the formula: P_mp = G * A * eta_ref * temp_correction
            P_mp = G * pv_area * eta_ref * temp_correction
            
            # Convert to kW
            pv_output = P_mp / 1000
            
            # Add inverter efficiency (95%)
            pv_output *= 0.95
            
            # Add small random noise for system variations
            pv_output += np.random.normal(0, 0.05)
        else:
            pv_output = 0
            T_cell = T_ambient
        
        # Ensure non-negative output
        pv_output = max(0, pv_output)
        
        # Clip to system capacity (rare but possible with over-irradiance)
        pv_output = min(pv_output, pv_capacity_kw * 1.1)
        
        solar_data.append({
            'timestamp': timestamp,
            'solar_irradiance_w_m2': G,
            'cell_temp_c': T_cell,
            'ambient_temp_c': T_ambient,
            'pv_generation_kw': pv_output
        })
    
    solar_df = pd.DataFrame(solar_data)
    solar_df.set_index('timestamp', inplace=True)
    
    return solar_df

if __name__ == "__main__":
    start_date = pd.Timestamp("2026-01-01 00:00:00")
    days = 365
    
    solar_df = simulate_solar_generation(start_date, days)
    
    solar_df.to_csv('solar_generation.csv')
    
    # Create a comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: PV Generation
    axes[0, 0].plot(solar_df.index, solar_df['pv_generation_kw'], 
                    label='PV Generation (kW)', color='orange', linewidth=0.5)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Power (kW)')
    axes[0, 0].set_title('Solar PV Generation in Karlsruhe (180 days)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Solar Irradiance
    axes[0, 1].plot(solar_df.index, solar_df['solar_irradiance_w_m2'], 
                    label='Solar Irradiance (W/m²)', color='gold', linewidth=0.5)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Irradiance (W/m²)')
    axes[0, 1].set_title('Solar Irradiance')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Temperatures
    axes[1, 0].plot(solar_df.index, solar_df['cell_temp_c'], 
                    label='Cell Temperature (°C)', color='red', linewidth=0.5, alpha=0.7)
    axes[1, 0].plot(solar_df.index, solar_df['ambient_temp_c'], 
                    label='Ambient Temperature (°C)', color='blue', linewidth=0.5, alpha=0.7)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('PV Cell and Ambient Temperatures')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Daily generation profile (averaged)
    solar_df['hour'] = solar_df.index.hour
    daily_profile = solar_df.groupby('hour')['pv_generation_kw'].mean()
    axes[1, 1].bar(daily_profile.index, daily_profile.values, 
                   color='orange', alpha=0.7, edgecolor='darkorange')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Average Power (kW)')
    axes[1, 1].set_title('Average Daily Generation Profile')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Solar data saved to solar_generation.csv")
    print(f"Total energy generated: {solar_df['pv_generation_kw'].sum():.2f} kWh")
    print(f"Average daily generation: {solar_df['pv_generation_kw'].sum()/days:.2f} kWh/day")
    print(f"Maximum irradiance: {solar_df['solar_irradiance_w_m2'].max():.1f} W/m²")
    print(f"Maximum cell temperature: {solar_df['cell_temp_c'].max():.1f}°C")
    print(f"Average cell temperature during operation: "
          f"{solar_df.loc[solar_df['pv_generation_kw'] > 0, 'cell_temp_c'].mean():.1f}°C")