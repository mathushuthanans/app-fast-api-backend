def calculate_pm25_aqi(pm25):
    if pm25 is None:
        return None
    try:
        bps = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500)
        ]
        for c_lo, c_hi, i_lo, i_hi in bps:
            if c_lo <= pm25 <= c_hi:
                return round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo)
        return None
    except (TypeError, ValueError):
        return None


def calculate_pm10_aqi(pm10):
    if pm10 is None:
        return None
    try:
        bps = [
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 604, 301, 500)
        ]
        for c_lo, c_hi, i_lo, i_hi in bps:
            if c_lo <= pm10 <= c_hi:
                return round((i_hi - i_lo) / (c_hi - c_lo) * (pm10 - c_lo) + i_lo)
        return None
    except (TypeError, ValueError):
        return None


def calculate_no2_aqi(no2):
    if not isinstance(no2, (int, float)) or no2 is None:
        return None
    try:
        bps = [
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 2049, 301, 500)
        ]
        for c_lo, c_hi, i_lo, i_hi in bps:
            if c_lo <= no2 <= c_hi:
                return round((i_hi - i_lo) / (c_hi - c_lo) * (no2 - c_lo) + i_lo)
        return None
    except (TypeError, ValueError):
        return None


def calculate_co_aqi(co):
    if co is None:
        return None
    try:
        bps = [
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 50.4, 301, 500)
        ]
        for c_lo, c_hi, i_lo, i_hi in bps:
            if c_lo <= co <= c_hi:
                return round((i_hi - i_lo) / (c_hi - c_lo) * (co - c_lo) + i_lo)
        return None
    except (TypeError, ValueError):
        return None


def calculate_o3_aqi(o3):
    if o3 is None:
        return None
    try:
        bps = [
            (0, 54, 0, 50),
            (55, 70, 51, 100),
            (71, 85, 101, 150),
            (86, 105, 151, 200),
            (106, 200, 201, 300),
            (201, 604, 301, 500)
        ]
        for c_lo, c_hi, i_lo, i_hi in bps:
            if c_lo <= o3 <= c_hi:
                return round((i_hi - i_lo) / (c_hi - c_lo) * (o3 - c_lo) + i_lo)
        return None
    except (TypeError, ValueError):
        return None


def calculate_overall_aqi(pm25, pm10, no2, co, o3):
    aqi_list = []
    
    # Calculate each AQI component with proper error handling
    try:
        if pm25 is not None:
            pm25_aqi = calculate_pm25_aqi(pm25)
            if pm25_aqi is not None:
                aqi_list.append(pm25_aqi)
        
        if pm10 is not None:
            pm10_aqi = calculate_pm10_aqi(pm10)
            if pm10_aqi is not None:
                aqi_list.append(pm10_aqi)
        
        if no2 is not None:
            no2_aqi = calculate_no2_aqi(no2)
            if no2_aqi is not None:
                aqi_list.append(no2_aqi)
        
        if co is not None:
            co_aqi = calculate_co_aqi(co)
            if co_aqi is not None:
                aqi_list.append(co_aqi)
        
        if o3 is not None:
            o3_aqi = calculate_o3_aqi(o3)
            if o3_aqi is not None:
                aqi_list.append(o3_aqi)
        
        # Return the highest AQI value if we have valid values
        if aqi_list:
            return max(aqi_list)
        return None
    except Exception as e:
        print(f"Error calculating overall AQI: {e}")
        return None